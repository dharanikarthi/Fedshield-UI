"""NFPC experiment suite (the Indian mule-detection headline).

    python -m fedshield.experiments.nfpc full     # 5-year window (benchmark-comparable)
    python -m fedshield.experiments.nfpc asof     # point-in-time (deployment-honest)

Sections (each saved into results/nfpc_<mode>.json):
  main          local-only vs FedShield vs centralised (regional bank split)
  label_shield  FedShield with vs without red-herring label confidence
  registry      + cross-bank suspect registry, local scope vs consortium scope
  pattern_skew  non-IID split where banks see different typologies
  byzantine     poisoned bank (label flip / gradient scaling), gates off vs on
  dp            ε sweep
  fairness      wrongful-hold (FPR) parity across rural/PMJDY/gender/age
  seeds         seed stability of local vs federated
  windows       suspicious-window hit rate against mule_flag_date
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from fedshield.config import RESULTS, SEED, WORK
from fedshield.eval.metrics import fairness, pick_threshold, scores
from fedshield.experiments.common import (class_weights, comparison, evaluate, run_central,
                                          run_fed, run_local, save)
from fedshield.features.nfpc_master import build_master, partition, registry_features
from fedshield.fl.federation import Bank
from fedshield.fl.gbdt import GBDTParams

PARAMS = GBDTParams(n_trees=250, max_depth=6, learning_rate=0.05, subsample=0.8, colsample=0.7,
                    n_bins=64, min_child_weight=3.0, seed=SEED)
NAMES = ["Bank-A", "Bank-B", "Bank-C"]
NON_FEATURES = {"acct", "account_id", "customer_id", "is_mule", "mule_flag_date", "alert_reason",
                "flagged_by_branch", "branch_state", "label_conf", "rh_reason", "first_ts", "last_ts",
                "split", "bank", "branch_code",
                "days_since_kyc"}   # last KYC is often a post-flag re-KYC: label proxy (see experiments/leakage.py)
FED_KW = {"secure_agg": True, "byzantine": True, "weighting": True}


def splits_for(m, seed=SEED):
    rng = np.random.RandomState(seed)
    y = m["is_mule"].to_numpy()
    split = np.empty(len(m), dtype=object)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_te, n_va = int(0.2 * len(idx)), int(0.1 * len(idx))
        split[idx[:n_te]] = "test"
        split[idx[n_te:n_te + n_va]] = "val"
        split[idx[n_te + n_va:]] = "train"
    return split


def build(m, feats, bank, use_conf=True, attack=None):
    X = m[feats].astype("float32").to_numpy()
    y = m["is_mule"].to_numpy().astype(int)
    sp = m["split"].to_numpy()
    tr, va, te = sp == "train", sp == "val", sp == "test"
    conf = m["label_conf"].to_numpy() if use_conf else np.ones(len(m))
    w = np.zeros(len(m))
    w[tr] = class_weights(y[tr], conf[tr])
    banks, test = [], {}
    for b, name in enumerate(NAMES):
        mb = bank == b
        banks.append(Bank(name, X[mb & tr], y[mb & tr], w[mb & tr], X[mb & va], y[mb & va],
                          attack=(attack if (attack and name == "Bank-C") else None)))
        test[name] = (X[mb & te], y[mb & te])
    return banks, test, {"X_val": X[va], "y_val": y[va], "X_test": X[te], "y_test": y[te]}


def suspicious_windows(model, m, feats, mode, thr):
    """Window = contiguous months around the account's peak month whose volume is
    >= 40% of the peak. Hit = mule_flag_date within [start, end + 60 days]."""
    mon = pd.read_parquet(WORK / "nfpc" / f"monthly_{mode}.parquet")
    mon["vol"] = mon["credit"] + mon["debit"]
    te = m[(m["split"] == "test")].copy()
    te["p"] = model.predict_proba(te[feats].astype("float32").to_numpy())
    flagged = te[te["p"] >= thr]
    mon = mon[mon["acct"].isin(flagged["acct"])].sort_values(["acct", "month"])
    wins = {}
    for a, g in mon.groupby("acct"):
        v = g["vol"].to_numpy()
        months = g["month"].to_numpy()
        k = int(np.argmax(v))
        lo = hi = k
        while lo > 0 and v[lo - 1] >= 0.4 * v[k]:
            lo -= 1
        while hi < len(v) - 1 and v[hi + 1] >= 0.4 * v[k]:
            hi += 1
        wins[a] = (pd.Timestamp(months[lo]), pd.Timestamp(months[hi]) + pd.offsets.MonthEnd(0))
    tp = flagged[(flagged["is_mule"] == 1)]
    fd = pd.to_datetime(tp["mule_flag_date"], errors="coerce")
    hits = [wins[a][0] <= d <= wins[a][1] + pd.Timedelta(days=60)
            for a, d in zip(tp["acct"], fd) if a in wins and pd.notna(d)]
    return wins, {"flagged": int(len(flagged)), "true_mules_with_window": int(len(hits)),
                  "flag_date_in_window_rate": float(np.mean(hits)) if hits else float("nan"),
                  "median_window_days": float(np.median([(e - s).days for s, e in wins.values()])) if wins else 0}


def main(mode="full", sections=None):
    t0 = time.time()
    sections = set(sections or ["main", "label_shield", "registry", "pattern_skew", "byzantine",
                                "dp", "fairness", "seeds", "windows"])
    print(f"building master ({mode}) ...", flush=True)
    m = build_master(mode)
    m["split"] = splits_for(m)
    bank = partition(m, "state")
    m["bank"] = bank
    base_feats = [c for c in m.columns if c not in NON_FEATURES and m[c].dtype != object
                  and not np.issubdtype(m[c].dtype, np.datetime64)]
    print(f"  {len(m):,} accounts, {len(base_feats)} features, mules {int(m.is_mule.sum()):,}", flush=True)
    R = {"mode": mode, "n_accounts": int(len(m)), "n_features": len(base_feats),
         "features": base_feats,
         "banks": {n: {"accounts": int((bank == i).sum()), "mules": int(m.is_mule[bank == i].sum()),
                       "states": sorted(m.loc[bank == i, "branch_state"].dropna().unique().tolist())}
                   for i, n in enumerate(NAMES)},
         "label_shield_counts": m.loc[m.is_mule == 1, "rh_reason"].str.split("|").explode()
                                 .replace("", "clean").value_counts().to_dict()}
    results_path = RESULTS / f"nfpc_{mode}.json"

    def checkpoint():
        R["seconds"] = time.time() - t0
        save(f"nfpc_{mode}", R)

    fed_main = None
    if "main" in sections:
        print("== main", flush=True)
        banks, test, sp = build(m, base_feats, bank)
        R["main"], fed_main, cen = comparison(banks, PARAMS, base_feats, sp, test, "main", fed_kw=FED_KW)
        checkpoint()

    if "label_shield" in sections:
        print("== label shield ablation", flush=True)
        banks, test, sp = build(m, base_feats, bank, use_conf=False)
        mdl, _ = run_fed(banks, PARAMS, base_feats, **FED_KW)
        R["label_shield"] = {"without": evaluate(mdl, sp, test, "FedShield, raw labels"),
                             "with": R["main"]["rows"][-2] if "main" in R else None}
        checkpoint()

    reg_feats, fed_reg, m_reg = base_feats, None, m
    if "registry" in sections:
        confirmed = ((m["split"] == "train") & (m["is_mule"] == 1)).to_numpy()
        R["registry"] = {}
        for scope in ("local", "fed"):
            print(f"== registry {scope}", flush=True)
            rf = registry_features(m, bank, confirmed, scope, mode)
            mr = pd.concat([m.reset_index(drop=True), rf], axis=1)
            feats = base_feats + list(rf.columns)
            banks, test, sp = build(mr, feats, bank)
            res, fm, _ = comparison(banks, PARAMS, feats, sp, test, f"registry-{scope}",
                                    central=(scope == "fed"), fed_kw=FED_KW)
            R["registry"][scope] = res
            if scope == "fed":
                reg_feats, fed_reg, m_reg = feats, fm, mr
        checkpoint()

    if "pattern_skew" in sections:
        print("== pattern-skew partition", flush=True)
        pb = partition(m, "pattern")
        banks, test, sp = build(m, base_feats, pb)
        R["pattern_skew"], _, _ = comparison(banks, PARAMS, base_feats, sp, test, "pattern-skew",
                                             fed_kw=FED_KW)
        R["pattern_skew"]["banks"] = {n: int(m.is_mule[pb == i].sum()) for i, n in enumerate(NAMES)}
        checkpoint()

    if "byzantine" in sections:
        R["byzantine"] = {}
        for attack in ("label_flip", "gradient_scale"):
            for gates in (False, True):
                print(f"== byzantine {attack} gates={gates}", flush=True)
                banks, test, sp = build(m, base_feats, bank, attack=attack)
                mdl, log = run_fed(banks, PARAMS, base_feats, secure_agg=True, byzantine=gates,
                                   weighting=gates)
                r = evaluate(mdl, sp, test, f"{attack}, gates={'on' if gates else 'off'}")
                r["gate2"] = log["gate2"]
                r["gate1_rejections"] = int(sum(len(x["gate1_rejected"]) for x in log["rounds"]))
                r["accepted"] = log["config"]["accepted"]
                R["byzantine"][f"{attack}_gates_{'on' if gates else 'off'}"] = r
        checkpoint()

    if "dp" in sections:
        R["dp"] = []
        dp_params = GBDTParams(n_trees=60, max_depth=4, learning_rate=0.15, subsample=0.8,
                               colsample=0.3, n_bins=32, min_child_weight=10.0, seed=SEED)
        banks, test, sp = build(m, base_feats, bank)
        for eps in (None, 64.0, 16.0, 8.0, 4.0, 1.0):
            print(f"== dp eps={eps}", flush=True)
            mdl, log = run_fed(banks, dp_params, base_feats, secure_agg=True, byzantine=False,
                               weighting=False, dp_epsilon=eps)
            r = evaluate(mdl, sp, test, f"DP eps={eps}")
            R["dp"].append({"epsilon_target": eps, "epsilon_spent": max(log["epsilon_final"].values()),
                            "sigma": log["config"].get("dp_sigma"), **r["overall"]})
        checkpoint()

    final_model, final_feats, final_m = (fed_reg, reg_feats, m_reg) if fed_reg is not None else (fed_main, base_feats, m)
    if final_model is not None:
        va = final_m["split"].to_numpy() == "val"
        thr = pick_threshold(final_model.predict_proba(final_m.loc[va, final_feats].astype("float32").to_numpy()),
                             final_m.loc[va, "is_mule"].to_numpy())
        te = final_m["split"].to_numpy() == "test"
        p_te = final_model.predict_proba(final_m.loc[te, final_feats].astype("float32").to_numpy())
        y_te = final_m.loc[te, "is_mule"].to_numpy()
        if "fairness" in sections:
            mt = final_m[te]
            age = pd.cut(mt["customer_age_years"], [0, 25, 40, 60, 200], labels=["<25", "25-40", "40-60", "60+"])
            R["fairness"] = {
                "branch_type": fairness(p_te, y_te, thr, mt["branch_type_code"].map({0: "urban", 1: "semi-urban", 2: "rural"}).fillna("na").to_numpy()),
                "pmjdy": fairness(p_te, y_te, thr, np.where(mt["scheme_pmjdy"] == 1, "PMJDY", "other")),
                "gender": fairness(p_te, y_te, thr, np.where(mt["gender_f"] == 1, "F", "M")),
                "age": fairness(p_te, y_te, thr, age.astype(str).to_numpy()),
            }
        if "windows" in sections:
            wins, R["windows"] = suspicious_windows(final_model, final_m, final_feats, mode, thr)
        # artifacts for the API / UI
        art = {"model": final_model, "features": final_feats, "threshold": thr, "mode": mode,
               "bank_names": NAMES}
        (WORK / "models").mkdir(exist_ok=True)
        with open(WORK / "models" / f"nfpc_{mode}.pkl", "wb") as fh:
            pickle.dump(art, fh)
        cols = ["acct", "account_id", "is_mule", "bank", "split", "branch_state", "mule_flag_date",
                "alert_reason", "label_conf", "rh_reason"]
        sc = final_m[cols].copy()
        sc["score"] = final_model.predict_proba(final_m[final_feats].astype("float32").to_numpy())
        if "windows" in sections:
            sc["win_start"] = sc["acct"].map({a: w[0] for a, w in wins.items()})
            sc["win_end"] = sc["acct"].map({a: w[1] for a, w in wins.items()})
        sc.to_parquet(WORK / "models" / f"nfpc_{mode}_scores.parquet", index=False)
        final_m[["acct"] + final_feats].to_parquet(WORK / "models" / f"nfpc_{mode}_features.parquet", index=False)
        R["final_model"] = {"features": len(final_feats), "threshold": thr,
                            "test": scores(p_te, y_te, thr),
                            "top_features": sorted(zip(final_feats, final_model.feature_gain().round(4).tolist()),
                                                   key=lambda t: -t[1])[:25]}
        checkpoint()

    if "seeds" in sections:
        print("== seed stability", flush=True)
        rows = []
        small = GBDTParams(**{**PARAMS.__dict__, "n_trees": 150})
        for s in (1, 2, 3):
            ms = m.copy()
            ms["split"] = splits_for(ms, seed=s)
            banks, test, sp = build(ms, base_feats, bank)
            p = GBDTParams(**{**small.__dict__, "seed": s})
            loc = [evaluate(run_local(b, p, base_feats), {"X_val": b.X_val, "y_val": b.y_val,
                                                         "X_test": sp["X_test"], "y_test": sp["y_test"]},
                            {b.name: test[b.name]}, b.name)["per_bank"][b.name]["auprc"] for b in banks]
            fm, _ = run_fed(banks, p, base_feats, **FED_KW)
            rf = evaluate(fm, sp, test, "fed")
            rows.append({"seed": s, "local_own_mean": float(np.mean(loc)),
                         "fed_own_mean": float(np.mean([rf["per_bank"][n]["auprc"] for n in NAMES])),
                         "fed_overall": rf["overall"]["auprc"]})
            print(f"   seed {s}: {rows[-1]}", flush=True)
        R["seeds"] = rows
        checkpoint()
    print(f"saved {results_path}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full", sys.argv[2].split(",") if len(sys.argv) > 2 else None)
