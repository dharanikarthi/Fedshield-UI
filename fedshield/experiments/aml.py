"""AMLworld experiment — the cross-bank result.

98% of laundering transactions in HI-Small cross institutions, so no single
bank sees a whole chain. We compare, on the same model class:

  A. local-only           each bank trains on its own accounts
  B. FedShield            federated GBDT across the 5 banks (SecAgg on)
  C. centralised          pooled data, privacy-free ceiling
  D. + registry (local)   B + suspect-registry features, own-bank flags only
  E. + registry (fed)     B + registry features from ALL banks' confirmed cases
  F. + live alerts        E + round-1 high-score alerts published to the registry

    python -m fedshield.experiments.aml [HI-Small]
"""
import sys
import time

import numpy as np
import pandas as pd

from fedshield.config import AML_WORK, SEED
from fedshield.eval.metrics import pick_threshold
from fedshield.experiments.common import class_weights, comparison, evaluate, run_fed, save
from fedshield.fl.federation import Bank
from fedshield.fl.gbdt import GBDTParams
from fedshield.registry import SuspectRegistry

PARAMS = GBDTParams(n_trees=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample=0.8,
                    n_bins=64, min_child_weight=5.0, seed=SEED)
NON_FEATURES = {"acct_id", "inst", "y", "pattern", "entity_id", "split"}


def make_splits(df):
    rng = np.random.RandomState(SEED)
    r = rng.rand(len(df))
    df["split"] = np.where(r < 0.7, "train", np.where(r < 0.8, "val", "test"))
    return df


def registry_features(df, legs, flagged_ids, scope, reg: SuspectRegistry, bank_names):
    """scope='local': a bank only knows its own flags; 'fed': registry across all banks."""
    reg.entries.clear()
    keys = df["acct_id"].astype(str).to_numpy()
    inst = df["inst"].to_numpy()
    for b in np.unique(inst):
        sel = np.isin(df["acct_id"].to_numpy(), flagged_ids) & (inst == b)
        reg.publish(bank_names[b], keys[sel], source="EFRM-confirmed")
    n = int(df["acct_id"].max()) + 1
    n_flag_banks = np.zeros(n, dtype=np.int16)
    flag_bank = np.full(n, -1, dtype=np.int16)
    n_flag_banks[df["acct_id"].to_numpy()] = reg.lookup(keys)
    flag_bank[df["acct_id"].to_numpy()] = inst
    hit = n_flag_banks[legs["cp"].to_numpy()] > 0
    if scope == "local":
        hit &= flag_bank[legs["cp"].to_numpy()] == legs["inst"].to_numpy()
    hit &= legs["cp"].to_numpy() != legs["acct"].to_numpy()
    lg = legs.assign(hit=hit, hit_usd=np.where(hit, legs["usd"], 0.0))
    g = lg.groupby(["acct", "dir"]).agg(n_hit=("hit", "sum"), usd_hit=("hit_usd", "sum"),
                                         usd=("usd", "sum")).unstack(fill_value=0)
    f = pd.DataFrame(index=g.index)
    for d in ("in", "out"):
        f[f"reg_cp_{d}"] = g[("n_hit", d)] if ("n_hit", d) in g else 0
        f[f"reg_usd_{d}_share"] = (g[("usd_hit", d)] / g[("usd", d)].clip(lower=1)) if ("usd_hit", d) in g else 0
    # entity-level: does the same customer (KYC identity) own another flagged account?
    ent = df[["acct_id", "entity_id", "inst"]].copy()
    ent["flag"] = np.isin(ent["acct_id"], flagged_ids)
    if scope == "local":
        grp = ent.groupby(["entity_id", "inst"])["flag"].transform("sum")
    else:
        grp = ent.groupby("entity_id")["flag"].transform("sum")
    ent["reg_entity_other"] = grp - ent["flag"].astype(int)
    out = df[["acct_id"]].merge(f, left_on="acct_id", right_index=True, how="left").fillna(0)
    out["reg_entity_other"] = ent["reg_entity_other"].to_numpy()
    return out.drop(columns="acct_id").reset_index(drop=True)


def build_banks(df, feats, names):
    X = df[feats].astype("float32").to_numpy()
    y = df["y"].to_numpy().astype(int)
    tr = (df["split"] == "train").to_numpy()
    w = np.zeros(len(df))
    w[tr] = class_weights(y[tr])
    banks, test = [], {}
    for b in sorted(df["inst"].unique()):
        m = (df["inst"] == b).to_numpy()
        va = m & (df["split"] == "val").to_numpy()
        te = m & (df["split"] == "test").to_numpy()
        banks.append(Bank(names[b], X[m & tr], y[m & tr], w[m & tr], X[va], y[va]))
        test[names[b]] = (X[te], y[te])
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    splits = {"X_val": X[va], "y_val": y[va], "X_test": X[te], "y_test": y[te]}
    return banks, test, splits


def typology_recall(model, df, feats, thr):
    te = df[(df["split"] == "test") & (df["y"] == 1)]
    p = model.predict_proba(te[feats].astype("float32").to_numpy())
    pat = te["pattern"].fillna("UNLISTED").to_numpy()
    return {k: {"n": int((pat == k).sum()), "recall": float((p[pat == k] >= thr).mean())}
            for k in np.unique(pat)}


def main(variant="HI-Small"):
    t0 = time.time()
    d = AML_WORK / variant
    df = make_splits(pd.read_parquet(d / "features.parquet"))
    names = {i: f"Bank-{chr(65 + i)}" for i in sorted(df["inst"].unique())}
    base_feats = [c for c in df.columns if c not in NON_FEATURES]
    tx = pd.read_parquet(d / "txns.parquet", columns=["src", "dst", "src_inst", "dst_inst", "usd"])
    legs = pd.concat([
        pd.DataFrame({"acct": tx.src, "cp": tx.dst, "inst": tx.src_inst, "usd": tx.usd, "dir": "out"}),
        pd.DataFrame({"acct": tx.dst, "cp": tx.src, "inst": tx.dst_inst, "usd": tx.usd, "dir": "in"})],
        ignore_index=True)
    del tx
    results = {"variant": variant, "n_accounts": len(df), "positives": int(df.y.sum()),
               "banks": {names[b]: {"accounts": int((df.inst == b).sum()), "positives": int(df[df.inst == b].y.sum())}
                         for b in names},
               "cross_bank_laundering_share": None, "experiments": {}}

    print("== A/B/C: base features")
    banks, test, splits = build_banks(df, base_feats, names)
    res, fed_base, _ = comparison(banks, PARAMS, base_feats, splits, test, "base",
                                  fed_kw={"secure_agg": True, "byzantine": True, "weighting": True})
    results["experiments"]["base"] = res
    thr_base = pick_threshold(fed_base.predict_proba(splits["X_val"]), splits["y_val"])

    reg = SuspectRegistry()
    confirmed = df.loc[(df.split == "train") & (df.y == 1), "acct_id"].to_numpy()
    for scope in ("local", "fed"):
        print(f"== registry scope={scope}")
        rf = registry_features(df, legs, confirmed, scope, reg, names)
        dfr = pd.concat([df.reset_index(drop=True), rf], axis=1)
        feats = base_feats + list(rf.columns)
        banks, test, splits = build_banks(dfr, feats, names)
        res, fed_m, _ = comparison(banks, PARAMS, feats, splits, test, f"registry-{scope}",
                                   central=(scope == "fed"),
                                   fed_kw={"secure_agg": True, "byzantine": True, "weighting": True})
        res["registry_stats"] = reg.stats()
        results["experiments"][f"registry_{scope}"] = res
        if scope == "fed":
            thr = pick_threshold(fed_m.predict_proba(splits["X_val"]), splits["y_val"])
            results["typology_recall"] = {
                "fed_base": typology_recall(fed_base, df, base_feats, thr_base),
                "fed_registry": typology_recall(fed_m, dfr, feats, thr)}
            # F: live alerts — round-1 model flags its top 0.5% unlabelled (val/test) accounts
            p_all = fed_base.predict_proba(df[base_feats].astype("float32").to_numpy())
            unl = (df.split != "train").to_numpy()
            cut = np.quantile(p_all[unl], 0.995)
            alerts = df.loc[unl & (p_all >= cut), "acct_id"].to_numpy()
            print(f"== live alerts: {len(alerts)} published")
            rf2 = registry_features(df, legs, np.concatenate([confirmed, alerts]), "fed", reg, names)
            dfr2 = pd.concat([df.reset_index(drop=True), rf2], axis=1)
            banks, test, splits = build_banks(dfr2, feats, names)
            m2, log2 = run_fed(banks, PARAMS, feats, secure_agg=True, byzantine=True, weighting=True)
            results["experiments"]["registry_fed_live_alerts"] = {
                "rows": [evaluate(m2, splits, test, "FedShield + registry + live alerts")],
                "alerts_published": int(len(alerts))}
    results["seconds"] = time.time() - t0
    print(f"saved {save(f'aml_{variant}', results)}  ({results['seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["HI-Small"]))
