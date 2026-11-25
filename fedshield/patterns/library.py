"""Layer 1 (evidence fusion) + Layer 5 (living pattern library) + Layer 6 (versioned snapshot).

Evidence feeds are SIMULATED from the NFPC labels, because no public EFRM / AML / I4C
data exists:
  EFRM  bank fraud-monitoring alert     70% of mules, 0.5% false alerts on others
  AML   suspicious-transaction report   45% of mules (80% for structuring/layering)
  I4C   cybercrime victim complaint     55% of mules, arriving 2-10 days after flag
Fusion (probabilistic union): w = 1 - prod(1 - w_source), w_EFRM=.6, w_AML=.7, w_I4C=.85.

Pattern states, per alert-reason typology:
  emerging   cases at only one bank, or test recall < 50%
  confirmed  cases at >= 2 banks and recall >= 50%
  dormant    no new case in the last 12 months of data
  retired    no new case in the last 24 months of data

    python -m fedshield.patterns.library full
"""
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd

from fedshield.config import SEED, WORK

W = {"EFRM": 0.6, "AML": 0.7, "I4C": 0.85}
STR_HEAVY = {"Structuring Transactions Below Threshold", "Layered Transaction Pattern",
             "Routine Investigation", "Rapid Movement of Funds"}


def simulate_evidence(sc):
    rng = np.random.RandomState(SEED)
    mule = sc["is_mule"].to_numpy() == 1
    n = len(sc)
    efrm = np.where(mule, rng.rand(n) < 0.70, rng.rand(n) < 0.005)
    p_aml = np.where(sc["alert_reason"].isin(STR_HEAVY), 0.8, 0.45)
    aml = mule & (rng.rand(n) < p_aml)
    i4c = mule & (rng.rand(n) < 0.55)
    fused = 1 - (1 - W["EFRM"] * efrm) * (1 - W["AML"] * aml) * (1 - W["I4C"] * i4c)
    return pd.DataFrame({"EFRM": efrm, "AML": aml, "I4C": i4c, "fused_conf": fused}, index=sc.index)


def main(mode="full"):
    mdir = WORK / "models"
    with open(mdir / f"nfpc_{mode}.pkl", "rb") as fh:
        art = pickle.load(fh)
    sc = pd.read_parquet(mdir / f"nfpc_{mode}_scores.parquet")
    ev = simulate_evidence(sc)
    sc = pd.concat([sc, ev], axis=1)
    sc["flag_date"] = pd.to_datetime(sc["mule_flag_date"], errors="coerce")
    end = pd.Timestamp("2025-06-30")
    feats = pd.read_parquet(mdir / f"nfpc_{mode}_features.parquet").set_index("acct")
    model, names, thr = art["model"], art["features"], art["threshold"]

    lib = []
    mules = sc[sc["is_mule"] == 1]
    for pat, g in mules.groupby(mules["alert_reason"].fillna("Unlabelled")):
        train = g[g["split"] == "train"]
        test = g[g["split"] == "test"]
        recall = float((test["score"] >= thr).mean()) if len(test) else float("nan")
        banks = int(train["bank"].nunique())
        last = g["flag_date"][g["flag_date"] <= end].max()
        months_quiet = (end - last).days / 30.4 if pd.notna(last) else 99
        if months_quiet > 24:
            state = "retired"
        elif months_quiet > 12:
            state = "dormant"
        elif banks >= 2 and recall >= 0.5:
            state = "confirmed"
        else:
            state = "emerging"
        sample = g["acct"].to_numpy()[:150]
        X = feats.loc[feats.index.intersection(sample), names].astype("float32").to_numpy()
        sig = []
        if len(X):
            c = model.contributions(X).mean(0)
            sig = [names[i] for i in np.argsort(-c)[:4] if c[i] > 0]
        lib.append({"pattern": pat, "state": state, "banks": banks, "cases": int(len(g)),
                    "test_cases": int(len(test)), "recall": recall,
                    "last_case": None if pd.isna(last) else last.date().isoformat(),
                    "sources": {s: int(g[s].sum()) for s in ("EFRM", "AML", "I4C")},
                    "mean_fused_confidence": float(g["fused_conf"].mean()),
                    "signature": sig})
    lib.sort(key=lambda p: -p["cases"])
    (WORK / "patterns.json").write_text(json.dumps(lib, indent=1))

    # Layer 6: versioned snapshot of this federation's output
    blob = pickle.dumps([(t.feat.tobytes(), t.thr.tobytes(), t.value.tobytes()) for t in model.trees])
    snap = {"version": hashlib.sha256(blob).hexdigest()[:12], "mode": mode, "trees": len(model.trees),
            "features": len(names), "threshold": thr,
            "evidence_breakdown": {s: int(sc[s].sum()) for s in ("EFRM", "AML", "I4C")},
            "mules_with_any_evidence": int((sc["is_mule"].eq(1) & (sc[["EFRM", "AML", "I4C"]].any(axis=1))).sum()),
            "patterns": {p["pattern"]: p["state"] for p in lib}}
    (WORK / "rounds").mkdir(exist_ok=True)
    (WORK / "rounds" / f"snapshot_{snap['version']}.json").write_text(json.dumps(snap, indent=1))
    print(json.dumps(snap, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["full"]))
