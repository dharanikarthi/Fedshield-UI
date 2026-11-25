"""Local-only scores for the role-based demo.

Each bank trains on ONLY its own training accounts, without consortium registry
features (a bank alone cannot see other banks' confirmations), and picks its own
threshold on its own validation accounts. Output: work/models/nfpc_full_local.parquet
(acct, local_score, local_threshold) and work/demo_cases.json — real test mules that
the bank's local model missed and the federated model caught.

    python -m fedshield.experiments.demo_prep
"""
import json
import pickle

import numpy as np
import pandas as pd

from fedshield.config import WORK
from fedshield.eval.metrics import pick_threshold
from fedshield.experiments.common import class_weights, run_local
from fedshield.experiments.nfpc import NAMES, PARAMS
from fedshield.fl.federation import Bank


def main():
    with open(WORK / "models" / "nfpc_full.pkl", "rb") as fh:
        art = pickle.load(fh)
    feats_all, thr_fed = art["features"], art["threshold"]
    local_feats = [f for f in feats_all if not f.startswith("reg_")]
    F = pd.read_parquet(WORK / "models" / "nfpc_full_features.parquet").set_index("acct")
    sc = pd.read_parquet(WORK / "models" / "nfpc_full_scores.parquet").set_index("acct")
    sc = sc.loc[F.index]
    X = F[local_feats].astype("float32").to_numpy()
    y = sc["is_mule"].to_numpy().astype(int)
    sp = sc["split"].to_numpy()
    conf = sc["label_conf"].to_numpy() if "label_conf" in sc else np.ones(len(sc))
    out, models = [], {}
    for b, name in enumerate(NAMES):
        mb = sc["bank"].to_numpy() == b
        tr, va = mb & (sp == "train"), mb & (sp == "val")
        bank = Bank(name, X[tr], y[tr], class_weights(y[tr], conf[tr]), X[va], y[va])
        model = run_local(bank, PARAMS, local_feats)
        thr = pick_threshold(model.predict_proba(X[va]), y[va])
        p = model.predict_proba(X[mb])
        models[name] = {"model": model, "threshold": thr}
        out.append(pd.DataFrame({"acct": F.index[mb], "local_score": p, "local_threshold": thr}))
        print(f"  {name}: local model trained on {int(tr.sum()):,} accounts, threshold {thr:.3f}", flush=True)
    loc = pd.concat(out, ignore_index=True)
    loc.to_parquet(WORK / "models" / "nfpc_full_local.parquet", index=False)
    with open(WORK / "models" / "nfpc_full_local_models.pkl", "wb") as fh:
        pickle.dump({"features": local_feats, "banks": models}, fh)

    d = sc.join(loc.set_index("acct"))
    te = d[(d["split"] == "test")]
    caught = te[(te["is_mule"] == 1) & (te["score"] >= thr_fed) & (te["local_score"] < te["local_threshold"])]
    lost = te[(te["is_mule"] == 1) & (te["score"] < thr_fed) & (te["local_score"] >= te["local_threshold"])]
    caught = caught.assign(gap=caught["score"] - caught["local_score"]).sort_values("gap", ascending=False)
    cases = [{"account_id": r.account_id, "bank": NAMES[int(r.bank)], "local_score": float(r.local_score),
              "local_threshold": float(r.local_threshold), "fed_score": float(r.score),
              "fed_threshold": float(thr_fed), "alert_reason": r.alert_reason} for r in caught.head(12).itertuples()]
    summary = {"test_mules": int((te["is_mule"] == 1).sum()),
               "caught_by_fed_missed_by_local": int(len(caught)),
               "caught_by_local_missed_by_fed": int(len(lost)),
               "per_bank": {n: int((caught["bank"] == i).sum()) for i, n in enumerate(NAMES)}}
    (WORK / "demo_cases.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
