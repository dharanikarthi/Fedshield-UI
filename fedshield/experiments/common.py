"""Shared experiment runners: local-only, federated, centralised — same model class."""
import json
import time

import numpy as np

from fedshield.config import RESULTS
from fedshield.eval.metrics import pick_threshold, scores
from fedshield.fl.federation import Bank, FedConfig, federate
from fedshield.fl.gbdt import GBDTParams


def class_weights(y, conf=None, cap=None):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    if cap:
        spw = min(spw, cap)
    w = np.where(y == 1, spw, 1.0)
    return w * (conf if conf is not None else 1.0)


def evaluate(model, splits, banks_test, name):
    """splits: dict with X_val,y_val (pooled validation) ; banks_test: {bank: (X, y)}"""
    thr = pick_threshold(model.predict_proba(splits["X_val"]), splits["y_val"])
    Xt, yt = splits["X_test"], splits["y_test"]
    out = {"model": name, "overall": scores(model.predict_proba(Xt), yt, thr), "per_bank": {}}
    for b, (X, y) in banks_test.items():
        out["per_bank"][b] = scores(model.predict_proba(X), y, thr)
    return out


def run_local(bank: Bank, params: GBDTParams, feats):
    cfg = FedConfig(params=GBDTParams(**params.__dict__), secure_agg=False, byzantine=False,
                    weighting=False)
    model, log = federate([bank], cfg, feats)
    return model


def run_fed(banks, params: GBDTParams, feats, **kw):
    cfg = FedConfig(params=GBDTParams(**params.__dict__), **kw)
    return federate(banks, cfg, feats)


def run_central(banks, params: GBDTParams, feats, edges=None):
    pool = Bank("pooled", np.vstack([b.X for b in banks]), np.concatenate([b.y for b in banks]),
                np.concatenate([b.w for b in banks]), np.vstack([b.X_val for b in banks]),
                np.concatenate([b.y_val for b in banks]))
    cfg = FedConfig(params=GBDTParams(**params.__dict__), secure_agg=False, byzantine=False,
                    weighting=False)
    return federate([pool], cfg, feats, edges=edges)[0]


def comparison(banks, params, feats, splits, banks_test, tag, central=True, fed_kw=None,
               verbose=True):
    """Local-only (each bank, scored on its own test rows) vs FedShield vs centralised."""
    rows = []
    t0 = time.time()
    local_own = {}
    for b in banks:
        m = run_local(b, params, feats)
        r = evaluate(m, {"X_val": b.X_val, "y_val": b.y_val, "X_test": splits["X_test"],
                         "y_test": splits["y_test"]}, {b.name: banks_test[b.name]}, f"{b.name} local-only")
        local_own[b.name] = r["per_bank"][b.name]
        rows.append(r)
        if verbose:
            print(f"  [{tag}] {b.name} local-only own-test AUPRC={r['per_bank'][b.name]['auprc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    fed, log = run_fed(banks, params, feats, **(fed_kw or {}))
    rf = evaluate(fed, splits, banks_test, "FedShield")
    rows.append(rf)
    if verbose:
        print(f"  [{tag}] FedShield overall AUPRC={rf['overall']['auprc']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    cen = None
    if central:
        cen = run_central(banks, params, feats, edges=log["edges"])
        rc = evaluate(cen, splits, banks_test, "Centralised (pooled, privacy-free)")
        rows.append(rc)
        if verbose:
            print(f"  [{tag}] Centralised overall AUPRC={rc['overall']['auprc']:.4f}", flush=True)
    summary = {
        "local_own_auprc_mean": float(np.mean([v["auprc"] for v in local_own.values()])),
        "fed_own_auprc_mean": float(np.mean([rf["per_bank"][b]["auprc"] for b in local_own])),
        "fed_overall_auprc": rf["overall"]["auprc"],
        "central_overall_auprc": rows[-1]["overall"]["auprc"] if central else None,
        "per_bank_gain": {b: rf["per_bank"][b]["auprc"] - local_own[b]["auprc"] for b in local_own},
    }
    return {"rows": rows, "summary": summary, "fed_log": _slim_log(log)}, fed, cen


def _slim_log(log):
    return {k: v for k, v in log.items() if k != "edges"}


def save(name, obj):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
