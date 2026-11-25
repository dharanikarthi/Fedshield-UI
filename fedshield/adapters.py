"""Plug-in point for datasets.

Every dataset is reduced to ONE contract, and everything downstream (federation,
gates, DP, registry, evaluation, live pipeline, UI) consumes only that contract:

    Prepared(name, unit, frame, features)
      frame columns: <features...>, y (0/1), bank (str), split ('train'|'test'), entity (id)

To plug in a new dataset: write a function that returns Prepared and decorate it
with @adapter("name"). Nothing else in the codebase changes.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedshield.config import WORK

REGISTRY = {}
# columns that must never be features, in any dataset (labels, label proxies, ids)
GLOBAL_BLOCK = {"y", "w", "bank", "split", "entity", "acct", "acct_id", "account_id", "is_mule", "pattern",
                "entity_id", "inst", "month", "mule_flag_date", "alert_reason", "account_status",
                "freeze_date", "unfreeze_date", "flagged_by_branch", "branch_state", "score"}


@dataclass
class Prepared:
    name: str
    unit: str            # what one row is: account / application / transaction
    frame: pd.DataFrame
    features: list

    def banks(self):
        return sorted(self.frame["bank"].unique())


def adapter(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def load(name, sample=None, seed=42) -> Prepared:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(REGISTRY)}")
    p = REGISTRY[name]()
    f = p.frame
    if sample and len(f) > sample:
        # stratified: keep every positive up to half the sample, fill with negatives
        rng = np.random.RandomState(seed + 1001)   # decorrelated from the split, which used the base seed
        y = f["y"].to_numpy()
        pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        pi = rng.choice(pi, min(len(pi), sample // 2), replace=False)
        ni = rng.choice(ni, sample - len(pi), replace=False)
        f = f.iloc[rng.permutation(np.concatenate([pi, ni]))].reset_index(drop=True)
    leak = [c for c in p.features if c in GLOBAL_BLOCK]
    assert not leak, f"leakage guard: {leak} cannot be features"
    f[p.features] = f[p.features].astype("float32")
    return Prepared(p.name, p.unit, f, p.features)


def available():
    out = {}
    for n, path in (("nfpc", WORK / "models" / "nfpc_full_features.parquet"),
                    ("aml", WORK / "aml" / "HI-Small" / "features.parquet"),
                    ("baf", WORK / "baf" / "Base.parquet")):
        out[n] = path.exists()
    return out


# ----------------------------------------------------------------------------- built-in adapters
@adapter("nfpc")
def _nfpc():
    feats = pd.read_parquet(WORK / "models" / "nfpc_full_features.parquet")
    sc = pd.read_parquet(WORK / "models" / "nfpc_full_scores.parquet", columns=["acct", "is_mule", "bank", "split"])
    f = feats.merge(sc, on="acct")
    f = f.rename(columns={"is_mule": "y", "acct": "entity"})
    f["bank"] = "Bank-" + f["bank"].map(lambda b: "ABC"[int(b)])
    f["split"] = f["split"].replace({"val": "train"})
    names = [c for c in feats.columns if c != "acct"]
    return Prepared("nfpc", "account", f, names)


@adapter("aml")
def _aml():
    f = pd.read_parquet(WORK / "aml" / "HI-Small" / "features.parquet")
    f["bank"] = "Inst-" + f["inst"].astype(str)
    # entity-level split: whole laundering entities stay on one side
    key = f["entity_id"].astype("string").fillna(f["acct_id"].astype("string"))
    h = pd.util.hash_pandas_object(key, index=False).to_numpy() % 5
    f["split"] = np.where(h == 0, "test", "train")
    f = f.rename(columns={"acct_id": "entity"})
    names = [c for c in f.columns if c not in GLOBAL_BLOCK]
    return Prepared("aml", "account", f, names)


@adapter("baf")
def _baf():
    f = pd.read_parquet(WORK / "baf" / "Base.parquet")
    f["bank"] = "PT-" + f["bank"].astype(str)
    f["entity"] = np.arange(len(f))
    names = [c for c in f.columns if c not in GLOBAL_BLOCK and c != "age_50plus"]
    return Prepared("baf", "application", f, names)
