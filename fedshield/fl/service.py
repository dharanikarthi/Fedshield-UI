"""Distributed deployment of the histogram protocol: one HTTP service per bank,
one aggregator process. The aggregator never receives rows — only secure-
aggregation-masked histogram vectors, which it can sum but not read individually.

Bank:        BANK_NAME=Bank-A BANK_DATA=work/partitions/Bank-A.parquet \
             uvicorn fedshield.fl.service:bank_app --port 9001
Aggregator:  python -m fedshield.fl.service aggregate http://bank-a:9001 http://bank-b:9002 ...
"""
import base64
import json
import os
import pickle
import sys
import time

import numpy as np

from fedshield.fl.gbdt import (GBDTModel, GBDTParams, HistParty, Tree, apply_bins,
                               base_score_from_counts, build_tree, merge_edges)
from fedshield.fl.privacy import SecureAggregator


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode()


def _unb64(s: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


# ============================================================================ bank side
def make_bank_app():
    import pandas as pd
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="FedShield bank node")
    st = {}

    def load():
        if st:
            return st
        df = pd.read_parquet(os.environ["BANK_DATA"])
        feats = json.loads(os.environ.get("BANK_FEATURES", "null")) or [c for c in df.columns if c not in ("y", "w")]
        st.update(name=os.environ["BANK_NAME"], df=df, feats=feats,
                  X=df[feats].astype("float32").to_numpy(), y=df["y"].to_numpy(), w=df["w"].to_numpy())
        return st

    class Init(BaseModel):
        params: dict
        edges: list
        base_score: float
        participants: list

    class Hist(BaseModel):
        nodes: list
        allowed: list
        session: int

    class Splits(BaseModel):
        splits: dict

    class TreeMsg(BaseModel):
        tree: dict

    @app.get("/info")
    def info():
        s = load()
        return {"name": s["name"], "rows": int(len(s["y"])), "features": s["feats"]}

    @app.get("/quantiles")
    def quantiles(n_bins: int = 64):
        from fedshield.fl.gbdt import local_quantiles
        s = load()
        q = local_quantiles(s["X"], n_bins)
        return {"q": np.nan_to_num(q, nan=0.0).tolist(), "n": int(len(s["y"])),
                "pos_w": float((s["y"] * s["w"]).sum()), "tot_w": float(s["w"].sum())}

    @app.post("/init")
    def init(m: Init):
        s = load()
        p = GBDTParams(**m.params)
        p.base_score = m.base_score
        edges = [np.asarray(e) for e in m.edges]
        s["party"] = HistParty(s["name"], apply_bins(s["X"], edges), s["y"], s["w"], p)
        s["party"].raw[:] = m.base_score
        s["participants"] = m.participants
        return {"ok": True}

    @app.post("/start_tree")
    def start_tree():
        load()["party"].start_tree()
        return {"ok": True}

    @app.post("/histograms")
    def histograms(h: Hist):
        s = load()
        party = s["party"]
        party.allowed = np.asarray(h.allowed)
        hist = party.histograms(h.nodes)
        sent = np.zeros(hist.shape[1], bool)
        sent[party.allowed] = True
        hist = np.where(sent[None, :, None, None], hist, 0.0)
        sa = SecureAggregator(s["participants"], session_seed=h.session)
        return {"masked": _b64(sa.mask(s["name"], hist)), "shape": list(hist.shape)}

    @app.post("/apply_splits")
    def apply_splits(m: Splits):
        load()["party"].apply_splits({int(k): tuple(v) for k, v in m.splits.items()})
        return {"ok": True}

    @app.post("/finish_tree")
    def finish_tree(m: TreeMsg):
        t = Tree(*(np.asarray(m.tree[k]) for k in ("feat", "thr", "left", "right", "value", "gain")))
        t.feat = t.feat.astype(np.int32); t.thr = t.thr.astype(np.int32)
        t.left = t.left.astype(np.int32); t.right = t.right.astype(np.int32)
        load()["party"].finish_tree(t)
        return {"ok": True}

    return app


bank_app = make_bank_app() if os.environ.get("BANK_NAME") else None


# ============================================================================ aggregator side
class RemoteParty:
    """Proxy with the HistParty interface used by build_tree()."""

    def __init__(self, url, client):
        self.url, self.c = url.rstrip("/"), client
        info = self.c.get(f"{self.url}/info").json()
        self.name, self.rows, self.features = info["name"], info["rows"], info["features"]
        self.F = len(self.features)
        self.allowed = None
        self.session = 0
        self._masked = None

    def start_tree(self):
        self.c.post(f"{self.url}/start_tree")

    def histograms(self, node_ids):
        r = self.c.post(f"{self.url}/histograms", json={
            "nodes": [int(n) for n in node_ids], "allowed": [int(a) for a in self.allowed],
            "session": self.session}).json()
        self.session += 1
        return (_unb64(r["masked"], np.uint64), tuple(r["shape"]))

    def apply_splits(self, splits):
        self.c.post(f"{self.url}/apply_splits", json={"splits": {str(k): list(v) for k, v in splits.items()}})

    def finish_tree(self, tree):
        self.c.post(f"{self.url}/finish_tree", json={"tree": {k: getattr(tree, k).tolist() for k in
                                                             ("feat", "thr", "left", "right", "value", "gain")}})


def aggregate_remote(urls, params: GBDTParams, out_path):
    import httpx
    client = httpx.Client(timeout=600)
    parties = [RemoteParty(u, client) for u in urls]
    names = [p.name for p in parties]
    print(f"aggregator: {len(parties)} banks {names}; rows never leave the banks")
    qs = [client.get(f"{p.url}/quantiles", params={"n_bins": params.n_bins}).json() for p in parties]
    edges = merge_edges([np.asarray(q["q"]) for q in qs], [q["n"] for q in qs])
    params.base_score = base_score_from_counts(sum(q["pos_w"] for q in qs), sum(q["tot_w"] for q in qs))
    for p in parties:
        client.post(f"{p.url}/init", json={"params": params.__dict__, "edges": [e.tolist() for e in edges],
                                           "base_score": params.base_score, "participants": names})
    sa = SecureAggregator(names)          # aggregator side: only sums; it never holds pair seeds
    model = GBDTModel(params, edges, parties[0].features)
    rng = np.random.RandomState(params.seed)
    t0 = time.time()

    def aggregate(masked_list):
        vecs = [m for m, _ in masked_list]
        shape = masked_list[0][1]
        return sa.unmask_sum(vecs, shape)

    for t in range(params.n_trees):
        assert len({p.session for p in parties}) == 1   # all banks mask with the same session id
        tree = build_tree(parties, aggregate, params, rng)
        model.trees.append(tree)
        if (t + 1) % 10 == 0:
            print(f"  round {t + 1}/{params.n_trees}  {time.time() - t0:.0f}s", flush=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"model": model, "banks": names}, fh)
    print(f"saved {out_path}")
    return model


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        from fedshield.config import WORK
        p = GBDTParams(n_trees=int(os.environ.get("N_TREES", 100)), max_depth=6, learning_rate=0.08)
        aggregate_remote(sys.argv[2:], p, WORK / "models" / "distributed.pkl")
