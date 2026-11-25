"""RBI supervisory view: cross-bank mule clusters, evidence provenance, pattern
library. Every account appears ONLY as a keyed-hash pseudonym; no balances,
transactions or customer attributes leave this module."""
import duckdb
import numpy as np
import pandas as pd

from fedshield.api.auth import mask_id
from fedshield.config import WORK
from fedshield.patterns.library import simulate_evidence


def _components(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return list(groups.values())


def clusters(scores, bank_names, thr, max_cp_degree=20, min_banks=2, limit=40):
    """Flagged accounts (federated score >= threshold, or confirmed mules) linked by
    rare shared counterparties, keeping only clusters that span >= min_banks banks."""
    sc = scores
    flagged = sc[(sc["score"] >= thr) | ((sc["is_mule"] == 1) & (sc["split"] == "train"))]
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=4")
    con.register("f", flagged[["acct"]])
    ep = (WORK / "nfpc" / "edges_full.parquet").as_posix()
    e = con.execute(f"""
        WITH d AS (SELECT cp, count(DISTINCT acct) deg FROM '{ep}' GROUP BY cp HAVING deg BETWEEN 2 AND {max_cp_degree}),
             fe AS (SELECT DISTINCT e.acct, e.cp FROM '{ep}' e JOIN f USING (acct) JOIN d USING (cp))
        SELECT a.acct a, b.acct b, a.cp FROM fe a JOIN fe b ON a.cp = b.cp AND a.acct < b.acct""").df()
    if e.empty:
        return []
    info = sc.set_index("acct")
    ev = simulate_evidence(sc).set_index(sc["acct"])
    out = []
    for comp in _components(zip(e["a"], e["b"])):
        banks = sorted({bank_names[int(info.loc[a, "bank"])] for a in comp})
        if len(banks) < min_banks:
            continue
        rows = info.loc[comp]
        sub = e[e["a"].isin(comp)]
        src = ev.loc[comp]
        out.append({
            "cluster": mask_id("cluster:" + ",".join(map(str, sorted(comp))))[2:],
            "size": len(comp), "banks": banks,
            "members": [{"id": mask_id(info.loc[a, "account_id"]), "bank": bank_names[int(info.loc[a, "bank"])],
                         "score": round(float(info.loc[a, "score"]), 3),
                         "status": "confirmed" if (info.loc[a, "is_mule"] == 1 and info.loc[a, "split"] == "train") else "flagged",
                         "sources": [k for k in ("EFRM", "AML", "I4C") if bool(src.loc[a, k])]} for a in comp][:60],
            "links": [{"a": mask_id(info.loc[r.a, "account_id"]), "b": mask_id(info.loc[r.b, "account_id"]),
                       "via": "CP-" + mask_id(str(r.cp))[2:8]} for r in sub.head(150).itertuples()],
            "shared_counterparties": int(sub["cp"].nunique()),
            "confirmed": int(((rows["is_mule"] == 1) & (rows["split"] == "train")).sum()),
            "evidence": {k: int(src[k].sum()) for k in ("EFRM", "AML", "I4C")},
            "max_score": round(float(rows["score"].max()), 3),
        })
    out.sort(key=lambda c: (-len(c["banks"]), -c["size"]))
    return out[:limit]
