"""Record the UI's demo-mode fixtures from the real FedShield API (in-process, no server needed).

The Vercel demo (ui: `npm run build:demo`) has no backend. It answers every REST call and
WebSocket from the JSON written here, so every number on screen came from the real models:

    python scripts/record_demo_fixtures.py                  # all fixtures -> ui/demo-public/demo
    python scripts/record_demo_fixtures.py --only api       # REST responses only
    python scripts/record_demo_fixtures.py --only pipeline  # live federation runs (~20 min, resumable)

Needs the artefact bundle (FEDSHIELD_WORK / FEDSHIELD_RESULTS, see docs/DEPLOYMENT.md).
Set FEDSHIELD_SECRET so the RBI pseudonyms are stable between recordings.
"""
import argparse
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "ui" / "demo-public" / "demo"
BANKS = {"bank_a": "Bank-A", "bank_b": "Bank-B", "bank_c": "Bank-C"}
STATES = [("local", False), ("federated", True)]


def clean(o):
    """NaN/inf -> null, the same way a browser would need it (JSON has no NaN)."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def save(rel, obj):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(json.loads(json.dumps(obj, default=str))), separators=(",", ":"), allow_nan=False))


def record_api():
    import fedshield.api.server as srv
    from fastapi import HTTPException
    from fedshield.api.auth import USERS, mask_id

    sess = {u: {"sub": u, **v} for u, v in USERS.items()}
    save("personas.json", srv.personas())
    save("datasets.json", srv.datasets())
    save("federation.json", srv.federation())
    save("experiments.json", srv.experiments())
    for u in USERS:
        save(f"me/{u}.json", srv.me(sess=sess[u]))

    ids = {b: set() for b in BANKS.values()}
    for state, dep in STATES:
        srv.set_deployed(dep)
        for u, b in BANKS.items():
            al = srv.alerts(limit=300, include_clear=False, sess=sess[u])
            cs = srv.demo_cases(sess=sess[u])
            save(f"alerts/{b}_{state}.json", al)
            save(f"cases/{b}_{state}.json", cs)
            ids[b] |= {a["account_id"] for a in al} | {c["account_id"] for c in cs.get("cases", [])}
            print(f"  {state:9s} {b}: {len(al)} alerts, {len(cs.get('cases', []))} watch-list cases", flush=True)
        save(f"rbi_overview_{state}.json", srv.rbi_overview(sess=sess["rbi"]))
        save(f"cases/rbi_{state}.json", srv.demo_cases(sess=sess["rbi"]))

    owner = {}
    for state, dep in STATES:
        srv.set_deployed(dep)
        for u, b in BANKS.items():
            for a in sorted(ids[b]):
                save(f"account/{state}/{a}.json", srv.account(a, sess=sess[u]))
                owner[a] = b
    for u, b in BANKS.items():
        for a in sorted(ids[b]):
            save(f"graph/{a}.json", srv.graph(a, 25, sess=sess[u]))
    for u, b in BANKS.items():                       # privacy probe targets: other banks' accounts
        t = srv.me(sess=sess[u])["probe_target"]
        if t:
            owner.setdefault(t["account_id"], t["bank"])
    srv.set_deployed(False)
    save("owners.json", owner)
    save("masks.json", {a: mask_id(a) for a in owner})
    print(f"  {len(owner)} accounts with detail + money-flow graph", flush=True)


def record_stream(n=4000):
    from fedshield.stream.scorer import StreamScorer
    s, out = StreamScorer.load(), []
    for ev in s.events():
        t = time.perf_counter()
        o = s.score(ev)
        o["latency_ms"] = (time.perf_counter() - t) * 1000
        out.append(o)
        if len(out) >= n:
            break
    save("stream.json", out)
    print(f"  {len(out)} scored payment events", flush=True)


def record_pipeline():
    from fedshield.api.pipeline import LiveRun
    for attack, dp, byz, trees in itertools.product([None, "label_flip", "gradient_scale"], [None, 64, 16, 8],
                                                    [True, False], [20, 40, 80]):
        key = f"{attack or 'none'}_{dp or 'off'}_{'byz' if byz else 'nobyz'}_{trees}"
        if (OUT / "pipeline" / f"{key}.json").exists():
            continue
        lr = LiveRun()
        lr.on_done = lambda: None                    # so the run ends with the "deployed" event, as in the UI
        t = time.time()
        lr._run(dict(dataset="nfpc", sample=30000, n_trees=trees, max_depth=4, attack=attack, attack_bank=None,
                     dp_epsilon=dp, byzantine=byz, secure_agg=True, pace=0))   # the UI replays at the server's pace
        for e in lr.events:
            e.pop("t", None)
        save(f"pipeline/{key}.json", lr.events)
        print(f"  {key}: {len(lr.events)} events, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["api", "stream", "pipeline"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fav = ROOT / "ui" / "public" / "favicon.svg"
    if fav.exists():
        shutil.copy2(fav, OUT.parent / "favicon.svg")
    for name, fn in [("api", record_api), ("stream", record_stream), ("pipeline", record_pipeline)]:
        if a.only in (None, name):
            print(f"recording {name} ...", flush=True)
            fn()
    print(f"done -> {OUT}")
