"""FedShield API — serves one bank's fraud-officer console, the aggregator's
federation dashboard and the live stream.

    uvicorn fedshield.api.server:app --port 8000
"""
import asyncio
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fedshield.api.auth import SCOPE, USERS, issue, mask_id, need, session
from fedshield.api.reasons import explain
from fedshield.config import RESULTS, ROOT, WORK

MODE = "full"
app = FastAPI(title="FedShield API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
AUDIT = WORK / "audit_log.jsonl"
S = {}


def _fetch_bundle():
    """On hosts with an ephemeral disk (Render), pull the artefact bundle at first use.
    FEDSHIELD_BUNDLE_URL = direct download URL of fedshield_work.zip (scripts/make_deploy_bundle.py)."""
    import os
    import urllib.request
    import zipfile
    url = os.environ.get("FEDSHIELD_BUNDLE_URL")
    if not url or (WORK / "models" / f"nfpc_{MODE}.pkl").exists():
        return
    tmp = WORK.parent / "_bundle.zip"
    print(f"downloading artefact bundle ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    with zipfile.ZipFile(tmp) as z:
        for m in z.namelist():                     # bundle layout: work/..., results/...
            top, _, rest = m.partition("/")
            if not rest or m.endswith("/"):
                continue
            dest = (WORK if top == "work" else RESULTS) / rest
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                out.write(src.read())
    tmp.unlink()
    print("bundle ready", flush=True)


def _load():
    if S:
        return S
    _fetch_bundle()
    mdir = WORK / "models"
    with open(mdir / f"nfpc_{MODE}.pkl", "rb") as fh:
        art = pickle.load(fh)
    S["art"] = art
    S["scores"] = pd.read_parquet(mdir / f"nfpc_{MODE}_scores.parquet")
    feats = pd.read_parquet(mdir / f"nfpc_{MODE}_features.parquet")
    S["feats"] = feats.set_index("acct")
    S["medians"] = feats[art["features"]].median(numeric_only=True).to_dict()
    S["monthly"] = pd.read_parquet(WORK / "nfpc" / f"monthly_{MODE}.parquet")
    S["results"] = {p.stem: json.loads(p.read_text()) for p in RESULTS.glob("*.json")}
    lp, lm = mdir / f"nfpc_{MODE}_local.parquet", mdir / f"nfpc_{MODE}_local_models.pkl"
    if lp.exists():
        S["scores"] = S["scores"].merge(pd.read_parquet(lp), on="acct", how="left")
    if lm.exists():
        with open(lm, "rb") as fh:
            S["local"] = pickle.load(fh)
    return S


def _tier(p, thr):
    if p >= max(0.9, thr):
        return "BLOCK-REVIEW"
    if p >= thr:
        return "HOLD-REVIEW"
    if p >= 0.5 * thr:
        return "MONITOR"
    return "CLEAR"


ACTIONS = {"monitor": "Enhanced monitoring", "cooling": "Cooling-off period on new beneficiaries",
           "limit": "Reduce outbound limit", "hold": "Hold outbound debits (lien)",
           "freeze": "Freeze (requires second approver)", "dismiss": "Dismiss — false positive"}


DEMO_STATE = WORK / "demo_state.json"


def deployed():
    """False until the aggregator deploys a federated model: bank officers then see
    their local-only model; afterwards the federated global model."""
    return json.loads(DEMO_STATE.read_text()).get("deployed", False) if DEMO_STATE.exists() else False


def set_deployed(v, by="admin"):
    DEMO_STATE.write_text(json.dumps({"deployed": bool(v), "by": by,
                                      "ts": datetime.now(timezone.utc).isoformat()}))


def _view(r, s):
    """(score, threshold, model) the bank currently sees for a scores row."""
    ls = r.get("local_score", np.nan)
    if deployed() or ls is None or pd.isna(ls):
        return float(r["score"]), s["art"]["threshold"], "federated"
    return float(ls), float(r["local_threshold"]), "local"


def _own(sess, bank_name):
    if sess["role"] != "bank":
        raise HTTPException(403, "only bank officers can open account-level data")
    if sess["bank"] != bank_name:
        raise HTTPException(403, f"{sess['bank']} cannot access {bank_name} accounts")


class Login(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(l: Login):
    tok, info = issue(l.username, l.password)
    return {"token": tok, **info}


@app.get("/api/personas")
def personas():
    return [{"user": u, **v, "scope": SCOPE[v["role"]]} for u, v in USERS.items()]


@app.get("/api/me")
def me(sess=Depends(session)):
    probe = None
    if sess["role"] == "bank":     # an account at ANOTHER bank, for the live 403 privacy probe
        s = _load()
        names = s["art"]["bank_names"]
        other = s["scores"][(s["scores"]["bank"] != names.index(sess["bank"])) & (s["scores"]["split"] == "test")]
        if len(other):
            r = other.iloc[0]
            probe = {"account_id": r["account_id"], "bank": names[int(r["bank"])]}
    return {**sess, "scope": SCOPE[sess["role"]], "deployed": deployed(), "probe_target": probe}


@app.post("/api/demo/reset")
def demo_reset(sess=Depends(session)):
    need(sess, "admin")
    set_deployed(False)
    return {"deployed": False}


@app.post("/api/demo/deploy")
def demo_deploy(sess=Depends(session)):
    need(sess, "admin")
    set_deployed(True)
    return {"deployed": True}


@app.get("/api/health")
def health():
    return {"ok": True, "mode": MODE, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/summary")
def summary():
    s = _load()
    sc, thr = s["scores"], s["art"]["threshold"]
    te = sc[sc["split"] == "test"]
    r = s["results"]
    nf = r.get(f"nfpc_{MODE}", {})
    return {
        "threshold": thr,
        "banks": s["art"]["bank_names"],
        "test_accounts": int(len(te)),
        "alerts": int((te["score"] >= thr).sum()),
        "final_model": nf.get("final_model", {}),
        "nfpc_main": nf.get("main", {}).get("summary"),
        "nfpc_registry": {k: v.get("summary") for k, v in nf.get("registry", {}).items()},
        "aml": {k: v.get("summary") for k, v in r.get("aml_HI-Small", {}).get("experiments", {}).items()
                if isinstance(v, dict) and "summary" in v},
    }


@app.get("/api/alerts")
def alerts(limit: int = 200, include_clear: bool = False, sess=Depends(session)):
    need(sess, "bank")
    s = _load()
    names = s["art"]["bank_names"]
    sc = s["scores"]
    df = sc[(sc["split"] == "test") & (sc["bank"] == names.index(sess["bank"]))]
    acted = _actions_by_account()
    rows = []
    for r in df.to_dict("records"):
        score, thr, model = _view(r, s)
        if not include_clear and score < 0.5 * thr:
            continue
        rows.append({"account_id": r["account_id"], "score": score, "threshold": thr, "model": model,
                     "tier": _tier(score, thr), "bank": sess["bank"], "state": r["branch_state"],
                     "window": [_iso(r.get("win_start")), _iso(r.get("win_end"))],
                     "status": acted.get(r["account_id"], {}).get("action"), "ground_truth": int(r["is_mule"])})
    rows.sort(key=lambda x: -x["score"])
    return rows[:limit]


@app.get("/api/demo/cases")
def demo_cases(sess=Depends(session)):
    """Real test mules the bank's local model missed and the federated model caught."""
    p = WORK / "demo_cases.json"
    d = json.loads(p.read_text()) if p.exists() else {"summary": {}, "cases": []}
    if sess["role"] == "bank":
        d["cases"] = [c for c in d["cases"] if c["bank"] == sess["bank"]]
        if not deployed():
            d["cases"] = [{k: v for k, v in c.items() if not k.startswith("fed_")} for c in d["cases"]]
    elif sess["role"] == "rbi":
        d["cases"] = [{**c, "account_id": mask_id(c["account_id"])} for c in d["cases"]]
    else:
        d["cases"] = []
    return d


def _finite(o):
    """NaN/NaT -> None. JSON has no NaN, so accounts with a missing flag date or alert
    reason (most non-mules) used to fail with HTTP 500."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if o is not None and not isinstance(o, (str, bool, int, float, list, dict)) and pd.isna(o):
        return None
    return o


def _iso(v):
    return None if v is None or pd.isna(v) else pd.Timestamp(v).date().isoformat()


@app.get("/api/accounts/{account_id}")
def account(account_id: str, sess=Depends(session)):
    s = _load()
    art = s["art"]
    row = s["scores"][s["scores"]["account_id"] == account_id]
    if row.empty:
        raise HTTPException(404, "unknown account")
    r = row.iloc[0]
    _own(sess, art["bank_names"][int(r["bank"])])
    acct = int(r["acct"])
    score, thr, which = _view(r.to_dict(), s)
    x = s["feats"].loc[[acct], art["features"]].astype("float32").to_numpy()
    vals = [float(v) for v in x[0]]
    if which == "local" and "local" in s:
        lf = s["local"]["features"]
        xl = s["feats"].loc[[acct], lf].astype("float32").to_numpy()
        cl = s["local"]["banks"][sess["bank"]]["model"].contributions(xl)[0]
        reasons, mitig = explain(lf, [float(v) for v in xl[0]], cl, s["medians"])
    else:
        contrib = art["model"].contributions(x)[0]
        reasons, mitig = explain(art["features"], vals, contrib, s["medians"])
    mon = s["monthly"][s["monthly"]["acct"] == acct].sort_values("month")
    reg = {f: vals[art["features"].index(f)] for f in art["features"] if f.startswith("reg_")}
    return _finite({
        "account_id": account_id, "score": score, "tier": _tier(score, thr), "threshold": thr, "model": which,
        "local_score": None if pd.isna(r.get("local_score", np.nan)) else float(r["local_score"]),
        "local_threshold": None if pd.isna(r.get("local_threshold", np.nan)) else float(r["local_threshold"]),
        "fed_score": float(r["score"]) if which == "federated" else None, "fed_threshold": art["threshold"], "bank": art["bank_names"][int(r["bank"])], "state": r["branch_state"],
        "reasons": reasons, "mitigating": mitig, "registry": reg,
        "window": [_iso(r.get("win_start")), _iso(r.get("win_end"))],
        "timeline": [{"month": pd.Timestamp(m).strftime("%Y-%m"), "n": int(n), "credit": float(c), "debit": float(d)}
                     for m, n, c, d in mon[["month", "n", "credit", "debit"]].itertuples(index=False)],
        "label": {"is_mule": int(r["is_mule"]), "flag_date": r["mule_flag_date"],
                  "alert_reason": r["alert_reason"], "label_confidence": float(r["label_conf"]),
                  "red_herring_flags": r["rh_reason"] or None},
        "history": [a for a in _audit() if a["account_id"] == account_id],
        "actions": ACTIONS,
    })


@app.get("/api/accounts/{account_id}/graph")
def graph(account_id: str, max_cp: int = 25, sess=Depends(session)):
    """Ego network: account -> top counterparties -> other accounts sharing them.
    Accounts at OTHER banks appear only as hashed pseudonyms (registry view)."""
    s = _load()
    row = s["scores"][s["scores"]["account_id"] == account_id]
    if row.empty:
        raise HTTPException(404, "unknown account")
    _own(sess, s["art"]["bank_names"][int(row.iloc[0]["bank"])])
    acct = int(account_id[5:])
    import duckdb
    ep = (WORK / "nfpc" / f"edges_{MODE}.parquet").as_posix()
    con = duckdb.connect()
    mine = con.execute(f"""SELECT cp, is_credit, n, v FROM read_parquet('{ep}') WHERE acct = ?
                           ORDER BY v DESC LIMIT ?""", [acct, max_cp]).df()
    if mine.empty:
        return {"nodes": [], "edges": []}
    cps = mine["cp"].unique().tolist()
    con.register("cps", pd.DataFrame({"cp": cps}))
    shared = con.execute(f"""SELECT e.acct, e.cp, sum(e.v) v FROM read_parquet('{ep}') e JOIN cps USING (cp)
                             WHERE e.acct <> ? GROUP BY 1, 2""", [acct]).df()
    deg = shared.groupby("cp")["acct"].nunique()
    sc = s["scores"].set_index("acct")
    nodes = [{"id": account_id, "type": "subject", "score": float(sc.loc[acct, "score"]) if acct in sc.index else None}]
    edges = []
    for r in mine.itertuples():
        cid = f"CP-{str(r.cp)[-6:]}"
        nodes.append({"id": cid, "type": "counterparty", "degree": int(deg.get(r.cp, 0))})
        edges.append({"source": cid, "target": account_id, "amount": float(r.v)} if r.is_credit
                     else {"source": account_id, "target": cid, "amount": float(r.v)})
    rare = shared[shared["cp"].map(deg) <= 20]          # rare counterparties only: merchants carry no signal
    for a, g in rare.groupby("acct"):
        if a not in sc.index:
            continue
        other = s["art"]["bank_names"][int(sc.loc[a, "bank"])]
        own = other == sess["bank"]
        nid = sc.loc[a, "account_id"] if own else mask_id(sc.loc[a, "account_id"])
        mule = bool(sc.loc[a, "is_mule"]) and sc.loc[a, "split"] == "train"
        nodes.append({"id": nid, "type": "confirmed_mule" if mule else "account",
                      "bank": other if own else "other bank (hashed)",
                      "score": float(sc.loc[a, "score"]) if own else None})
        for cp in g["cp"]:
            edges.append({"source": f"CP-{str(cp)[-6:]}", "target": nid, "amount": 0, "shared": True})
    seen, uniq = set(), []
    for n in nodes:
        if n["id"] not in seen:
            seen.add(n["id"]); uniq.append(n)
    uniq = uniq[:120]
    keep = {n["id"] for n in uniq}          # only edges whose two ends survived the 120-node cap
    return {"nodes": uniq, "edges": [e for e in edges if e["source"] in keep and e["target"] in keep]}


class Action(BaseModel):
    action: str
    officer: str = "officer-1"
    note: str = ""


@app.post("/api/accounts/{account_id}/action")
def act(account_id: str, a: Action, sess=Depends(session)):
    s = _load()
    row = s["scores"][s["scores"]["account_id"] == account_id]
    if row.empty:
        raise HTTPException(404, "unknown account")
    _own(sess, s["art"]["bank_names"][int(row.iloc[0]["bank"])])
    a.officer = sess["sub"]
    if a.action not in ACTIONS:
        raise HTTPException(400, f"action must be one of {list(ACTIONS)}")
    rec = {"account_id": account_id, "bank": sess["bank"], "action": a.action, "label": ACTIONS[a.action], "officer": a.officer,
           "note": a.note, "ts": datetime.now(timezone.utc).isoformat(),
           "feedback_label": 0 if a.action == "dismiss" else (1 if a.action in ("hold", "freeze") else None)}
    with open(AUDIT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _audit():
    if not AUDIT.exists():
        return []
    return [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]


def _actions_by_account():
    return {a["account_id"]: a for a in _audit()}


@app.get("/api/audit")
def audit(sess=Depends(session)):
    rows = list(reversed(_audit()))[:500]
    if sess["role"] == "bank":
        return [r for r in rows if r.get("bank") == sess["bank"]]
    if sess["role"] == "rbi":
        return [{**r, "account_id": mask_id(r["account_id"]), "note": ""} for r in rows]
    raise HTTPException(403, "the aggregator holds no account-level audit data")


@app.get("/api/federation")
def federation():
    s = _load()
    nf = s["results"].get(f"nfpc_{MODE}", {})
    src = nf.get("registry", {}).get("fed") or nf.get("main") or {}
    log = src.get("fed_log", {})
    return {
        "banks": nf.get("banks"),
        "gate2": log.get("gate2"), "weights": log.get("weights"), "config": log.get("config"),
        "rounds": log.get("rounds", []),
        "byzantine": {k: {kk: v.get(kk) for kk in ("overall", "gate2", "gate1_rejections", "accepted")}
                      for k, v in nf.get("byzantine", {}).items()},
        "dp": nf.get("dp"),
        "fairness": nf.get("fairness"),
        "label_shield": nf.get("label_shield_counts"),
        "patterns": _patterns(),
    }


def _patterns():
    p = WORK / "patterns.json"
    return json.loads(p.read_text()) if p.exists() else []


@app.get("/api/experiments")
def experiments():
    return _load()["results"]


# ------------------------------------------------------------------ live stream
@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    """Replays a PaySim-derived feed through the online scorer (see fedshield/stream)."""
    await ws.accept()
    from fedshield.stream.scorer import StreamScorer
    scorer = StreamScorer.load()
    from fedshield.stream.bus import Consumer, proxy
    if proxy():                       # Redpanda mode: score what the replayer publishes
        try:
            async with Consumer() as cons:
                while True:
                    for ev in await cons.records():
                        t = time.perf_counter()
                        out = scorer.score(ev)
                        out["latency_ms"] = (time.perf_counter() - t) * 1000
                        out["source"] = "redpanda"
                        await ws.send_json(out)
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return
    try:
        speed = float(ws.query_params.get("rate", 20))
        for ev in scorer.events():
            t = time.perf_counter()
            out = scorer.score(ev)
            out["latency_ms"] = (time.perf_counter() - t) * 1000
            await ws.send_json(out)
            await asyncio.sleep(1.0 / speed)
    except WebSocketDisconnect:
        return


# ------------------------------------------------------------------ RBI supervisory view
@app.get("/api/rbi/overview")
def rbi_overview(sess=Depends(session)):
    need(sess, "rbi", "admin")
    s = _load()
    if "clusters" not in s:
        cached = WORK / "rbi_clusters.json"           # precomputed by scripts/make_deploy_bundle.py
        if cached.exists():
            s["clusters"] = json.loads(cached.read_text())
        else:
            from fedshield.api.supervisor import clusters
            s["clusters"] = clusters(s["scores"], s["art"]["bank_names"], s["art"]["threshold"])
    sc, thr, names = s["scores"], s["art"]["threshold"], s["art"]["bank_names"]
    te = sc[sc["split"] == "test"]
    per_bank = {n: {"accounts": int((te["bank"] == i).sum()), "alerts": int(((te["bank"] == i) & (te["score"] >= thr)).sum())}
                for i, n in enumerate(names)}
    nf = s["results"].get(f"nfpc_{MODE}", {})
    demo = json.loads((WORK / "demo_cases.json").read_text())["summary"] if (WORK / "demo_cases.json").exists() else {}
    return {"per_bank": per_bank, "clusters": s["clusters"], "patterns": _patterns(),
            "fairness": nf.get("fairness"), "federation_effect": demo,
            "deployed": deployed(), "model": nf.get("final_model", {}).get("test")}


# ------------------------------------------------------------------ live federation pipeline
class RunReq(BaseModel):
    dataset: str = "nfpc"
    sample: int = 30000
    n_trees: int = 40
    attack: str | None = None          # None | label_flip | gradient_scale
    dp_epsilon: float | None = None
    byzantine: bool = True
    secure_agg: bool = True


@app.get("/api/datasets")
def datasets():
    from fedshield.adapters import REGISTRY, available
    return {"registered": sorted(REGISTRY), "available": available()}


@app.post("/api/pipeline/run")
def pipeline_run(r: RunReq, sess=Depends(session)):
    need(sess, "admin")
    from fedshield.api.pipeline import LIVE
    LIVE.on_done = lambda: set_deployed(True)
    if not LIVE.start(**r.model_dump()):
        raise HTTPException(409, "a run is already in progress")
    return {"started": True, "config": LIVE.cfg}


@app.get("/api/pipeline/state")
def pipeline_state():
    from fedshield.api.pipeline import LIVE
    return {"running": LIVE.running, "config": LIVE.cfg, "events": len(LIVE.events)}


@app.websocket("/api/pipeline")
async def pipeline_ws(ws: WebSocket):
    """Streams the live run's events (replays from the start, then follows)."""
    from fedshield.api.pipeline import LIVE
    await ws.accept()
    seq = 0
    try:
        while True:
            new = LIVE.since(seq)
            for e in new:
                await ws.send_json(e)
            seq += len(new)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


UI_DIST = ROOT / "ui" / "dist"
if UI_DIST.exists():
    from fastapi.responses import FileResponse
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = UI_DIST / path
        if path and f.is_file() and UI_DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(UI_DIST / "index.html")   # client-side routes (/pipeline, /stream, ...)
