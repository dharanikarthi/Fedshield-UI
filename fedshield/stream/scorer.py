"""Real-time path (Layer 7): score each payment on the bank's own systems.

PaySim is used ONLY as a realistic event stream (its accounts have no history,
so its accuracy is not a FedShield result). Each event is enriched from an
online feature store — Redis if REDIS_URL is set, else in-process — with
streaming signals the batch model can't see: receiver fan-in over the last 24
steps (hours), sender velocity, and whether the payment drains the account.

    python -m fedshield.stream.scorer train      # trains + saves work/models/stream.pkl
"""
import os
import pickle
import sys
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from fedshield.config import PAYSIM_FILE, SEED, WORK

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
DISPLAY = {"PAYMENT": "UPI merchant payment", "TRANSFER": "UPI P2P transfer", "CASH_OUT": "Cash-out (agent/ATM)",
           "CASH_IN": "Cash deposit", "DEBIT": "Debit"}
FEATS = ["amount_log", "hour", "drains_sender", "sender_bal_log", "dest_is_merchant",
         "dest_fanin_24h", "dest_rx_24h", "sender_tx_24h", "amount_vs_sender_mean"] + [f"type_{t}" for t in TYPES]
BANKS = ["Bank-A", "Bank-B", "Bank-C"]


class FeatureStore:
    """Minimal online store. Redis-backed when REDIS_URL is set."""

    def __init__(self, window=24):
        self.window = window
        self.r = None
        if os.environ.get("REDIS_URL"):
            try:
                import redis
                self.r = redis.Redis.from_url(os.environ["REDIS_URL"])
                self.r.ping()
            except Exception:
                self.r = None
        self.dest = defaultdict(deque)           # dest -> deque[(step, sender)]
        self.src = defaultdict(deque)            # sender -> deque[(step, amount)]

    def _trim(self, dq, step):
        while dq and dq[0][0] < step - self.window:
            dq.popleft()

    def update_and_get(self, step, sender, dest, amount):
        if self.r is not None:
            p = self.r.pipeline()
            kd, ks = f"d:{dest}", f"s:{sender}"
            p.zremrangebyscore(kd, 0, step - self.window - 1)
            p.zremrangebyscore(ks, 0, step - self.window - 1)
            p.zadd(kd, {f"{sender}:{step}:{amount}": step})
            p.zadd(ks, {f"{step}:{amount}": step})
            p.zrange(kd, 0, -1)
            p.zrange(ks, 0, -1)
            p.expire(kd, 3600); p.expire(ks, 3600)
            res = p.execute()
            d_items, s_items = res[4], res[5]
            senders = {x.decode().split(":")[0] for x in d_items}
            amts = [float(x.decode().split(":")[1]) for x in s_items[:-1]]
            return len(senders), len(d_items), len(s_items) - 1, (np.mean(amts) if amts else amount)
        dq, sq = self.dest[dest], self.src[sender]
        self._trim(dq, step); self._trim(sq, step)
        prev_amts = [a for _, a in sq]
        dq.append((step, sender)); sq.append((step, amount))
        return len({s for _, s in dq}), len(dq), len(prev_amts), (np.mean(prev_amts) if prev_amts else amount)


def featurise(df, store=None):
    """Batch featurisation that mirrors the online path exactly (processed in time order)."""
    store = store or FeatureStore()
    rows = np.zeros((len(df), len(FEATS)), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        fanin, rx, stx, smean = store.update_and_get(r.step, r.nameOrig, r.nameDest, r.amount)
        rows[i, :9] = [np.log1p(r.amount), r.step % 24,
                       float(r.oldbalanceOrg > 0 and r.newbalanceOrig == 0 and r.amount >= 0.99 * r.oldbalanceOrg),
                       np.log1p(r.oldbalanceOrg), float(r.nameDest.startswith("M")),
                       fanin, rx, stx, r.amount / max(smean, 1.0)]
        rows[i, 9 + TYPES.index(r.type)] = 1
    return rows


def train():
    from fedshield.experiments.common import class_weights, run_fed
    from fedshield.eval.metrics import pick_threshold, scores
    from fedshield.fl.federation import Bank
    from fedshield.fl.gbdt import GBDTParams
    t0 = time.time()
    df = pd.read_csv(PAYSIM_FILE)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    # keep all fraud + a 4% sample of legitimate traffic (keeps featurisation tractable)
    rng = np.random.RandomState(SEED)
    keep = (df["isFraud"] == 1) | (rng.rand(len(df)) < 0.04)
    df = df[keep].reset_index(drop=True)
    X = featurise(df)
    y = df["isFraud"].to_numpy()
    bank = (pd.util.hash_array(df["nameOrig"].to_numpy()) % 3).astype(int)
    tr, va, te = df.step < 400, (df.step >= 400) & (df.step < 500), df.step >= 500
    w = class_weights(y)
    banks = [Bank(BANKS[b], X[tr & (bank == b)], y[tr & (bank == b)], w[tr & (bank == b)],
                  X[va & (bank == b)], y[va & (bank == b)]) for b in range(3)]
    params = GBDTParams(n_trees=120, max_depth=5, learning_rate=0.1, n_bins=64, seed=SEED)
    model, log = run_fed(banks, params, FEATS, secure_agg=True, byzantine=True, weighting=True)
    thr = pick_threshold(model.predict_proba(X[va]), y[va])
    res = scores(model.predict_proba(X[te]), y[te], thr)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    events = df[te].copy()
    (WORK / "models").mkdir(exist_ok=True)
    with open(WORK / "models" / "stream.pkl", "wb") as fh:
        pickle.dump({"model": model, "threshold": thr, "metrics": res}, fh)
    events.to_parquet(WORK / "models" / "stream_events.parquet", index=False)
    print(f"stream model saved ({time.time() - t0:.0f}s); NOTE: PaySim metrics are not a FedShield result")


class StreamScorer:
    def __init__(self, art, events):
        self.model, self.thr = art["model"], art["threshold"]
        self.events_df = events
        self.store = FeatureStore()

    @classmethod
    def load(cls):
        with open(WORK / "models" / "stream.pkl", "rb") as fh:
            art = pickle.load(fh)
        ev = pd.read_parquet(WORK / "models" / "stream_events.parquet")
        return cls(art, ev)

    def events(self):
        # enrich the replay: ~1 fraud per 25 events so the demo shows alerts
        rng = np.random.RandomState(int(time.time()) % 10000)
        fr = self.events_df[self.events_df.isFraud == 1]
        ok = self.events_df[self.events_df.isFraud == 0]
        while True:
            chunk = pd.concat([ok.sample(240, random_state=rng), fr.sample(10, random_state=rng)])
            for r in chunk.sort_values("step").itertuples(index=False):
                yield r

    def score(self, r):
        x = featurise(pd.DataFrame([r._asdict()]), self.store)
        p = float(self.model.predict_proba(x)[0])
        c = self.model.contributions(x)[0]
        top = [FEATS[i] for i in np.argsort(-c)[:3] if c[i] > 0]
        decision = "BLOCK" if p >= max(0.9, self.thr) else ("STEP-UP" if p >= self.thr else "ALLOW")
        return {"step": int(r.step), "type": DISPLAY[r.type], "amount": float(r.amount),
                "sender": r.nameOrig[-6:], "receiver": r.nameDest[-6:],
                "bank": BANKS[int(pd.util.hash_array(np.array([r.nameOrig]))[0] % 3)],
                "score": p, "decision": decision, "reasons": top, "is_fraud": int(r.isFraud),
                "receiver_fanin_24h": float(x[0, FEATS.index("dest_fanin_24h")])}


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
