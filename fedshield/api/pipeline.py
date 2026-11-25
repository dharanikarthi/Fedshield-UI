"""Live federation run for the UI: a real (small) FedShield training run whose every
stage — quantile sketch, Gate 2, Gate 1, masked histograms, secure sum, tree — is
emitted as an event the browser animates. Works for any registered adapter."""
import json
import threading
import time

import numpy as np

from fedshield.adapters import load
from fedshield.fl.federation import Bank, FedConfig, federate
from fedshield.fl.gbdt import GBDTParams


class LiveRun:
    def __init__(self):
        self.events, self.lock, self.thread, self.cfg = [], threading.Lock(), None, None
        self.on_done = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def _push(self, e):
        e = json.loads(json.dumps(e, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        with self.lock:
            e["seq"] = len(self.events)
            e["t"] = time.time()
            self.events.append(e)

    def since(self, seq):
        with self.lock:
            return self.events[seq:]

    def start(self, dataset="nfpc", sample=30000, n_trees=40, max_depth=4, attack=None,
              attack_bank=None, dp_epsilon=None, byzantine=True, secure_agg=True, pace=0.15):
        if self.running:
            return False
        self.events = []
        self.cfg = dict(dataset=dataset, sample=sample, n_trees=n_trees, attack=attack,
                        dp_epsilon=dp_epsilon, byzantine=byzantine, secure_agg=secure_agg)
        self.thread = threading.Thread(target=self._run, args=(dict(self.cfg, max_depth=max_depth,
                                       attack_bank=attack_bank, pace=pace),), daemon=True)
        self.thread.start()
        return True

    def _run(self, c):
        try:
            self._push({"stage": "load", "dataset": c["dataset"], "status": "loading"})
            p = load(c["dataset"], sample=c["sample"])
            f = p.frame
            tr, te = f[f["split"] == "train"], f[f["split"] == "test"]
            names = p.banks()
            atk_bank = c["attack_bank"] or (names[-1] if c["attack"] else None)
            banks = []
            for n in names:
                d = tr[tr["bank"] == n]
                y = d["y"].to_numpy().astype(int)
                pos = max(y.mean(), 1e-4)
                w = np.where(y == 1, min(0.5 / pos, 50.0), 1.0)
                v = d.sample(min(len(d), 3000), random_state=0)
                banks.append(Bank(n, d[p.features].to_numpy(), y, w, v[p.features].to_numpy(),
                                  v["y"].to_numpy().astype(int), attack=c["attack"] if n == atk_bank else None))
            self._push({"stage": "load", "dataset": p.name, "unit": p.unit, "status": "ready",
                        "rows": int(len(f)), "features": len(p.features), "banks": names,
                        "test_rows": int(len(te))})
            pace = c["pace"]

            def emit(e):
                self._push(e)
                if e["stage"] in ("histograms", "tree"):
                    time.sleep(pace)          # so the browser can show each hop

            params = GBDTParams(n_trees=c["n_trees"], max_depth=c["max_depth"], learning_rate=0.15,
                                colsample=0.7, subsample=0.8)
            cfg = FedConfig(params=params, secure_agg=c["secure_agg"], dp_epsilon=c["dp_epsilon"],
                            byzantine=c["byzantine"], weighting=True, eval_every=5, on_event=emit)
            federate(banks, cfg, p.features, te[p.features].to_numpy(), te["y"].to_numpy().astype(int))
            if self.on_done and p.name == "nfpc":
                self.on_done()
                self._push({"stage": "deployed", "message": "global model deployed to member banks"})
        except Exception as ex:          # surfaced in the UI rather than killing the server
            self._push({"stage": "error", "message": f"{type(ex).__name__}: {ex}"})


LIVE = LiveRun()
