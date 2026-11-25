"""Event bus for the payment stream: Redpanda via its HTTP proxy (Pandaproxy).

Producer : python -m fedshield.stream.bus replay        (REDPANDA_PROXY must be set)
Consumer : the API's /api/stream websocket reads from the bus when REDPANDA_PROXY is
           set, otherwise it replays in-process (same events, no broker).
"""
import os
import sys
import time
import uuid
from collections import namedtuple

TOPIC = "payments"
KJSON = "application/vnd.kafka.json.v2+json"


def proxy():
    return os.environ.get("REDPANDA_PROXY")


def produce_forever(rate=20.0, batch=10):
    import httpx

    from fedshield.stream.scorer import StreamScorer
    url = f"{proxy().rstrip('/')}/topics/{TOPIC}"
    c = httpx.Client(timeout=10)
    src = StreamScorer.load().events()
    sent, t0 = 0, time.time()
    while True:
        recs = [{"value": {k: (v.item() if hasattr(v, "item") else v) for k, v in next(src)._asdict().items()}}
                for _ in range(batch)]
        try:
            c.post(url, json={"records": recs}, headers={"Content-Type": KJSON}).raise_for_status()
            sent += batch
        except Exception as e:      # broker still starting: back off and retry
            print(f"produce failed ({e}); retrying", flush=True)
            time.sleep(2)
            continue
        if sent % 1000 == 0:
            print(f"  produced {sent:,} events  {sent / (time.time() - t0):.0f}/s", flush=True)
        time.sleep(batch / rate)


class Consumer:
    """Async consumer (one per websocket) using a fresh consumer instance on 'latest'."""

    def __init__(self):
        import httpx
        self.c = httpx.AsyncClient(timeout=15)
        self.base = None
        self.group = f"api-{uuid.uuid4().hex[:8]}"

    async def __aenter__(self):
        p = proxy().rstrip("/")
        r = await self.c.post(f"{p}/consumers/{self.group}",
                              json={"name": self.group, "format": "json", "auto.offset.reset": "latest"},
                              headers={"Content-Type": KJSON})
        r.raise_for_status()
        self.base = r.json()["base_uri"]
        await self.c.post(f"{self.base}/subscription", json={"topics": [TOPIC]}, headers={"Content-Type": KJSON})
        return self

    async def records(self):
        r = await self.c.get(f"{self.base}/records", headers={"Accept": KJSON}, params={"timeout": 1000})
        r.raise_for_status()
        rows = [m["value"] for m in r.json()]
        if not rows:
            return []
        T = namedtuple("Ev", list(rows[0].keys()))
        return [T(**v) for v in rows]

    async def __aexit__(self, *a):
        try:
            await self.c.delete(self.base, headers={"Content-Type": KJSON})
        finally:
            await self.c.aclose()


if __name__ == "__main__":
    if sys.argv[1:] == ["replay"]:
        assert proxy(), "set REDPANDA_PROXY, e.g. http://redpanda:8082"
        produce_forever(float(os.environ.get("REPLAY_RATE", 20)))
