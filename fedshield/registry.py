"""Cross-bank suspect registry (privacy-preserving intelligence sharing).

Banks publish keyed hashes (HMAC-SHA256 with a consortium key held by the
banks, not the registry operator) of accounts they have confirmed or flagged.
The registry stores only digests + the publishing bank + evidence source, so
the operator (RBI/NPCI/DPIP role) cannot enumerate or reverse account numbers.
A bank looks up the digests of its customers' counterparties — it learns only
"this counterparty was flagged by N other banks", never who they are elsewhere.
"""
import hashlib
import hmac
from collections import defaultdict


class SuspectRegistry:
    def __init__(self, consortium_key: bytes = b"fedshield-consortium-demo-key"):
        self._key = consortium_key
        self.entries = defaultdict(dict)     # digest -> {bank: source}

    def digest(self, account_key) -> str:
        return hmac.new(self._key, str(account_key).encode(), hashlib.sha256).hexdigest()[:32]

    def publish(self, bank, account_keys, source="EFRM"):
        for k in account_keys:
            self.entries[self.digest(k)][bank] = source

    def lookup(self, account_keys, exclude_bank=None):
        """Number of distinct OTHER banks (or any, if exclude_bank None) flagging each key."""
        out = []
        for k in account_keys:
            e = self.entries.get(self.digest(k))
            out.append(0 if not e else sum(1 for b in e if b != exclude_bank))
        return out

    def stats(self):
        by_src = defaultdict(int)
        for e in self.entries.values():
            for s in e.values():
                by_src[s] += 1
        return {"digests": len(self.entries), "by_source": dict(by_src)}
