"""Privacy layers for the histogram protocol.

SecureAggregator — pairwise additive masking (Bonawitz et al. 2017, without the
  dropout-recovery part). Each pair of banks derives a shared mask from a
  pairwise seed; bank i adds it, bank j subtracts it. Values are fixed-point
  encoded into uint64 (mod 2^64) so masks cancel exactly. The aggregator only
  ever sees masked vectors and their (exact) sum.

DPAccountant — record-level Gaussian mechanism with zero-concentrated DP
  accounting. Each tree level is one query on the bank's histograms; one record
  falls in exactly one bin per feature, so with |g| <= c_g, |h| <= c_h the L2
  sensitivity of a level query is sqrt(F * (c_g^2 + c_h^2)).
  rho per query = Δ² / (2σ²); rho composes additively;
  ε(δ) = rho + 2·sqrt(rho·ln(1/δ))   (Bun & Steinke 2016).
"""
import hashlib
import math

import numpy as np

SCALE = 2 ** 24   # fixed-point precision for secure aggregation


class SecureAggregator:
    def __init__(self, party_names, session_seed=0):
        self.names = list(party_names)
        self.session = session_seed
        self.round = 0
        self.log = []

    def _seed(self, a, b):
        key = f"{min(a, b)}|{max(a, b)}|{self.session}|{self.round}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")

    def mask(self, name, vec):
        """Run by the bank: encode + add pairwise masks."""
        enc = np.round(vec.ravel() * SCALE).astype(np.int64).view(np.uint64)
        for other in self.names:
            if other == name:
                continue
            m = np.random.default_rng(self._seed(name, other)).integers(
                0, 2 ** 63, size=enc.size, dtype=np.uint64) * np.uint64(2)
            enc = enc + m if name < other else enc - m
        return enc

    def unmask_sum(self, masked, shape):
        """Run by the aggregator: sum masked vectors; masks cancel."""
        tot = np.zeros_like(masked[0])
        for m in masked:
            tot = tot + m
        self.round += 1
        return (tot.view(np.int64).astype(np.float64) / SCALE).reshape(shape)


class DPAccountant:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.rho = 0.0
        self.queries = 0

    @staticmethod
    def sensitivity(n_features, c_g, c_h):
        return math.sqrt(n_features * (c_g ** 2 + c_h ** 2))

    def spend(self, sensitivity, sigma):
        self.rho += sensitivity ** 2 / (2 * sigma ** 2)
        self.queries += 1

    @property
    def epsilon(self):
        if self.rho == 0:
            return 0.0
        return self.rho + 2 * math.sqrt(self.rho * math.log(1 / self.delta))

    @staticmethod
    def sigma_for_budget(eps, delta, n_queries, sensitivity):
        """Noise multiplier so that n_queries Gaussian queries hit (eps, delta)."""
        # solve rho + 2 sqrt(rho L) = eps for rho
        L = math.log(1 / delta)
        r = (math.sqrt(L + eps) - math.sqrt(L)) ** 2
        rho_q = r / n_queries
        return sensitivity / math.sqrt(2 * rho_q)
