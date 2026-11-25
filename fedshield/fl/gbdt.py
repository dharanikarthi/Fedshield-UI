"""Horizontal federated gradient-boosted trees (histogram protocol).

Equivalent to XGBoost's `hist` algorithm, split across parties:

  bank        : holds rows; computes per-(node, feature, bin) gradient/hessian sums
  aggregator  : sums the banks' histograms (optionally secure-aggregated and
                DP-noised), picks splits, broadcasts the tree

No row, feature value or label leaves a bank — only binned G/H sums. With no
noise this is lossless: the federated model equals the centralised one built on
the pooled data with the same bins.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _hist_kernel(B, slot, g, h, k, nb):
    n, F = B.shape
    out = np.zeros((k, F, nb, 2))
    for f in prange(F):
        for i in range(n):
            s = slot[i]
            if s >= 0:
                b = B[i, f]
                out[s, f, b, 0] += g[i]
                out[s, f, b, 1] += h[i]
    return out


# ----------------------------------------------------------------------------- binning
def local_quantiles(X, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    with np.errstate(all="ignore"):
        return np.nanquantile(X, qs, axis=0)            # (n_bins-1, F)


def merge_edges(quantile_list, sizes):
    """Federated bin edges: size-weighted average of each bank's local quantiles."""
    w = np.asarray(sizes, float) / np.sum(sizes)
    q = np.nansum([wi * np.nan_to_num(qi, nan=0.0) for wi, qi in zip(w, quantile_list)], axis=0)
    q = np.sort(q, axis=0)
    return [np.unique(q[:, j]) for j in range(q.shape[1])]


def apply_bins(X, edges):
    """Missing values go to bin 0; observed values to 1..len(edges)+1."""
    B = np.zeros(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        col = X[:, j]
        b = np.searchsorted(e, col, side="right") + 1
        b[np.isnan(col)] = 0
        B[:, j] = b
    return B


# ----------------------------------------------------------------------------- params
@dataclass
class GBDTParams:
    n_trees: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    reg_lambda: float = 1.0
    min_child_weight: float = 5.0
    min_gain: float = 0.0
    subsample: float = 0.8
    colsample: float = 0.8
    n_bins: int = 64
    base_score: float = 0.0          # logit
    seed: int = 42


@dataclass
class Tree:
    feat: np.ndarray                 # (n_nodes,) feature of the split, -1 = leaf
    thr: np.ndarray                  # bin threshold: go left if bin <= thr
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray                # leaf value (already scaled by learning rate)
    gain: np.ndarray

    def predict_bins(self, B):
        node = np.zeros(len(B), dtype=np.int32)
        active = self.feat[node] >= 0
        while active.any():
            idx = np.where(active)[0]
            n = node[idx]
            go_left = B[idx, self.feat[n]] <= self.thr[n]
            node[idx] = np.where(go_left, self.left[n], self.right[n])
            active[idx] = self.feat[node[idx]] >= 0
        return self.value[node]


@dataclass
class GBDTModel:
    params: GBDTParams
    edges: list
    feature_names: list
    trees: list = field(default_factory=list)

    def raw_bins(self, B, n_trees=None):
        out = np.full(len(B), self.params.base_score, dtype=np.float64)
        for t in self.trees[:n_trees]:
            out += t.predict_bins(B)
        return out

    def predict_proba(self, X, n_trees=None):
        return _sigmoid(self.raw_bins(apply_bins(X, self.edges), n_trees))

    def feature_gain(self):
        g = np.zeros(len(self.feature_names))
        for t in self.trees:
            m = t.feat >= 0
            np.add.at(g, t.feat[m], t.gain[m])
        return g / max(g.sum(), 1e-12)

    def contributions(self, X):
        """Per-feature contribution (Saabas path attribution) — exact for each
        prediction, sums to raw score minus bias. Used for per-alert reasons."""
        B = apply_bins(X, self.edges)
        contrib = np.zeros((len(B), len(self.feature_names)))
        for t in self.trees:
            mean = _node_means(t)
            node = np.zeros(len(B), dtype=np.int32)
            for _ in range(self.params.max_depth + 1):
                f = t.feat[node]
                inner = f >= 0
                if not inner.any():
                    break
                idx = np.where(inner)[0]
                n = node[idx]
                nxt = np.where(B[idx, f[idx]] <= t.thr[n], t.left[n], t.right[n])
                np.add.at(contrib, (idx, f[idx]), mean[nxt] - mean[n])
                node[idx] = nxt
        return contrib


def _node_means(t):
    """Cover-free approximation: a node's mean = average of its children's means."""
    mean = t.value.astype(float).copy()
    for n in range(len(t.feat) - 1, -1, -1):
        if t.feat[n] >= 0:
            mean[n] = 0.5 * (mean[t.left[n]] + mean[t.right[n]])
    return mean


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------- party side
class HistParty:
    """A bank's side of the protocol. Holds binned data; never shares rows."""

    def __init__(self, name, B, y, w, params: GBDTParams, seed=0):
        self.name, self.B, self.y, self.w = name, B, y.astype(np.float64), w.astype(np.float64)
        self.p = params
        self.F = B.shape[1]
        self.nb = params.n_bins + 2
        self.raw = np.full(len(y), params.base_score)
        self.rng = np.random.RandomState(seed)
        self.grad_scale = 1.0                         # set by aggregator (contribution weight)
        self.attack = None                            # used by the Byzantine experiment

    def start_tree(self):
        p = _sigmoid(self.raw)
        y = self.y
        if self.attack == "label_flip":
            y = 1.0 - y
        g = (p - y) * self.w
        h = np.maximum(p * (1 - p), 1e-6) * self.w
        if self.attack == "gradient_scale":
            g = -10.0 * g
        rows = self.rng.rand(len(y)) < self.p.subsample
        self.g, self.h = g * rows * self.grad_scale, h * rows * self.grad_scale
        self.node = np.where(rows, 0, -1).astype(np.int32)

    def histograms(self, node_ids):
        """(len(node_ids), F, nb, 2) G/H sums for the requested open nodes."""
        k = len(node_ids)
        remap = -np.ones(int(self.node.max()) + 2 if len(self.node) else 1, dtype=np.int32)
        ids = np.asarray([n for n in node_ids if n < len(remap)], dtype=np.int64)
        remap[ids] = np.searchsorted(np.asarray(node_ids), ids)
        slot = np.where(self.node >= 0, remap[np.maximum(self.node, 0)], -1).astype(np.int32)
        return _hist_kernel(self.B, slot, self.g, self.h, k, self.nb)

    def apply_splits(self, splits):
        """splits: {node: (feat, thr, left, right)}"""
        for n, (f, t, l, r) in splits.items():
            m = self.node == n
            go_left = self.B[:, f] <= t
            self.node[m & go_left] = l
            self.node[m & ~go_left] = r

    def finish_tree(self, tree: Tree):
        self.raw += tree.predict_bins(self.B)


# ----------------------------------------------------------------------------- aggregator side
def best_splits(hist, params: GBDTParams, allowed_feats):
    """hist: (k, F, nb, 2) aggregated. Returns per node (gain, feat, thr, G, H)."""
    lam, mcw = params.reg_lambda, params.min_child_weight
    G, H = hist[..., 0], np.maximum(hist[..., 1], 0)
    GL, HL = np.cumsum(G, axis=2), np.cumsum(H, axis=2)
    Gt, Ht = GL[:, :, -1:], HL[:, :, -1:]
    GR, HR = Gt - GL, Ht - HL
    gain = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - Gt ** 2 / (Ht + lam)
    bad = (HL < mcw) | (HR < mcw)
    gain = np.where(bad, -np.inf, gain)
    mask = np.zeros(gain.shape[1], bool)
    mask[allowed_feats] = True
    gain[:, ~mask, :] = -np.inf
    gain = gain[:, :, :-1]                               # last bin cannot be a threshold
    flat = gain.reshape(len(gain), -1).argmax(1)
    f, b = np.unravel_index(flat, gain.shape[1:])
    best = gain[np.arange(len(gain)), f, b]
    tot_G = G[:, 0, :].sum(1)
    tot_H = H[:, 0, :].sum(1)
    return best, f, b, tot_G, tot_H


def build_tree(parties, aggregate, params: GBDTParams, rng, on_level=None):
    """Grow one tree level by level. `aggregate(list_of_hist) -> hist` hides how
    the sum is computed (plain, secure aggregation, DP-noised...)."""
    for p in parties:
        p.start_tree()
    F = parties[0].F
    n_col = max(1, int(round(params.colsample * F)))
    allowed = np.sort(rng.choice(F, n_col, replace=False))
    for p in parties:
        p.allowed = allowed          # only these features' histograms leave the bank
    feat, thr, left, right, value, gain = [-1], [0], [-1], [-1], [0.0], [0.0]
    open_nodes = [0]
    for depth in range(params.max_depth + 1):
        if not open_nodes:
            break
        hist = aggregate([p.histograms(open_nodes) for p in parties])
        g, f, b, Gt, Ht = best_splits(hist, params, allowed)
        if on_level:
            on_level(depth, len(open_nodes))
        splits, nxt = {}, []
        for i, n in enumerate(open_nodes):
            leaf_val = -params.learning_rate * Gt[i] / (Ht[i] + params.reg_lambda)
            if depth == params.max_depth or not np.isfinite(g[i]) or g[i] <= params.min_gain:
                value[n] = leaf_val
                continue
            l, r = len(feat), len(feat) + 1
            for _ in range(2):
                feat.append(-1); thr.append(0); left.append(-1); right.append(-1)
                value.append(0.0); gain.append(0.0)
            feat[n], thr[n], left[n], right[n], gain[n] = int(f[i]), int(b[i]), l, r, float(g[i])
            splits[n] = (int(f[i]), int(b[i]), l, r)
            nxt += [l, r]
        for p in parties:
            p.apply_splits(splits)
        open_nodes = nxt
    tree = Tree(*(np.asarray(a) for a in (feat, thr, left, right, value, gain)))
    tree.feat = tree.feat.astype(np.int32); tree.thr = tree.thr.astype(np.int32)
    tree.left = tree.left.astype(np.int32); tree.right = tree.right.astype(np.int32)
    for p in parties:
        p.finish_tree(tree)
    return tree


def base_score_from_counts(pos_w, tot_w):
    p = np.clip(pos_w / max(tot_w, 1e-12), 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))
