"""FedShield federation: banks + neutral aggregator running the histogram GBDT
protocol with secure aggregation, differential privacy, two-stage Byzantine
validation and AUC-PR-weighted contributions.

Layer mapping (one-pager):
  L3 Two-stage Byzantine validation
     Gate 1 (every tree): each bank reports a low-dimensional "gradient
            direction" vector from its own root histogram; a bank whose
            direction disagrees with the robust consensus (cosine < tau) sits
            the tree out.
     Gate 2 (start): peer validation — each bank's small local probe model is
            scored by the other banks on their own validation rows (only the
            model crosses the boundary). Banks with outlying peer AUC-PR are
            rejected.
  L4 AUC-PR dynamic weighting: the same peer AUC-PR sets each accepted bank's
     contribution weight (gradient scale).
  L6 Versioned round management: every tree is a round; the log records the
     participants, weights, ε spent and per-bank validation AUC-PR.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score

from fedshield.fl.gbdt import (GBDTModel, GBDTParams, HistParty, apply_bins,
                               base_score_from_counts, build_tree, local_quantiles,
                               merge_edges)
from fedshield.fl.privacy import DPAccountant, SecureAggregator


@dataclass
class Bank:
    name: str
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    attack: str = None             # None | "label_flip" | "gradient_scale"


@dataclass
class FedConfig:
    params: GBDTParams = field(default_factory=GBDTParams)
    secure_agg: bool = True
    dp_epsilon: float = None       # None = no DP noise
    dp_delta: float = 1e-5
    dp_weight_clip: float = 5.0    # per-record weight cap when DP is on
    byzantine: bool = True
    gate1_tau: float = 0.0
    weighting: bool = True
    eval_every: int = 25
    verbose: bool = False
    on_event: object = None        # callable(dict): live pipeline telemetry for the UI


def _direction(hist_root):
    """Per-feature sign of association between bin index and gradient at the root."""
    G = hist_root[0, :, :, 0]                        # (F, nb)
    H = np.maximum(hist_root[0, :, :, 1], 1e-9)
    bins = np.arange(G.shape[1])
    m = H.sum(1, keepdims=True)
    bc = bins - (H * bins).sum(1, keepdims=True) / m
    v = (G * bc).sum(1)
    return v / (np.linalg.norm(v) + 1e-12)


def _probe_auprc(banks, seed):
    """Gate 2 / L4: each bank trains a small local model; peers score it."""
    import xgboost as xgb
    K = len(banks)
    M = np.full((K, K), np.nan)
    for i, b in enumerate(banks):
        y = (1 - b.y) if b.attack == "label_flip" else b.y
        if b.attack == "gradient_scale":
            y = 1 - y
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        mdl = xgb.XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.1,
                                scale_pos_weight=spw, tree_method="hist",
                                n_jobs=4, random_state=seed, verbosity=0)
        mdl.fit(b.X, y)
        for j, o in enumerate(banks):
            if j != i and o.y_val.sum() > 0:
                M[i, j] = average_precision_score(o.y_val, mdl.predict_proba(o.X_val)[:, 1])
    return np.nanmean(M, axis=1)


def federate(banks, cfg: FedConfig, feature_names, X_eval=None, y_eval=None, edges=None):
    p = cfg.params
    rng = np.random.RandomState(p.seed)
    t0 = time.time()
    log = {"rounds": [], "gate2": {}, "weights": {}, "config": {}}

    # ---- federated binning (shared quantile sketch, no rows exchanged)
    if edges is None:
        edges = merge_edges([local_quantiles(b.X, p.n_bins) for b in banks], [len(b.X) for b in banks])
    log["edges"] = edges

    # ---- Gate 2 + AUC-PR weights
    names = [b.name for b in banks]
    accepted = list(names)
    weights = {n: 1.0 for n in names}
    if cfg.byzantine or cfg.weighting:
        peer = _probe_auprc(banks, p.seed)
        med = np.median(peer)
        mad = np.median(np.abs(peer - med)) + 1e-6
        for b, s in zip(banks, peer):
            z = (s - med) / (1.4826 * mad)
            reject = cfg.byzantine and (z < -3.0 or s < 0.5 * med)
            log["gate2"][b.name] = {"peer_auprc": float(s), "robust_z": float(z), "rejected": bool(reject)}
            if reject:
                accepted.remove(b.name)
        if cfg.weighting:
            acc_scores = np.array([peer[names.index(n)] for n in accepted])
            w = acc_scores / acc_scores.mean()
            weights = {n: float(0.5 + 0.5 * wi) for n, wi in zip(accepted, w)}
    log["weights"] = weights
    emit = cfg.on_event or (lambda e: None)
    emit({"stage": "setup", "banks": {b.name: {"rows": int(len(b.y)), "positives": int(b.y.sum()),
                                               "attack": b.attack} for b in banks},
          "gate2": log["gate2"], "weights": weights, "accepted": accepted,
          "secure_agg": cfg.secure_agg, "dp_epsilon": cfg.dp_epsilon, "n_trees": p.n_trees})

    # ---- parties
    parties = {}
    for b in banks:
        w = b.w.copy()
        if cfg.dp_epsilon is not None:
            w = np.minimum(w, cfg.dp_weight_clip)
        hp = HistParty(b.name, apply_bins(b.X, edges), b.y, w, p, seed=p.seed + len(parties))
        hp.attack = b.attack
        hp.grad_scale = weights.get(b.name, 1.0)
        parties[b.name] = hp
    acc_parties = [parties[n] for n in accepted]
    pos = sum((hp.y * hp.w).sum() for hp in acc_parties)
    tot = sum(hp.w.sum() for hp in acc_parties)
    p.base_score = base_score_from_counts(pos, tot)
    for hp in parties.values():
        hp.raw[:] = p.base_score
    model = GBDTModel(p, edges, list(feature_names))

    # ---- DP calibration
    F = banks[0].X.shape[1]
    accountants = {n: DPAccountant(cfg.dp_delta) for n in names}
    sigma = None
    if cfg.dp_epsilon is not None:
        c = cfg.dp_weight_clip * max(weights.values())
        n_col = max(1, int(round(p.colsample * F)))
        sens = DPAccountant.sensitivity(n_col, c, 0.25 * c)
        sigma = DPAccountant.sigma_for_budget(cfg.dp_epsilon, cfg.dp_delta,
                                              p.n_trees * (p.max_depth + 1), sens)
        log["config"]["dp_sigma"] = sigma
        log["config"]["dp_sensitivity"] = sens

    for t in range(p.n_trees):
        # Gate 1: direction consensus on this tree's gradients
        tree_parties = list(acc_parties)
        gate1_out, cos = [], {}
        if cfg.byzantine and len(tree_parties) >= 3:
            for hp in tree_parties:
                hp.start_tree()
            dirs = {hp.name: _direction(hp.histograms([0])) for hp in tree_parties}  # F-dim summary only
            cos = {}
            for hp in tree_parties:
                others = np.median([dirs[o.name] for o in tree_parties if o is not hp], axis=0)
                cos[hp.name] = float(dirs[hp.name] @ others / (np.linalg.norm(others) + 1e-12))
            med = float(np.median(list(cos.values())))
            # outlier only: clearly opposed to consensus AND far below the typical agreement
            cand = sorted((c, n) for n, c in cos.items() if c < cfg.gate1_tau and c < med - 0.5)
            gate1_out = [n for _, n in cand[: (len(tree_parties) - 1) // 2]]
            tree_parties = [hp for hp in tree_parties if hp.name not in gate1_out]
        emit({"stage": "gate1", "round": t + 1, "cos": cos,
              "rejected": gate1_out, "participants": [hp.name for hp in tree_parties]})
        sa = SecureAggregator([hp.name for hp in tree_parties], session_seed=t) if cfg.secure_agg else None

        def aggregate(hists, _tp=tree_parties, _sa=sa):
            shape = hists[0].shape
            out = []
            for hp, h in zip(_tp, hists):
                sent = np.zeros(h.shape[1], bool)
                sent[hp.allowed] = True
                h = np.where(sent[None, :, None, None], h, 0.0)   # unsent features are never revealed
                if sigma is not None:
                    noise = np.random.normal(0, sigma, size=h.shape)
                    h = h + np.where(sent[None, :, None, None], noise, 0.0)
                    sens_q = DPAccountant.sensitivity(len(hp.allowed), cfg.dp_weight_clip * hp.grad_scale,
                                                      0.25 * cfg.dp_weight_clip * hp.grad_scale)
                    accountants[hp.name].spend(sens_q, sigma)
                out.append(h)
            emit({"stage": "histograms", "round": t + 1,
                  "sent": {hp.name: {"bytes": int(h.nbytes), "features": int(len(hp.allowed))}
                           for hp, h in zip(_tp, out)},
                  "masked_preview": {hp.name: [int(x) % 100000 for x in _sa.mask(hp.name, h).ravel()[:6]]
                                     for hp, h in zip(_tp, out)} if _sa is not None else {}})
            if _sa is not None:
                return _sa.unmask_sum([_sa.mask(hp.name, h) for hp, h in zip(_tp, out)], shape)
            return np.sum(out, axis=0)

        tree = build_tree(tree_parties, aggregate, p, rng)
        for hp in parties.values():
            if hp not in tree_parties:
                hp.finish_tree(tree)
        model.trees.append(tree)

        rec = {"round": t + 1, "participants": [hp.name for hp in tree_parties],
               "gate1_rejected": gate1_out,
               "epsilon": {n: round(a.epsilon, 4) for n, a in accountants.items()}}
        if (t + 1) % cfg.eval_every == 0 or t == p.n_trees - 1:
            for b in banks:
                if b.y_val.sum() > 0:
                    rec[f"val_auprc_{b.name}"] = float(average_precision_score(
                        b.y_val, model.predict_proba(b.X_val)))
            if X_eval is not None:
                rec["eval_auprc"] = float(average_precision_score(y_eval, model.predict_proba(X_eval)))
            if cfg.verbose:
                print(f"    tree {t + 1}/{p.n_trees}  {rec.get('eval_auprc', '')}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        log["rounds"].append(rec)
        top = int(np.argmax(np.bincount(tree.feat[tree.feat >= 0], weights=tree.gain[tree.feat >= 0],
                                         minlength=len(feature_names)))) if (tree.feat >= 0).any() else None
        emit({"stage": "tree", **{k: v for k, v in rec.items()}, "n_trees": p.n_trees,
              "leaves": int((tree.feat < 0).sum()), "top_feature": None if top is None else feature_names[top],
              "elapsed": time.time() - t0})
    log["config"].update({"secure_agg": cfg.secure_agg, "dp_epsilon": cfg.dp_epsilon,
                          "byzantine": cfg.byzantine, "weighting": cfg.weighting,
                          "accepted": accepted, "seconds": time.time() - t0})
    log["epsilon_final"] = {n: a.epsilon for n, a in accountants.items()}
    emit({"stage": "done", "seconds": time.time() - t0, "epsilon_final": log["epsilon_final"]})
    return model, log
