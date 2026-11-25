import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fedshield.fl.federation import Bank, FedConfig, federate
from fedshield.fl.gbdt import GBDTParams
from fedshield.fl.privacy import DPAccountant, SecureAggregator
from fedshield.registry import SuspectRegistry


def _toy(n=6000, f=8, seed=0):
    r = np.random.RandomState(seed)
    X = r.randn(n, f).astype("float32")
    logit = 2 * X[:, 0] - 1.5 * X[:, 1] * (X[:, 2] > 0) - 3
    y = (r.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _banks(X, y, k=3):
    idx = np.array_split(np.arange(len(y)), k)
    w = np.where(y == 1, 5.0, 1.0)
    return [Bank(f"B{i}", X[s], y[s], w[s], X[s][:300], y[s][:300]) for i, s in enumerate(idx)]


def test_secure_aggregation_masks_cancel_exactly():
    sa = SecureAggregator(["A", "B", "C"], session_seed=7)
    vs = [np.random.RandomState(i).randn(5, 3) * 1000 for i in range(3)]
    masked = [sa.mask(n, v) for n, v in zip("ABC", vs)]
    # an individual masked vector reveals nothing recognisable
    assert np.corrcoef(masked[0].view(np.int64).astype(float), vs[0].ravel())[0, 1] < 0.5
    total = sa.unmask_sum(masked, (5, 3))
    assert np.allclose(total, sum(vs), atol=1e-5)


def test_federated_equals_pooled_with_same_bins():
    X, y = _toy()
    banks = _banks(X, y)
    p = dict(n_trees=15, max_depth=4, learning_rate=0.2, subsample=1.0, colsample=1.0)
    fed, log = federate(banks, FedConfig(params=GBDTParams(**p), secure_agg=True, byzantine=False,
                                         weighting=False), [f"f{i}" for i in range(8)])
    pool = Bank("pool", X, y, np.concatenate([b.w for b in banks]), X[:300], y[:300])
    cen, _ = federate([pool], FedConfig(params=GBDTParams(**p), secure_agg=False, byzantine=False,
                                        weighting=False), [f"f{i}" for i in range(8)], edges=log["edges"])
    assert abs(average_precision_score(y, fed.predict_proba(X)) -
               average_precision_score(y, cen.predict_proba(X))) < 1e-3


def test_gate_excludes_label_flipping_bank():
    X, y = _toy(9000)
    banks = _banks(X, y, 3)
    banks[2].attack = "label_flip"
    _, log = federate(banks, FedConfig(params=GBDTParams(n_trees=10, max_depth=3), byzantine=True,
                                       weighting=True), [f"f{i}" for i in range(8)])
    excluded = log["gate2"]["B2"]["rejected"] or sum("B2" in r["gate1_rejected"] for r in log["rounds"]) >= 5
    assert excluded


def test_dp_accountant_budget():
    sens = DPAccountant.sensitivity(10, 1.0, 0.25)
    sigma = DPAccountant.sigma_for_budget(8.0, 1e-5, 100, sens)
    acc = DPAccountant(1e-5)
    for _ in range(100):
        acc.spend(sens, sigma)
    assert acc.epsilon == pytest.approx(8.0, rel=1e-6)
    assert DPAccountant.sigma_for_budget(1.0, 1e-5, 100, sens) > sigma


def test_registry_hashes_and_excludes_publisher():
    reg = SuspectRegistry(b"k")
    reg.publish("Bank-A", ["ACCT_1"])
    assert "ACCT_1" not in str(dict(reg.entries))          # stores digests only
    assert reg.lookup(["ACCT_1", "ACCT_2"]) == [1, 0]
    assert reg.lookup(["ACCT_1"], exclude_bank="Bank-A") == [0]


def test_registry_features_never_use_own_label(tmp_path, monkeypatch):
    from fedshield.features import nfpc_master as nm
    # two accounts share a rare counterparty; only account 1 is a confirmed mule
    edges = pd.DataFrame({"acct": [1, 2, 3], "cp": [10, 10, 11], "is_credit": [True, True, True]})
    ips = pd.DataFrame({"acct": [1, 2], "ip": [5, 6]})
    monkeypatch.setattr(nm, "OUT", tmp_path)
    edges.to_parquet(tmp_path / "edges_full.parquet")
    ips.to_parquet(tmp_path / "ips_full.parquet")
    m = pd.DataFrame({"acct": [1, 2, 3]})
    f = nm.registry_features(m, np.array([0, 1, 1]), np.array([True, False, False]), "fed")
    assert f.loc[0, "reg_cp_flagged"] == 0        # the mule's own label does not flag itself
    assert f.loc[1, "reg_cp_flagged"] == 1        # its neighbour (other bank) is flagged
    loc = nm.registry_features(m, np.array([0, 1, 1]), np.array([True, False, False]), "local")
    assert loc.loc[1, "reg_cp_flagged"] == 0      # local scope: Bank-1 cannot see Bank-0's confirmation


def test_leakage_columns_blocked():
    from fedshield.experiments.nfpc import NON_FEATURES
    from fedshield.features.nfpc_master import BLOCKED
    for c in ["account_status", "freeze_date", "unfreeze_date", "mule_flag_date", "alert_reason", "flagged_by_branch",
              "days_since_kyc"]:
        assert c in BLOCKED or c in NON_FEATURES
