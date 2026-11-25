# Architecture

```
            ┌──────────── each bank (data never leaves) ────────────┐
 raw txns → │ DuckDB ingest → feature library → label shield        │
            │ HistParty: gradients → per-node histograms            │
            │ mask(histogram) with pairwise secure-aggregation seeds│──┐ masked uint64 vectors
            └───────────────────────────────────────────────────────┘  │
                                                                       ▼
            ┌──────────────── neutral aggregator ───────────────────┐
            │ Gate 2 (peer AUC-PR probe, start)                      │
            │ Gate 1 (gradient-direction consensus, per tree)        │
            │ unmask SUM only → best split → broadcast               │
            │ zCDP accountant (optional Gaussian noise)              │
            └────────────────────────────────────────────────────────┘
                      │ global model                     ▲ HMAC digests
                      ▼                                  │
            bank consoles (API, role-scoped)     suspect registry (rare counterparties, devices)
                      │
            RBI supervisory view (pseudonymised clusters, evidence, audit)
```

## Key modules

| Module | Role |
|---|---|
| `fedshield/adapters.py` | **The plug-in contract.** Each dataset becomes `Prepared(frame[features, y, bank, split, entity])`. A leakage guard runs at load |
| `fedshield/fl/gbdt.py` | Histogram GBDT: numba histogram kernel, quantile-sketch bin merging, Saabas attributions |
| `fedshield/fl/federation.py` | `federate()`: gates, contribution weights, secure aggregation, DP, live telemetry (`on_event`) |
| `fedshield/fl/privacy.py` | `SecureAggregator` (pairwise additive masks in uint64 fixed point), `DPAccountant` (zCDP) |
| `fedshield/fl/service.py` | Distributed mode: one HTTP service per bank, plus the aggregator CLI |
| `fedshield/registry.py`, `features/nfpc_master.registry_features` | Hashed consortium registry. The account's own label is always excluded |
| `fedshield/experiments/*` | `nfpc` (all sections), `aml`, `leakage`, `demo_prep`, `submission` |
| `fedshield/api/server.py` | REST + WebSocket. `auth.py` (personas, HMAC tokens, scope), `supervisor.py` (RBI clusters), `pipeline.py` (live run) |
| `fedshield/stream/` | Online scorer, Redis feature store, Redpanda bus |
| `ui/src/pages` | `Login`, `Console` (bank), `Pipeline` / `Federation` (admin), `Supervisor` (RBI), `Stream`, `Results` |

## Why a custom federated GBDT and not Flower?

Flower's simulation needs Ray, which is fragile on Windows. A histogram protocol is also exactly
what secure aggregation needs: the sum of the banks' histograms equals the pooled histogram, so the federated
model is **lossless** against centralised training on the same bins (`tests/test_core.py::test_federated_equals_pooled_with_same_bins`).
`fl/service.py` runs the same protocol across real processes and containers.

## Leakage policy

Blocked as features: `account_status`, `freeze_date`, `unfreeze_date`, `mule_flag_date`, `alert_reason`,
`flagged_by_branch`, address/passbook update dates, and **`days_since_kyc`**, since post-flag re-KYC turned out to leak the label.
`experiments/leakage.py` re-extracts transaction features using only data before each account's cutoff and
compares the two. Results are in `results/nfpc_leakage.json`.

## Memory discipline

Machine target: about 12 GB RAM shared with other applications. All heavy work is DuckDB with a memory cap
(`FEDSHIELD_DUCKDB_MEM`), on-disk spilling and per-batch checkpoints. Feature extraction is resumable after a kill.
The 110M-row IP table is never loaded into pandas.
