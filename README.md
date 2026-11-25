# FedShield

**Federated mule-account intelligence for Indian banks.** Banks train one mule-detection
model together without sharing any customer data. Each bank sends only masked gradient
histograms, an aggregator that sees nothing individual combines them, and a hashed suspect
registry links accounts across banks without revealing identities.

Built on the RBIH NFPC Phase-2 data, with IBM AMLworld and BAF plugged into the same core
through one adapter contract.

<!-- RESULTS:START -->
## Results (NFPC Phase-2, 96,091 labelled accounts, 3 simulated banks split by state, leak-free)

| | AUPRC |
|---|---|
| Each bank alone, on its own test accounts (mean) | 0.756 |
| **FedShield** (secure aggregation + gates + consortium registry), same test accounts | **0.770** |
| Centralised (all data pooled; not allowed in reality) | 0.773 |

- **Final model, held-out test:** AUC 0.945, AUPRC 0.769, precision 0.83, recall 0.81, F1 0.82. Of the top 1% of accounts by score, 87% are mules.
- **Federation catches what banks miss alone:** 37 held-out mules caught that the bank's own model missed, against 6 lost. Bank-C, the smallest bank, gains +0.036 AUPRC.
- **Stable over seeds:** FedShield beats local-only in all 3 seeds (+0.008 to +0.030).
- **Banks with different fraud patterns:** FedShield 0.781 vs centralised 0.782.
- **Malicious bank:**
  - Gradient scaling: AUPRC 0.028 with the gates off, 0.774 with them on.
  - Label flipping: 0.697 with the gates off, 0.774 with them on.
- **Fairness:** the highest false-positive rate is at most 1.63× the lowest across rural, semi-urban and urban branches. By gender it's 1.04×.
- **Leakage check** (`experiments/leakage.py`):
  - `days_since_kyc` was a label proxy. Post-flag re-KYC lifted static-only AUPRC from 0.38 to 0.68. It is now blocked.
  - With transaction features recomputed using only data before each cutoff: 0.896 against 0.920 full-window, on a sample with an 18% mule rate.
- **Differential privacy:** expensive at this scale. A smaller model scores 0.652 without DP and 0.198–0.221 at ε 16–64. Shown as an honest limitation.
- **Label shield:** 1,907 doubtful mule labels down-weighted: 800 flagged outside the transaction window, 705 "Routine Investigation", 245 with no reason, 157 flagged before any activity.
- **Submission:** `results/submission/submission.csv` covers 64,062 test accounts. 1,776 are predicted mules, each with a suspicious window from its peak 7-day burst (median 10 days).
<!-- RESULTS:END -->

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): protocol, modules, leakage policy, memory discipline
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Vercel (UI), API hosting, Docker Compose, TLS-inspecting networks
- [docs/SESSION_LOG.md](docs/SESSION_LOG.md): how this was built, the decisions and the final numbers
- [results/FedShield_onepager.html](results/FedShield_onepager.html): the one-page summary (open it and print to PDF)

## Deploy the UI to Vercel

Import the repo, set **Root Directory = `ui`** and the env var **`VITE_API_BASE=https://<api-host>`**, then deploy.
The API can't run on Vercel (it needs model artefacts and WebSockets). Host it as a container or behind a tunnel; see [DEPLOYMENT](docs/DEPLOYMENT.md).

## What's in the box

| Layer | Where | What it does |
|---|---|---|
| Ingest | `fedshield/ingest/` | DuckDB out-of-core ingest. NFPC: 398M transactions joined with `transactions_additional`, sharded by account |
| Features | `fedshield/features/nfpc_v2.py` | MuleHunter feature families (rolling 1h/12h/1d/7d peaks, channel counts, burst ratios, concurrency, digit patterns, MCC outliers, unique counterparties). Batched, memory-capped, resumable |
| Label shield | `fedshield/features/nfpc_master.py` | Down-weights doubtful mule labels: flag dates outside the data window, "Routine Investigation", missing reason, flag before any activity |
| Leakage guards | `NON_FEATURES`, `BLOCKED`, `fedshield/adapters.py`, `experiments/leakage.py` | Freeze/flag/alert fields blocked. `days_since_kyc` blocked after the leakage check showed post-flag re-KYC. Point-in-time re-extraction to verify |
| Federation | `fedshield/fl/` | Histogram GBDT (numba), lossless against pooled training. Pairwise-masked secure aggregation, zCDP accountant, two Byzantine gates |
| Registry | `fedshield/registry.py` | HMAC-hashed suspect registry. Rare shared counterparties and devices, never the account's own label |
| Patterns | `fedshield/patterns/library.py` | Typology states (emerging / confirmed / dormant / retired) and versioned model snapshots. EFRM, AML and I4C evidence feeds are simulated |
| API | `fedshield/api/` | FastAPI with 5 role personas and **server-enforced scope**, explanations, ego graphs, audit log, live federation stream |
| Stream | `fedshield/stream/` | Online scorer with a Redis feature store, Redpanda producer and consumer (in-process fallback) |
| UI | `ui/` | React + TypeScript + Tailwind + Cytoscape + Recharts |

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin on Linux/Mac
# 1. data → features (≈ 70 min for NFPC on 8 threads, ~2 GB RAM)
python -m fedshield.ingest.nfpc
python -m fedshield.features.nfpc_v2 && python -m fedshield.features.nfpc_v2 --merge
# 2. experiments and artefacts
python -m fedshield.experiments.nfpc full
python -m fedshield.experiments.leakage
python -m fedshield.patterns.library full
python -m fedshield.experiments.demo_prep
python -m fedshield.experiments.submission
# 3. app
cd ui && npm ci && npm run build && cd ..
uvicorn fedshield.api.server:app --port 8000        # http://localhost:8000
```

Memory: every heavy step goes through DuckDB with `FEDSHIELD_DUCKDB_MEM` (default 2–3 GB) and spills to `work/duckdb_tmp`.

### Docker (consortium topology)

```bash
python scripts/export_partitions.py     # each bank's private partition
docker compose up --build               # 3 bank nodes, aggregator, Redis, Redpanda, replayer, API/UI on :8000
```

Each bank container mounts **only its own** partition. The aggregator talks to the banks over HTTP and only ever receives masked vectors.

### Kaggle

`python scripts/make_notebooks.py` writes `notebooks/01_nfpc_fedshield.ipynb`, `02_amlworld_fedshield.ipynb` and `03_baf_adapter.ipynb`.
Attach the dataset and this repo as a Kaggle dataset, then Run All. The notebooks call the same modules.

## Demo: 5 personas (password `fedshield`)

| User | Sees | Cannot see |
|---|---|---|
| `bank_a`, `bank_b`, `bank_c` | Own alerts, scores, explanations, money-flow graph (other banks' accounts hashed) | Any other bank's data. Try **Privacy probe**: the server returns 403 |
| `admin` | Live federation: packets, gates, privacy ledger, learning curve. Deploys the global model | Any account or label |
| `rbi` | Cross-bank clusters (pseudonymised), alerts by bank, pattern library, audit trail | Raw customer data or identities |

**Two-minute script**
1. `admin` → **Reset banks to local**.
2. `bank_b` → pick an account from the **Watch list**. The local model scores it below its threshold, so it isn't flagged.
3. `admin` → **Run federation** (NFPC). Packets fly, set a malicious bank and the gate excludes it, and the model deploys when the run finishes.
4. `bank_b` → same account. **Federation effect** panel: local score → federated score, now flagged, with reasons.
5. `rbi` → the cross-bank cluster and the audit trail of the officer's action.

The watch-list accounts are real held-out mules that the bank's own model missed and the federated model caught (`work/demo_cases.json`). Nothing is staged.

## Add a dataset

```python
from fedshield.adapters import Prepared, adapter

@adapter("mybank")
def _mine():
    f = ...  # one row per account/application: feature columns + y, bank, split, entity
    return Prepared("mybank", "account", f, feature_names)
```

It immediately works in federation, gates, DP, evaluation, the live pipeline and the UI dataset picker.

## Honest limits

- Banks are **simulated** partitions of one dataset (by state). Real banks would differ more, which helps federation.
- EFRM, AML and I4C evidence feeds are **simulated** from labels, because no public data exists.
- Explanations are tree-path (Saabas) attributions, a fast approximation to TreeSHAP.
- Formal DP is only usable at loose budgets at this data size (see the DP sweep).
- Demo auth is for demonstration: hardcoded personas and HMAC tokens, not an IdP.
