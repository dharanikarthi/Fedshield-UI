# Build log: how FedShield was developed

A readable record of the build conversation (Nov 2026), so decisions can be revisited.
The full raw transcript is kept **locally** in `chat_history/` (gitignored, because it contains machine
paths and session metadata).

## 1. Starting point
- The original PoC (`fedshield (1).ipynb`, `FedShield.pdf`) claimed AUPRC 0.527 / 0.542. The notebook really showed 0.282,
  and the shipped CSVs came from an older "FINDA" notebook. Decision: rebuild, and report only numbers that the code reproduces.

## 2. Decisions
- **Datasets share no accounts**, so there's no federation across datasets. Instead: one federated core, one adapter per dataset.
  - NFPC is the main result. IBM AMLworld covers cross-bank laundering. BAF covers onboarding fraud and fairness. PaySim drives the live stream.
- **Run heavy jobs locally with DuckDB**, and use Kaggle only for reproducibility notebooks.
- **UI in React** (not Streamlit), dark theme, validated colour-blind-safe palette.
- **Custom histogram GBDT instead of Flower** (Ray is fragile on Windows). It's lossless against pooled training.
- **Build in one stretch**, without a phased roadmap.

## 3. Major steps
1. Ingest NFPC: 397,875,323 transactions joined with transactions_additional and sharded by account.
2. Features from the RBIH MuleHunter features document: rolling peaks, channels, bursts, concurrency, digit patterns, MCC outliers.
   - The extractor is memory-capped and resumable. It survived several out-of-memory kills, because other apps held most of the RAM.
3. Federation with secure aggregation, two Byzantine gates, a zCDP accountant, the hashed registry and the label shield.
4. Experiments: local vs FedShield vs centralised, registry, pattern skew, attacks, DP sweep, fairness, seeds, windows.
5. **Leakage audit**: `days_since_kyc` leaked the label through post-flag re-KYC (57% of mules had KYC in 2025, against 11% of non-mules).
   - It's now blocked, and everything was retrained.
   - Also fixed in the audit itself: a sampling-seed collision with the train/test split, and derived features that weren't being recomputed.
6. Pattern library, versioned snapshot, API, live pipeline page, adapter registry (NFPC, AML and BAF are all live).
7. Role-based demo:
   - 5 personas with server-enforced scope and a live 403 probe.
   - The local-to-federated deployment story, built on real held-out mules.
   - RBI supervisory dashboard.
8. Submission CSV, Redpanda replayer, Kaggle notebooks, README, one-pager, Docker Compose.

## 4. Final numbers (leak-free)
- **Main comparison**: local-only 0.756, FedShield **0.770**, centralised 0.773 AUPRC (each bank's own test accounts).
- **Final model**: AUC 0.945, P 0.83, R 0.81, F1 0.82.
- **Federation effect**: +37 / −6 held-out mules versus each bank's own model.
- **Attacks**: gradient-scaling attacker 0.028 → 0.774 with the gates on. Label flipping 0.697 → 0.774.
- **DP**: costly (≈ 0.2 AUPRC at ε 16–64 on the sweep model). Reported as a limitation.

## 5. Open items / ideas
- Split very large RBI clusters (the biggest has 586 accounts) or tighten the linking rule.
- Real TreeSHAP for the account view, a DPIP evidence feed (currently absent rather than faked).
- Host the API (Render/Railway/tunnel) for the Vercel UI. See `docs/DEPLOYMENT.md`.
