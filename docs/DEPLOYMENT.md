# Deployment

FedShield has two parts with very different hosting needs:

| Part | What it needs | Where it can run |
|---|---|---|
| **UI** (`ui/`, React static build) | Static hosting | **Vercel**, Netlify, any CDN, or served by the API itself |
| **API** (`fedshield/api`, FastAPI) | Python 3.11, ~1.5 GB RAM, the model artefacts (~310 MB, see below), long-lived WebSockets (live pipeline, stream) | A container host: Render, Railway, Fly.io, a VM, or your laptop with a tunnel |

> **Vercel cannot run the API.** Its serverless functions have size and time limits and
> don't hold WebSockets. That's why the UI deploys to Vercel and calls the API wherever
> it runs, through `VITE_API_BASE`.

## 0. Interactive demo on Vercel (no backend)

The UI can run entirely on its own. In demo mode every REST call and WebSocket is answered from
fixtures recorded from the real API: alerts, explanations, money-flow graphs, all 72 live
federation runs (3 attack settings × 4 DP settings × Byzantine on/off × 20/40/80 trees) and the
payment stream. Officer actions and the "deployed" state are kept in the visitor's browser.

1. Import the repo in Vercel and set **Root Directory = `ui`**. `ui/vercel.json` already runs
   `npm run build:demo`, serves `dist`, and keeps `/demo/*` fixtures out of the SPA rewrite.
2. Deploy. No environment variables are needed.

Locally: `cd ui && npm run dev:demo` (or `npm run build:demo && npx vite preview --mode demo`).

The fixtures live in `ui/demo-public/demo` (about 30 MB). They are only bundled by the demo build,
so the normal build (`npm run build`, used by the API image) is unchanged. To refresh them after
retraining:

```bash
FEDSHIELD_SECRET=<any fixed string> python scripts/record_demo_fixtures.py   # needs the artefact bundle
```

They contain per-account explanations and timelines derived from the NFPC data, so check the
dataset licence before making the deployment public.

To point Vercel at a live API instead, set the build command back to `npm run build` and set
`VITE_API_BASE` (section 1).

## 1. UI on Vercel

1. Import the GitHub repo in Vercel and set **Root Directory = `ui`**. `ui/vercel.json` already
   sets the Vite build, the `dist` output and the single-page-app rewrites.
2. Under **Settings → Environment Variables**, set
   `VITE_API_BASE = https://<your-api-host>`. No trailing slash; it's used for both `https://` and `wss://`.
3. Deploy. Every push to `main` redeploys.

The API allows any origin (CORS `*`) and uses bearer tokens, not cookies, so the cross-origin setup needs nothing else.

## 2. API (container)

The API reads its artefacts from `FEDSHIELD_WORK` (default `work/`):

```
work/models/nfpc_full.pkl, nfpc_full_scores.parquet, nfpc_full_features.parquet,
            nfpc_full_local.parquet, nfpc_full_local_models.pkl, stream.pkl, stream_events.parquet
work/nfpc/monthly_full.parquet   (~90 MB)    work/nfpc/edges_full.parquet (~163 MB)
work/patterns.json   work/demo_cases.json
results/*.json
```

These are **not in git**. They're derived from the licensed NFPC competition data, and some are over GitHub's 100 MB limit.
Produce them with the pipeline in the README (or the Kaggle notebook), then make them available to the container:

- **Render / Railway / Fly**: build `docker/Dockerfile.api`, attach a persistent disk at `/data/work`,
  and upload the artefacts to it once. Set `FEDSHIELD_SECRET` to a long random string and
  `FEDSHIELD_DEMO_PASSWORD` to something other than the default.
- **Laptop + tunnel** (quickest for a live demo): `uvicorn fedshield.api.server:app --port 8000`,
  then `cloudflared tunnel --url http://localhost:8000` (or ngrok). Put the tunnel URL in `VITE_API_BASE`.

### Render, step by step (what goes where)

You do **not** need the 15.9 GB `work/` folder. Nearly all of it is raw transaction shards and feature-extraction intermediates.
The API needs a 143 MB bundle:

| File (inside the bundle) | Size | Used for |
|---|---|---|
| `work/models/nfpc_full.pkl` | 1 MB | federated model and threshold |
| `work/models/nfpc_full_features.parquet` | 56 MB | features for explanations and the live pipeline |
| `work/models/nfpc_full_scores.parquet` | 2 MB | scores, banks, splits |
| `work/models/nfpc_full_local*.parquet/.pkl` | 4 MB | each bank's local-only model (before/after demo) |
| `work/models/stream.pkl`, `stream_events.parquet` | 1 MB | live stream |
| `work/nfpc/monthly_full.parquet` | 9 MB | account timelines (trimmed to test accounts) |
| `work/nfpc/edges_full.parquet` | 79 MB | money-flow graph (trimmed) |
| `work/rbi_clusters.json`, `patterns.json`, `demo_cases.json` | <1 MB | RBI view, pattern library, watch list |

1. `python scripts/make_deploy_bundle.py` writes `deploy/fedshield_work.zip`.
2. Upload the zip somewhere that serves a **direct download** link:
   - **Dropbox:** share the file and change `?dl=0` to `?dl=1`.
   - **GitHub Release asset:** only if the data can be public.
   - **S3 or R2 presigned URL.**
   - Avoid Google Drive: files over 100 MB get a confirmation page instead of the file.
3. In Render, choose **New → Blueprint** and pick this repo (`render.yaml`), or create a Docker web service with `docker/Dockerfile.api`.
4. Set `FEDSHIELD_BUNDLE_URL` to the link. On first request after each deploy, the API downloads and unpacks it (about 20 s).
   Render's disk is ephemeral, so this repeats after each redeploy or restart.
5. Choose a plan with at least 1 GB RAM. The free 512 MB plan runs out of memory during a live federation run.
6. Point Vercel's `VITE_API_BASE` at the Render URL.

Note: RBI cluster pseudonyms are computed when the bundle is built. To keep them identical to the runtime pseudonyms,
build the bundle with the same `FEDSHIELD_SECRET` you set on Render.

## 3. Everything together (Docker Compose)

```bash
python scripts/export_partitions.py
docker compose up --build        # UI + API on http://localhost:8000
```

Services: `bank-a/b/c` (each mounts only its own partition), `aggregator` (federates over HTTP with masked
histograms, then exits), `redis`, `redpanda`, `replayer` (publishes payments), `api` (UI + API).

**TLS-inspecting networks** (corporate proxies, Avast/Kaspersky web shields): if `pip install` or
`npm ci` fails with `CERTIFICATE_VERIFY_FAILED` inside the build, export the inspecting root CA as
PEM into `docker/certs/<name>.crt`. Both images trust any `*.crt` there. The folder is gitignored
because the certificate is specific to your machine.

## Security notes for a public deployment

- Demo personas and passwords are hardcoded on purpose, for the jury demo. Change `FEDSHIELD_DEMO_PASSWORD`,
  or put the API behind your identity provider before exposing real data.
- `FEDSHIELD_SECRET` signs session tokens and keys the pseudonyms on the RBI view. Set it; otherwise it's random per process.
- The "ground truth" panel in the officer console shows evaluation labels. Remove it for anything beyond a demo.
