"""Generate the Kaggle reproducibility notebooks in notebooks/.

Each notebook is a thin driver: the logic lives in the fedshield package (attach the
repo as a Kaggle dataset, or pip-install it from GitHub), so the notebooks and the
local prototype run exactly the same code.

    python scripts/make_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''# --- setup: put the FedShield code on the path and point it at the Kaggle inputs ---
import os, sys, glob, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb>=1.1", "numba>=0.60", "xgboost>=2.0"], check=True)
# the repo attached as a Kaggle dataset (any slug): find the folder that contains fedshield/
code = next(p for p in glob.glob("/kaggle/input/*/") + glob.glob("/kaggle/input/*/*/") if os.path.isdir(os.path.join(p, "fedshield")))
sys.path.insert(0, code)
os.environ["FEDSHIELD_WORK"] = "/kaggle/working/work"
os.environ["FEDSHIELD_RESULTS"] = "/kaggle/working/results"
os.environ["FEDSHIELD_DUCKDB_MEM"] = "12GB"        # Kaggle has ~30 GB RAM
print("code:", code)
'''


def nb(title, intro, cells):
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    out = [md(f"# {title}\n\n{intro}"), code(SETUP)]
    for kind, src in cells:
        out.append(md(src) if kind == "md" else code(src))
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


NFPC = nb("FedShield · NFPC Phase-2 (main result)",
          "Federated mule-account detection on the RBIH NFPC data: ingest → MuleHunter feature families → "
          "3-bank federation (secure aggregation, Byzantine gates, DP) → leakage check → submission.\n\n"
          "Attach: the NFPC dataset and the FedShield code dataset. Runtime ≈ 2–3 h on CPU.",
          [("code", 'os.environ["FEDSHIELD_NFPC_DIR"] = next(p for p in glob.glob("/kaggle/input/*/") if os.path.exists(os.path.join(p, "train_labels.parquet")))\nprint(os.environ["FEDSHIELD_NFPC_DIR"])'),
           ("md", "## 1 · Ingest: join transactions with transactions_additional, shard by account"),
           ("code", "!cd {code} && python -m fedshield.ingest.nfpc"),
           ("md", "## 2 · Features: v1 behavioural + MuleHunter families (rolling peaks, channel counts, digit patterns, concurrency, MCC outliers). Batched and resumable"),
           ("code", "!cd {code} && python -m fedshield.features.nfpc_v2 && python -m fedshield.features.nfpc_v2 --merge"),
           ("md", "## 3 · Experiments: local vs FedShield vs centralised, registry, pattern skew, Byzantine, DP, fairness, seeds"),
           ("code", "!cd {code} && python -m fedshield.experiments.nfpc full"),
           ("md", "## 4 · Leakage check: point-in-time features vs full-window features"),
           ("code", "!cd {code} && python -m fedshield.experiments.leakage"),
           ("md", "## 5 · Submission for test_accounts"),
           ("code", "!cd {code} && python -m fedshield.experiments.submission\nimport pandas as pd\npd.read_csv('/kaggle/working/results/submission/submission.csv').head()"),
           ("code", "import json\nr = json.load(open('/kaggle/working/results/nfpc_full.json'))\nprint(json.dumps(r['main']['summary'], indent=1))\nprint(json.dumps(json.load(open('/kaggle/working/results/nfpc_leakage.json')), indent=1)[:2000])")])

AML = nb("FedShield · IBM AMLworld HI-Small (cross-bank laundering)",
         "Five institutions from the real From/To bank columns. Shows what no single bank can see: rings spanning institutions, "
         "and the hashed suspect registry.\n\nAttach: the IBM AML dataset and the FedShield code dataset.",
         [("code", 'os.environ["FEDSHIELD_AML_DIR"] = next(p for p in glob.glob("/kaggle/input/*/") if glob.glob(os.path.join(p, "HI-Small_Trans.csv")))'),
          ("code", "!cd {code} && python -m fedshield.ingest.aml HI-Small && python -m fedshield.experiments.aml HI-Small")])

BAF = nb("FedShield · BAF (onboarding fraud, plug-in adapter demo)",
         "The same federation core on a dataset with no transactions at all: account-opening applications, three banks "
         "by payment type. Demonstrates the adapter contract (`fedshield/adapters.py`).\n\nAttach: the BAF dataset and the FedShield code dataset.",
         [("code", 'os.environ["FEDSHIELD_BAF_DIR"] = next(p for p in glob.glob("/kaggle/input/*/") if os.path.exists(os.path.join(p, "Base.csv")))'),
          ("code", "!cd {code} && python -m fedshield.ingest.baf Base"),
          ("code", "from fedshield.api.pipeline import LiveRun\nrun = LiveRun(); run.start(dataset='baf', sample=200000, n_trees=60, pace=0, attack='gradient_scale'); run.thread.join()\n"
                   "trees = [e for e in run.events if e['stage'] == 'tree' and 'eval_auprc' in e]\n"
                   "setup = next(e for e in run.events if e['stage'] == 'setup')\n"
                   "print('gate 2:', {b: g['rejected'] for b, g in setup['gate2'].items()})\nprint('final AUPRC', trees[-1]['eval_auprc'])")])

for name, book in (("01_nfpc_fedshield.ipynb", NFPC), ("02_amlworld_fedshield.ipynb", AML), ("03_baf_adapter.ipynb", BAF)):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(book, indent=1), encoding="utf-8")
    print("wrote", OUT / name)
