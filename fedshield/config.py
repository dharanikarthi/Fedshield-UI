"""Central paths and constants. Override the data root with FEDSHIELD_DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FEDSHIELD_DATA", ROOT / "Datasets"))
WORK = Path(os.environ.get("FEDSHIELD_WORK", ROOT / "work"))
RESULTS = Path(os.environ.get("FEDSHIELD_RESULTS", ROOT / "results"))

# per-dataset overrides (Kaggle mounts each dataset under /kaggle/input/<slug>)
NFPC_DIR = Path(os.environ.get("FEDSHIELD_NFPC_DIR", DATA / "archive (1)"))
AML_DIR = Path(os.environ.get("FEDSHIELD_AML_DIR", DATA / "archive (2)"))
BAF_DIR = Path(os.environ.get("FEDSHIELD_BAF_DIR", DATA / "archive (3)"))
PAYSIM_FILE = Path(os.environ.get("FEDSHIELD_PAYSIM_FILE", DATA / "archive (4)" / "PS_20174392719_1491204439457_log.csv"))

NFPC_SHARDS = WORK / "nfpc" / "shards"     # per-account-range transaction shards
NFPC_FEATS = WORK / "nfpc" / "features.parquet"
AML_WORK = WORK / "aml"
BAF_WORK = WORK / "baf"

# Columns that reveal the label or are produced after flagging. Never features.
NFPC_LEAKAGE = {
    "account_status", "freeze_date", "unfreeze_date",
    "mule_flag_date", "alert_reason", "flagged_by_branch",
}
NFPC_TXN_START = "2020-07-01"
NFPC_TXN_END = "2025-06-30"
SHARD_SIZE = 10_000          # accounts per shard (ACCT number // SHARD_SIZE)
SEED = 42

for p in (WORK, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
