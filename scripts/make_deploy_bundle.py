"""Build the slim artefact bundle the API needs in production (Render etc.).

The full work/ folder is ~16 GB (raw transaction shards, feature intermediates).
The API reads only a few files; this script copies those and trims the two big
tables to what the API actually queries:

  monthly_full  -> only accounts the consoles can open (held-out test accounts)
  edges_full    -> only edges of labelled accounts on counterparties touched by test accounts
  RBI clusters  -> precomputed once (work/rbi_clusters.json), so no edge scan at runtime

    python scripts/make_deploy_bundle.py          # -> deploy/work/ and deploy/fedshield_work.zip
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedshield.config import WORK  # noqa: E402

OUT = ROOT / "deploy" / "work"
COPY = ["models/nfpc_full.pkl", "models/nfpc_full_scores.parquet", "models/nfpc_full_features.parquet",
        "models/nfpc_full_local.parquet", "models/nfpc_full_local_models.pkl",
        "models/stream.pkl", "models/stream_events.parquet", "patterns.json", "demo_cases.json"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    (OUT / "nfpc").mkdir()
    for f in COPY:
        shutil.copy2(WORK / f, OUT / f)
    sc = pd.read_parquet(WORK / "models" / "nfpc_full_scores.parquet", columns=["acct", "split"])
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4")
    con.register("lab", sc[["acct"]])
    con.register("test", sc.loc[sc["split"] == "test", ["acct"]])
    w = WORK / "nfpc"
    con.execute(f"""COPY (SELECT m.* FROM '{(w / 'monthly_full.parquet').as_posix()}' m JOIN test USING (acct))
                    TO '{(OUT / 'nfpc' / 'monthly_full.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    ep = (w / "edges_full.parquet").as_posix()
    con.execute(f"""COPY (WITH cps AS (SELECT DISTINCT cp FROM '{ep}' JOIN test USING (acct))
                          SELECT e.* FROM '{ep}' e JOIN lab USING (acct) JOIN cps USING (cp))
                    TO '{(OUT / 'nfpc' / 'edges_full.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    # RBI clusters from the FULL edge table, then cached
    import pickle

    from fedshield.api.supervisor import clusters
    art = pickle.load(open(WORK / "models" / "nfpc_full.pkl", "rb"))
    full_sc = pd.read_parquet(WORK / "models" / "nfpc_full_scores.parquet")
    cl = clusters(full_sc, art["bank_names"], art["threshold"])
    (OUT / "rbi_clusters.json").write_text(json.dumps(cl))
    shutil.copytree(ROOT / "results", OUT.parent / "results", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("submission"))
    z = OUT.parent / "fedshield_work.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
        for p in (OUT.parent / "results").rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1e6:.1f} MB")
    print(f"bundle: {z}  {z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
