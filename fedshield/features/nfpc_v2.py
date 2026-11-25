"""NFPC feature extraction v2 — memory-bounded, resumable.

Adds the MuleHunter.ai feature families (RBIH "Necessary Fields and Features",
MH251106FS) on top of the v1 behavioural features, translated from the
document's per-transaction "past-N window from current transaction" form to the
account level as PEAK and BASELINE statistics:

  count/sum txn (all, credit, debit) past 1h/12h/1d/7d   -> peak over the account's history
  per-channel count/sum (UPI, IMPS, NEFT, ATM) past 1d/7d -> peak
  avg credit/debit past 0-7d vs 7-90d                    -> burst ratio (comparative profiling)
  concurrent credits/debits within 3 / 15 min             -> peak + mean concurrency
  night flag (23:00-07:00) past 1d                        -> share + peak
  num digit / sum digit of amount                         -> mean + std
  unique transacting party per week / month               -> peak
  MCC-amount anomaly                                       -> share of |z| > 3 vs the MCC's distribution

Memory: each shard is processed in sub-batches of BATCH accounts; DuckDB is capped
at MEM and spills to TEMP. Every sub-batch writes its own parquet files and is
checkpointed, so an interrupted run resumes where it stopped.

    python -m fedshield.features.nfpc_v2            # all shards
    python -m fedshield.features.nfpc_v2 --limit 1  # one sub-batch (smoke test)
    python -m fedshield.features.nfpc_v2 --merge    # merge parts into txn_feats_full.parquet
"""
import argparse
import glob
import os
import time

import duckdb

from fedshield.config import NFPC_DIR, NFPC_SHARDS, SHARD_SIZE, WORK
from fedshield.features.nfpc import FEATURE_SQL

OUT = WORK / "nfpc" / "v2"
TEMP = WORK / "duckdb_tmp"
BATCH = 2_500
MEM = os.environ.get("FEDSHIELD_DUCKDB_MEM", "3GB")
THREADS = int(os.environ.get("FEDSHIELD_DUCKDB_THREADS", 4))

# Per-transaction rolling windows (the document's "past N from current transaction"),
# reduced to per-account peaks. One sort per account partition serves every frame.
WINDOW_SQL = """
WITH t AS (
  SELECT acct, ts, abs(amount) AS amt, is_credit,
         channel IN ('UPC','UPD') AS is_upi, channel = 'IPM' AS is_imps,
         channel IN ('NTD') AS is_neft, (channel = 'ATW' OR atm IS NOT NULL) AS is_atm,
         (hour(ts) >= 23 OR hour(ts) < 7) AS is_night,
         -- MuleHunter "num digit" / "sum digit" of the integer part
         CASE WHEN abs(amount) >= 1 THEN floor(log10(abs(amount))) + 1 ELSE 1 END AS num_digit,
         ((floor(abs(amount) / 1) % 10) + (floor(abs(amount) / 10) % 10) + (floor(abs(amount) / 100) % 10) + (floor(abs(amount) / 1000) % 10) + (floor(abs(amount) / 10000) % 10) + (floor(abs(amount) / 100000) % 10) + (floor(abs(amount) / 1000000) % 10) + (floor(abs(amount) / 10000000) % 10) + (floor(abs(amount) / 100000000) % 10) + (floor(abs(amount) / 1000000000) % 10)) AS sum_digit,
         mcc
  FROM shard WHERE acct BETWEEN {lo} AND {hi}
),
w AS (
  SELECT acct, ts, amt, is_credit, is_night, num_digit, sum_digit, mcc,
    count(*)                                         OVER h1  AS c_1h,
    count(*)                                         OVER h12 AS c_12h,
    count(*)                                         OVER d1  AS c_1d,
    count(*)                                         OVER d7  AS c_7d,
    sum(CASE WHEN is_credit THEN amt ELSE 0 END)     OVER h1  AS cr_1h,
    sum(CASE WHEN is_credit THEN amt ELSE 0 END)     OVER d1  AS cr_1d,
    sum(CASE WHEN is_credit THEN amt ELSE 0 END)     OVER d7  AS cr_7d,
    sum(CASE WHEN is_credit THEN amt ELSE 0 END)     OVER d90 AS cr_90d,
    sum(CASE WHEN NOT is_credit THEN amt ELSE 0 END) OVER d1  AS dr_1d,
    sum(CASE WHEN NOT is_credit THEN amt ELSE 0 END) OVER d7  AS dr_7d,
    sum(CASE WHEN NOT is_credit THEN amt ELSE 0 END) OVER d90 AS dr_90d,
    sum(CASE WHEN is_credit THEN 1 ELSE 0 END)       OVER d7  AS ncr_7d,
    sum(CASE WHEN is_credit THEN 1 ELSE 0 END)       OVER d90 AS ncr_90d,
    sum(is_upi::INT)   OVER d1 AS upi_1d,  sum(is_upi::INT)   OVER d7 AS upi_7d,
    sum(is_imps::INT)  OVER d1 AS imps_1d, sum(is_imps::INT)  OVER d7 AS imps_7d,
    sum(is_neft::INT)  OVER d1 AS neft_1d, sum(is_neft::INT)  OVER d7 AS neft_7d,
    sum(is_atm::INT)   OVER d1 AS atm_1d,  sum(is_atm::INT)   OVER d7 AS atm_7d,
    sum(CASE WHEN is_upi THEN amt ELSE 0 END)  OVER d7 AS upi_amt_7d,
    sum(CASE WHEN is_atm THEN amt ELSE 0 END)  OVER d7 AS atm_amt_7d,
    sum(is_night::INT) OVER d1 AS night_1d
  FROM t
  WINDOW
    h1  AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL 1 HOUR  PRECEDING AND CURRENT ROW),
    h12 AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL 12 HOUR PRECEDING AND CURRENT ROW),
    d1  AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL 1 DAY   PRECEDING AND CURRENT ROW),
    d7  AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL 7 DAY   PRECEDING AND CURRENT ROW),
    d90 AS (PARTITION BY acct ORDER BY ts RANGE BETWEEN INTERVAL 90 DAY  PRECEDING AND CURRENT ROW)
),
conc AS (   -- concurrent same-direction transactions within 3 / 15 minutes
  SELECT acct, is_credit,
    count(*) OVER (PARTITION BY acct, is_credit ORDER BY ts RANGE BETWEEN INTERVAL 3 MINUTE PRECEDING AND CURRENT ROW) - 1 AS k3,
    count(*) OVER (PARTITION BY acct, is_credit ORDER BY ts RANGE BETWEEN INTERVAL 15 MINUTE PRECEDING AND CURRENT ROW) - 1 AS k15
  FROM t
),
conc_f AS (
  SELECT acct,
    max(k3)  FILTER (WHERE is_credit)     AS mh_conc_credit_3m_max,
    max(k15) FILTER (WHERE is_credit)     AS mh_conc_credit_15m_max,
    avg(k15) FILTER (WHERE is_credit)     AS mh_conc_credit_15m_mean,
    max(k3)  FILTER (WHERE NOT is_credit) AS mh_conc_debit_3m_max,
    max(k15) FILTER (WHERE NOT is_credit) AS mh_conc_debit_15m_max,
    avg(k15) FILTER (WHERE NOT is_credit) AS mh_conc_debit_15m_mean
  FROM conc GROUP BY acct
),
mccz AS (
  SELECT t.acct,
    avg((abs(ln(greatest(t.amt, 1)) - s.mu) > 3 * s.sd)::INT) AS mh_mcc_amount_outlier_share,
    max((ln(greatest(t.amt, 1)) - s.mu) / s.sd)               AS mh_mcc_amount_max_z
  FROM t JOIN mcc_stats s ON s.mcc = t.mcc AND s.sd > 0 GROUP BY t.acct
),
weekly AS (
  SELECT acct, max(n) AS mh_max_unique_party_week FROM (
    SELECT acct, date_trunc('week', ts) wk, count(DISTINCT cp) n FROM shard
    WHERE acct BETWEEN {lo} AND {hi} GROUP BY 1, 2) GROUP BY acct
),
monthly_u AS (
  SELECT acct, max(n) AS mh_max_unique_party_month, avg(n) AS mh_mean_unique_party_month FROM (
    SELECT acct, date_trunc('month', ts) m, count(DISTINCT cp) n FROM shard
    WHERE acct BETWEEN {lo} AND {hi} GROUP BY 1, 2) GROUP BY acct
),
peaks AS (
  SELECT acct,
    max(c_1h) AS mh_max_txn_1h, max(c_12h) AS mh_max_txn_12h,
    max(c_1d) AS mh_max_txn_1d, max(c_7d) AS mh_max_txn_7d,
    max(cr_1h) AS mh_max_credit_amt_1h, max(cr_1d) AS mh_max_credit_amt_1d,
    max(cr_7d) AS mh_max_credit_amt_7d, max(dr_1d) AS mh_max_debit_amt_1d, max(dr_7d) AS mh_max_debit_amt_7d,
    -- comparative profiling: last-7-day credit average vs the preceding 7-90 day average, at the peak
    max((cr_7d / greatest(ncr_7d, 1)) /
        greatest((cr_90d - cr_7d) / greatest(ncr_90d - ncr_7d, 1), 1))       AS mh_max_credit_avg_ratio_7d_vs_90d,
    max(cr_7d / greatest((cr_90d - cr_7d) / 12.0, 1))                         AS mh_max_credit_vol_burst_7d_vs_90d,
    max(dr_7d / greatest((dr_90d - dr_7d) / 12.0, 1))                         AS mh_max_debit_vol_burst_7d_vs_90d,
    max(least(cr_1d, dr_1d) / greatest(cr_1d, 1))                             AS mh_max_passthrough_1d,
    max(upi_1d) AS mh_max_upi_1d, max(upi_7d) AS mh_max_upi_7d,
    max(imps_1d) AS mh_max_imps_1d, max(imps_7d) AS mh_max_imps_7d,
    max(neft_1d) AS mh_max_neft_1d, max(neft_7d) AS mh_max_neft_7d,
    max(atm_1d) AS mh_max_atm_1d, max(atm_7d) AS mh_max_atm_7d,
    max(upi_amt_7d) AS mh_max_upi_amt_7d, max(atm_amt_7d) AS mh_max_atm_amt_7d,
    max(night_1d) AS mh_max_night_1d, avg(is_night::INT) AS mh_night_share_23_07,
    avg(num_digit) AS mh_num_digit_mean, stddev_samp(num_digit) AS mh_num_digit_std,
    avg(sum_digit) AS mh_sum_digit_mean, stddev_samp(sum_digit) AS mh_sum_digit_std,
    avg((num_digit = 5 AND sum_digit >= 30)::INT) AS mh_high_digitsum_5digit_share   -- e.g. 49,9xx
  FROM w GROUP BY acct
)
SELECT p.*, c.* EXCLUDE (acct), z.* EXCLUDE (acct), wk.* EXCLUDE (acct), mu.* EXCLUDE (acct)
FROM peaks p
LEFT JOIN conc_f c USING (acct) LEFT JOIN mccz z USING (acct)
LEFT JOIN weekly wk USING (acct) LEFT JOIN monthly_u mu USING (acct)
"""


def connect():
    TEMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM}'; SET threads={THREADS}; SET preserve_insertion_order=false; "
                f"SET temp_directory='{TEMP.as_posix()}'; SET enable_progress_bar=false")
    return con


def mcc_stats(con):
    path = OUT / "mcc_stats.parquet"
    if not path.exists():
        print("computing MCC amount distribution (one streaming pass) ...", flush=True)
        con.execute(f"""COPY (SELECT mcc, avg(ln(greatest(abs(amount),1))) mu, stddev_samp(ln(greatest(abs(amount),1))) sd,
                         count(*) n FROM read_parquet('{NFPC_SHARDS.as_posix()}/*/*.parquet') GROUP BY mcc)
                        TO '{path.as_posix()}' (FORMAT parquet)""")
    con.execute(f"CREATE OR REPLACE TABLE mcc_stats AS SELECT * FROM read_parquet('{path.as_posix()}')")


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("feats", "monthly", "daily", "edges", "ips"):
        (OUT / d).mkdir(exist_ok=True)
    con = connect()
    mcc_stats(con)
    import pandas as pd
    acc = pd.read_parquet(NFPC_DIR / "accounts.parquet", columns=["account_id", "last_mobile_update_date"])
    acc["acct"] = acc["account_id"].str[5:].astype(int)
    acc["mdate"] = pd.to_datetime(acc["last_mobile_update_date"], errors="coerce")
    con.register("mobile_upd", acc.loc[acc["mdate"].notna(), ["acct", "mdate"]])
    shards = sorted(glob.glob(str(NFPC_SHARDS / "shard=*")), key=lambda p: int(p.split("=")[-1]))
    done_file = OUT / "_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    t0, n_run = time.time(), 0
    for s in shards:
        k = int(s.split("=")[-1])
        for lo in range(k * SHARD_SIZE, (k + 1) * SHARD_SIZE, BATCH):
            hi = lo + BATCH - 1
            tag = f"{lo:06d}"
            if tag in done:
                continue
            con.execute(f"""CREATE OR REPLACE TEMP TABLE shard AS
                            SELECT * FROM read_parquet('{s}/*.parquet') WHERE acct BETWEEN {lo} AND {hi}""")
            if con.execute("SELECT count(*) FROM shard").fetchone()[0] == 0:
                done.add(tag)
                continue
            v1 = FEATURE_SQL.format(cutoff_join="").replace("FROM shard x", "FROM shard x")
            con.execute(f"""COPY (SELECT a.*, b.* EXCLUDE (acct) FROM ({v1}) a
                                  LEFT JOIN ({WINDOW_SQL.format(lo=lo, hi=hi)}) b USING (acct))
                            TO '{(OUT / 'feats' / f'{tag}.parquet').as_posix()}' (FORMAT parquet)""")
            con.execute(f"""COPY (SELECT acct, date_trunc('month', ts) AS month, count(*) n,
                                   coalesce(sum(abs(amount)) FILTER (WHERE is_credit),0) credit,
                                   coalesce(sum(abs(amount)) FILTER (WHERE NOT is_credit),0) debit
                                  FROM shard GROUP BY 1, 2)
                            TO '{(OUT / 'monthly' / f'{tag}.parquet').as_posix()}' (FORMAT parquet)""")
            con.execute(f"""COPY (SELECT acct, CAST(ts AS DATE) AS d, count(*)::INT n,
                                   coalesce(sum(abs(amount)) FILTER (WHERE is_credit),0)::FLOAT credit,
                                   coalesce(sum(abs(amount)) FILTER (WHERE NOT is_credit),0)::FLOAT debit,
                                   min(ts) first_ts, max(ts) last_ts
                                  FROM shard GROUP BY 1, 2)
                            TO '{(OUT / 'daily' / f'{tag}.parquet').as_posix()}' (FORMAT parquet)""")
            con.execute(f"""COPY (SELECT acct, cp, is_credit, count(*)::INT n, sum(abs(amount))::FLOAT v,
                                   min(ts) first_ts, max(ts) last_ts FROM shard GROUP BY 1, 2, 3)
                            TO '{(OUT / 'edges' / f'{tag}.parquet').as_posix()}' (FORMAT parquet)""")
            con.execute(f"""COPY (SELECT DISTINCT acct, ip FROM shard WHERE ip IS NOT NULL)
                            TO '{(OUT / 'ips' / f'{tag}.parquet').as_posix()}' (FORMAT parquet)""")
            con.execute("DROP TABLE shard")
            with open(done_file, "a") as fh:
                fh.write(tag + "\n")
            done.add(tag)
            n_run += 1
            el = time.time() - t0
            print(f"  accounts {lo:>6}-{hi:<6} done  ({n_run} batches, {el / 60:.1f} min, "
                  f"{el / n_run:.0f}s/batch)", flush=True)
            if limit and n_run >= limit:
                return
    print(f"all batches done in {(time.time() - t0) / 60:.1f} min")


def merge():
    """Combine the part files into the single-file layout the rest of the pipeline reads."""
    con = connect()
    base = WORK / "nfpc"
    for name, out in (("feats", "txn_feats_full"), ("monthly", "monthly_full"), ("daily", "daily_full"),
                      ("edges", "edges_full"), ("ips", "ips_full")):
        con.execute(f"COPY (SELECT * FROM read_parquet('{(OUT / name).as_posix()}/*.parquet')) "
                    f"TO '{(base / f'{out}.parquet').as_posix()}' (FORMAT parquet)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{(base / f'{out}.parquet').as_posix()}')").fetchone()[0]
        print(f"  {out}: {n:,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else main(a.limit)
