"""NFPC per-account behavioural features, computed shard by shard with DuckDB.

Two modes:
  full   — every transaction in the 5-year window (comparable with the
           competition / benchmark notebook).
  asof   — point-in-time: for each account, only transactions strictly before
           its cutoff. Mules: cutoff = mule_flag_date (clipped to the window).
           Others: a cutoff drawn from the mule flag-date distribution, so the
           two classes are observed over comparable horizons. This is the
           honest deployment setting (no post-flag behaviour, e.g. freezes).

Outputs (work/nfpc/):
  txn_feats_<mode>.parquet   one row per account
  monthly_<mode>.parquet     acct x month volumes (timeline + suspicious window)
  edges_<mode>.parquet       distinct (acct, cp, direction) — graph / registry
  ips_<mode>.parquet         distinct (acct, ip) — shared-device signal

    python -m fedshield.features.nfpc full|asof
"""
import glob
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from fedshield.config import NFPC_DIR, NFPC_SHARDS, NFPC_TXN_END, NFPC_TXN_START, SEED, WORK

OUT = WORK / "nfpc"

FEATURE_SQL = """
WITH t AS (
  SELECT x.*, abs(x.amount) AS amt,
         CAST(x.ts AS DATE) AS d,
         date_trunc('month', x.ts) AS m
  FROM shard x {cutoff_join}
),
base AS (
  SELECT acct,
    count(*)                                            AS n_txn,
    count(*) FILTER (WHERE is_credit)                   AS n_credit,
    count(*) FILTER (WHERE NOT is_credit)               AS n_debit,
    coalesce(sum(amt) FILTER (WHERE is_credit), 0)      AS sum_credit,
    coalesce(sum(amt) FILTER (WHERE NOT is_credit), 0)  AS sum_debit,
    avg(amt) AS mean_amt, median(amt) AS median_amt, stddev_samp(amt) AS std_amt, max(amt) AS max_amt,
    count(*) FILTER (WHERE amount < 0)                  AS n_reversal,
    count(DISTINCT cp)                                  AS n_cp,
    count(DISTINCT cp) FILTER (WHERE is_credit)         AS n_cp_in,
    count(DISTINCT cp) FILTER (WHERE NOT is_credit)     AS n_cp_out,
    min(ts) AS first_ts, max(ts) AS last_ts,
    count(DISTINCT d)                                   AS active_days,
    count(DISTINCT m)                                   AS active_months,
    -- amount patterns
    avg((amt >= 1000 AND amt % 1000 = 0)::INT)          AS round_1k_share,
    avg((amt >= 5000 AND amt % 5000 = 0)::INT)          AS round_5k_share,
    count(*) FILTER (WHERE amt >= 45000 AND amt < 50000) AS n_struct_50k,
    count(*) FILTER (WHERE amt >= 9000 AND amt < 10000)  AS n_struct_10k,
    avg((amt >= 45000 AND amt < 50000)::INT)            AS struct_50k_share,
    -- channels
    avg((channel = 'UPC')::INT) AS ch_upi_credit, avg((channel = 'UPD')::INT) AS ch_upi_debit,
    avg((channel = 'IPM')::INT) AS ch_imps,       avg((channel = 'NTD')::INT) AS ch_neft,
    avg((channel = 'ATW')::INT) AS ch_atm_wd,     avg((channel IN ('CSD','OCD'))::INT) AS ch_cash_dep,
    avg((channel = 'P2A')::INT) AS ch_p2a,        avg((channel = 'END')::INT) AS ch_pos,
    avg((channel = 'CHQ')::INT) AS ch_cheque,     avg((channel IN ('FTD','FTC'))::INT) AS ch_fund_transfer,
    count(DISTINCT channel) AS n_channels, count(DISTINCT mcc) AS n_mcc,
    -- time of day / calendar
    avg((hour(ts) >= 23 OR hour(ts) < 5)::INT)          AS night_share,
    avg((dayofweek(ts) IN (0, 6))::INT)                 AS weekend_share,
    avg((day(ts) >= 25 OR day(ts) <= 5)::INT)           AS month_edge_share,
    -- balance / additional fields
    min(bal) AS bal_min, avg(bal) AS bal_mean, max(bal) AS bal_max, stddev_samp(bal) AS bal_std,
    avg((abs(bal) < 1000)::INT) FILTER (WHERE bal IS NOT NULL) AS near_zero_bal_share,
    count(DISTINCT ip)                                  AS n_ip,
    avg((ip IS NOT NULL)::INT)                          AS ip_share,
    count(DISTINCT (round(lat, 1), round(lon, 1)))      AS n_locations,
    stddev_samp(lat) AS lat_std, stddev_samp(lon) AS lon_std,
    avg((ptt = 'BI')::INT) AS bank_induced_share,
    avg((sub = 'cash')::INT) AS cash_sub_share,
    avg((sub = 'loan')::INT) AS loan_sub_share,
    avg((atm IS NOT NULL)::INT) AS atm_deposit_share
  FROM t GROUP BY acct
),
bidir AS (
  SELECT acct, count(*) AS n_bidir FROM (
    SELECT DISTINCT acct, cp FROM t WHERE is_credit
    INTERSECT SELECT DISTINCT acct, cp FROM t WHERE NOT is_credit) GROUP BY acct
),
top_cp AS (   -- concentration of outflow on the single largest beneficiary
  SELECT acct, max(v) / greatest(sum(v), 1) AS top_beneficiary_share FROM (
    SELECT acct, cp, sum(amt) v FROM t WHERE NOT is_credit GROUP BY 1, 2) GROUP BY acct
),
daily AS (
  SELECT acct, d, count(*) n, coalesce(sum(amt) FILTER (WHERE is_credit),0) c,
         coalesce(sum(amt) FILTER (WHERE NOT is_credit),0) dr
  FROM t GROUP BY 1, 2
),
daily_f AS (
  SELECT acct,
    max(n) AS max_txn_day,
    -- same-day pass-through: share of credits matched by a same-day debit
    sum(least(c, dr)) / greatest(sum(c), 1) AS passthrough_sameday,
    max(gap) AS max_gap_days,
    quantile_cont(n, 0.5) AS median_txn_active_day
  FROM (SELECT *, d - lag(d) OVER (PARTITION BY acct ORDER BY d) AS gap FROM daily) GROUP BY acct
),
nextday AS ( -- credits drained within the next day
  SELECT a.acct, sum(least(a.c, a.dr + coalesce(b.dr, 0))) / greatest(sum(a.c), 1) AS passthrough_48h
  FROM daily a LEFT JOIN daily b ON a.acct = b.acct AND b.d = a.d + 1
  GROUP BY a.acct
),
dormant AS ( -- activity in the 30 days after the longest silence
  SELECT x.acct, count(*) AS n_after_dormancy, sum(x.n) AS txn_after_dormancy
  FROM daily x JOIN (
     SELECT acct, arg_max(d, gap) AS wake_d, max(gap) AS g FROM (
       SELECT acct, d, d - lag(d) OVER (PARTITION BY acct ORDER BY d) AS gap FROM daily)
     GROUP BY acct) w
    ON x.acct = w.acct AND x.d >= w.wake_d AND x.d < w.wake_d + 30 AND w.g >= 90
  GROUP BY x.acct
),
mobile AS ( -- volume in the 60 days after the last mobile-number change vs. a 60-day baseline before
  SELECT t.acct,
    count(*) FILTER (WHERE t.ts >= mu.mdate AND t.ts < mu.mdate + INTERVAL 60 DAY) AS n_post_mobile_60d,
    count(*) FILTER (WHERE t.ts <  mu.mdate AND t.ts >= mu.mdate - INTERVAL 60 DAY) AS n_pre_mobile_60d
  FROM t JOIN mobile_upd mu ON mu.acct = t.acct GROUP BY t.acct
)
SELECT b.*,
  coalesce(bi.n_bidir, 0)                                   AS n_bidir,
  coalesce(bi.n_bidir, 0) / greatest(b.n_cp, 1)             AS bidir_ratio,
  coalesce(tc.top_beneficiary_share, 0)                     AS top_beneficiary_share,
  df.max_txn_day, df.passthrough_sameday, coalesce(df.max_gap_days, 0) AS max_gap_days,
  df.median_txn_active_day,
  coalesce(nd.passthrough_48h, 0)                           AS passthrough_48h,
  coalesce(dm.txn_after_dormancy, 0)                        AS txn_after_dormancy,
  coalesce(mo.n_post_mobile_60d, 0)                         AS n_post_mobile_60d,
  coalesce(mo.n_pre_mobile_60d, 0)                          AS n_pre_mobile_60d
FROM base b
LEFT JOIN bidir bi USING (acct) LEFT JOIN top_cp tc USING (acct)
LEFT JOIN daily_f df USING (acct) LEFT JOIN nextday nd USING (acct)
LEFT JOIN dormant dm USING (acct) LEFT JOIN mobile mo USING (acct)
"""


def asof_cutoffs():
    """Per-account cutoff timestamps for the point-in-time mode."""
    lab = pd.read_parquet(NFPC_DIR / "train_labels.parquet")
    test = pd.read_parquet(NFPC_DIR / "test_accounts.parquet")
    lo, hi = pd.Timestamp(NFPC_TXN_START), pd.Timestamp(NFPC_TXN_END)
    fd = pd.to_datetime(lab["mule_flag_date"], errors="coerce")
    valid = fd[(fd > lo) & (fd <= hi)]
    rng = np.random.RandomState(SEED)
    sampled = pd.Series(rng.choice(valid.values, size=len(lab)), index=lab.index)
    cut = fd.where((lab["is_mule"] == 1) & fd.between(lo, hi, inclusive="right"), sampled)
    out = pd.DataFrame({"account_id": lab["account_id"], "cutoff": cut})
    out = pd.concat([out, pd.DataFrame({"account_id": test["account_id"],
                                        "cutoff": pd.Series(rng.choice(valid.values, size=len(test)))})])
    out["acct"] = out["account_id"].str[5:].astype(int)
    return out[["acct", "cutoff"]]


def main(mode="full"):
    assert mode in ("full", "asof")
    shards = sorted(glob.glob(str(NFPC_SHARDS / "shard=*")), key=lambda p: int(p.split("=")[-1]))
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false; SET threads=6; SET memory_limit='6GB'; "
                "SET preserve_insertion_order=false")
    acc = pd.read_parquet(NFPC_DIR / "accounts.parquet", columns=["account_id", "last_mobile_update_date"])
    acc["acct"] = acc["account_id"].str[5:].astype(int)
    acc["mdate"] = pd.to_datetime(acc["last_mobile_update_date"], errors="coerce")
    con.register("mobile_upd", acc.loc[acc["mdate"].notna(), ["acct", "mdate"]])
    cutoff_join = ""
    if mode == "asof":
        con.register("cutoffs", asof_cutoffs())
        cutoff_join = "JOIN cutoffs c ON c.acct = x.acct AND x.ts < c.cutoff"

    parts = {k: [] for k in ("feats", "monthly", "edges", "ips")}
    t0 = time.time()
    for i, s in enumerate(shards):
        con.execute(f"CREATE OR REPLACE TEMP TABLE shard AS SELECT * FROM read_parquet('{s}/*.parquet')")
        parts["feats"].append(con.execute(FEATURE_SQL.format(cutoff_join=cutoff_join)).df())
        src = f"(SELECT x.* FROM shard x {cutoff_join})"
        parts["monthly"].append(con.execute(f"""
            SELECT acct, date_trunc('month', ts) AS month, count(*) n,
                   coalesce(sum(abs(amount)) FILTER (WHERE is_credit),0) credit,
                   coalesce(sum(abs(amount)) FILTER (WHERE NOT is_credit),0) debit
            FROM {src} GROUP BY 1, 2""").df())
        parts["edges"].append(con.execute(f"""
            SELECT acct, cp, is_credit, count(*)::INT n, sum(abs(amount))::FLOAT v,
                   min(ts) first_ts, max(ts) last_ts
            FROM {src} GROUP BY 1, 2, 3""").df())
        parts["ips"].append(con.execute(
            f"SELECT DISTINCT acct, ip FROM {src} WHERE ip IS NOT NULL").df())
        print(f"  shard {i + 1}/{len(shards)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in parts.items():
        df = pd.concat(v, ignore_index=True)
        df.to_parquet(OUT / f"{'txn_feats' if k == 'feats' else k}_{mode}.parquet", index=False)
        print(f"  {k}: {df.shape}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["full"]))
