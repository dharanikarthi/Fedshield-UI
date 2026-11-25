"""NFPC ingest: join transactions + transactions_additional, write compact
per-account-range shards.

Transaction IDs are contiguous and both tables are sorted by ID, so each
transactions part only needs the additional files whose ID range overlaps it.
Output: work/nfpc/shards/shard=<k>/*.parquet, one shard per SHARD_SIZE accounts.

    python -m fedshield.ingest.nfpc
"""
import glob
import os
import time

import duckdb

from fedshield.config import NFPC_DIR, NFPC_SHARDS, SHARD_SIZE


def _file_ranges(con, files):
    out = []
    for f in files:
        lo, hi = con.execute(
            "select min(transaction_id), max(transaction_id) from read_parquet(?)", [f]
        ).fetchone()
        out.append((f, lo, hi))
    return out


def main():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false; SET threads=6; SET memory_limit='6GB'")
    txn = sorted(glob.glob(str(NFPC_DIR / "transactions" / "batch-*" / "*.parquet")))
    add = sorted(glob.glob(str(NFPC_DIR / "transactions_additional" / "batch-*" / "*.parquet")))
    print(f"{len(txn)} transaction parts, {len(add)} additional parts")
    add_ranges = _file_ranges(con, add)
    NFPC_SHARDS.mkdir(parents=True, exist_ok=True)
    done_marker = NFPC_SHARDS / "_done.txt"
    done = set(done_marker.read_text().split()) if done_marker.exists() else set()

    t0 = time.time()
    for i, f in enumerate(txn):
        tag = os.path.basename(os.path.dirname(f)) + "_" + os.path.basename(f)[:-8]
        if tag in done:
            continue
        lo, hi = con.execute(
            "select min(transaction_id), max(transaction_id) from read_parquet(?)", [f]
        ).fetchone()
        overlap = [a for a, alo, ahi in add_ranges if not (ahi < lo or alo > hi)]
        add_sql = (
            f"(select * from read_parquet({overlap!r}) where transaction_id between '{lo}' and '{hi}')"
            if overlap else
            "(select null::varchar transaction_id, null::double latitude, null::double longitude,"
            " null::varchar ip_address, null::double balance_after_transaction,"
            " null::varchar part_transaction_type, null::varchar atm_deposit_channel_code,"
            " null::varchar transaction_sub_type where false)"
        )
        con.execute(f"""
        COPY (
          SELECT
            CAST(substr(t.account_id, 6) AS INTEGER)              AS acct,
            CAST(t.transaction_timestamp AS TIMESTAMP)            AS ts,
            CAST(t.amount AS FLOAT)                               AS amount,
            (t.txn_type = 'C')                                    AS is_credit,
            t.channel                                             AS channel,
            CAST(t.mcc_code AS SMALLINT)                          AS mcc,
            hash(t.counterparty_id)                               AS cp,
            CAST(a.balance_after_transaction AS FLOAT)            AS bal,
            CAST(a.latitude AS FLOAT)                             AS lat,
            CAST(a.longitude AS FLOAT)                            AS lon,
            CASE WHEN a.ip_address IS NULL THEN NULL ELSE hash(a.ip_address) END AS ip,
            a.part_transaction_type                               AS ptt,
            a.transaction_sub_type                                AS sub,
            a.atm_deposit_channel_code                            AS atm,
            CAST(substr(t.account_id, 6) AS INTEGER) // {SHARD_SIZE} AS shard
          FROM read_parquet('{f}') t
          LEFT JOIN {add_sql} a USING (transaction_id)
        ) TO '{NFPC_SHARDS.as_posix()}'
          (FORMAT parquet, PARTITION_BY (shard), FILENAME_PATTERN '{tag}_{{i}}',
           OVERWRITE_OR_IGNORE, COMPRESSION zstd)
        """)
        with open(done_marker, "a") as fh:
            fh.write(tag + "\n")
        if i % 10 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(txn)} parts  {el / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {NFPC_SHARDS}")


if __name__ == "__main__":
    main()
