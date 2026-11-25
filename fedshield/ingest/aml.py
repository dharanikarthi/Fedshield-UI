"""IBM AMLworld adapter.

Builds, for one AMLworld variant (default HI-Small):
  work/aml/<variant>/txns.parquet      normalised transactions (USD amounts, int ids)
  work/aml/<variant>/accounts.parquet  account -> institution, entity, label, typology
  work/aml/<variant>/features.parquet  per-account features from the OWNING bank's view

Institutions: AMLworld has ~30k bank IDs. We group them into N_INST consortium
members by stable hash of the bank ID (the busiest bank ID is a 15-account hub,
so it cannot stand alone as an institution). A
member only sees transactions that touch one of its own accounts, which is the
real-world visibility constraint federation has to work around.

    python -m fedshield.ingest.aml [HI-Small]
"""
import re
import sys
import time

import duckdb
import pandas as pd

from fedshield.config import AML_DIR, AML_WORK

N_INST = 5


def parse_patterns(path):
    """Return DataFrame(src_bank, src_acct, dst_bank, dst_acct, ts, pattern, attempt)."""
    rows, pat, att = [], None, -1
    rx = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("BEGIN"):
                m = rx.search(line)
                pat = m.group(1) if m else "OTHER"
                att += 1
            elif line.startswith("END"):
                pat = None
            elif pat and line:
                p = line.split(",")
                rows.append((p[1], p[2], p[3], p[4], p[0], pat, att))
    return pd.DataFrame(rows, columns=["src_bank", "src_acct", "dst_bank", "dst_acct", "ts", "pattern", "attempt"])


def main(variant="HI-Small"):
    out = AML_WORK / variant
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false; SET threads=3; SET memory_limit='3500MB'; SET preserve_insertion_order=false")
    t0 = time.time()

    trans = (AML_DIR / f"{variant}_Trans.csv").as_posix()
    con.execute(f"""
    CREATE TABLE raw AS SELECT * FROM read_csv('{trans}', header=true, all_varchar=true,
      names=['ts','from_bank','from_acct','to_bank','to_acct','amt_recv','cur_recv',
             'amt_paid','cur_paid','fmt','is_laundering'])
    """)
    # USD conversion: median rate observed on cross-currency txns into/out of USD.
    con.execute("""
    CREATE TABLE fx AS
    SELECT cur, median(rate) AS usd_rate FROM (
      SELECT cur_paid AS cur, amt_recv::DOUBLE / NULLIF(amt_paid::DOUBLE,0) AS rate
        FROM raw WHERE cur_recv='US Dollar' AND cur_paid<>'US Dollar'
      UNION ALL
      SELECT cur_recv, amt_paid::DOUBLE / NULLIF(amt_recv::DOUBLE,0)
        FROM raw WHERE cur_paid='US Dollar' AND cur_recv<>'US Dollar'
    ) GROUP BY cur
    UNION ALL SELECT 'US Dollar', 1.0
    """)
    # Stable integer ids for accounts (bank+account is the unique key).
    con.execute("""
    CREATE TABLE acct_ids AS
    SELECT k, row_number() OVER (ORDER BY k) - 1 AS id FROM (
      SELECT DISTINCT from_bank || ':' || from_acct AS k FROM raw
      UNION SELECT DISTINCT to_bank || ':' || to_acct FROM raw)
    """)
    big_bank = con.execute(
        "select from_bank from raw group by 1 order by count(*) desc limit 1").fetchone()[0]
    con.execute(f"""
    CREATE TABLE txns AS
    SELECT
      strptime(r.ts, '%Y/%m/%d %H:%M') AS ts,
      s.id AS src, d.id AS dst,
      r.from_bank AS src_bank, r.to_bank AS dst_bank,
      hash(r.from_bank) % {N_INST} AS src_inst,
      hash(r.to_bank) % {N_INST} AS dst_inst,
      r.amt_paid::DOUBLE * coalesce(fx.usd_rate, 1.0) AS usd,
      r.cur_paid <> r.cur_recv AS fx_switch,
      r.cur_paid AS cur, r.fmt AS fmt,
      r.is_laundering::INT AS y
    FROM raw r
    JOIN acct_ids s ON s.k = r.from_bank || ':' || r.from_acct
    JOIN acct_ids d ON d.k = r.to_bank || ':' || r.to_acct
    LEFT JOIN fx ON fx.cur = r.cur_paid
    """)
    con.execute(f"COPY txns TO '{(out / 'txns.parquet').as_posix()}' (FORMAT parquet)")

    # Account table: institution, entity, label, typology
    acc_csv = (AML_DIR / f"{variant}_accounts.csv").as_posix()
    con.execute(f"""
    CREATE TABLE ents AS SELECT * FROM read_csv('{acc_csv}', header=true, all_varchar=true,
      names=['bank_name','bank_id','acct','entity_id','entity_name'])
    """)
    pats = parse_patterns(AML_DIR / f"{variant}_Patterns.txt")
    con.register("pats", pats)
    con.execute(f"""
    CREATE TABLE accounts AS
    WITH a AS (
      SELECT id, split_part(k, ':', 1) AS bank, split_part(k, ':', 2) AS acct FROM acct_ids),
    lab AS (
      SELECT acct_id, max(y) AS y FROM (
        SELECT src AS acct_id, y FROM txns UNION ALL SELECT dst, y FROM txns) GROUP BY 1),
    typ AS (
      SELECT k, first(pattern ORDER BY attempt) AS pattern, count(DISTINCT attempt) AS n_attempts FROM (
        SELECT src_bank || ':' || src_acct AS k, pattern, attempt FROM pats
        UNION ALL SELECT dst_bank || ':' || dst_acct, pattern, attempt FROM pats) GROUP BY 1)
    SELECT a.id AS acct_id, a.bank,
      hash(a.bank) % {N_INST} AS inst,
      e.entity_id, coalesce(e.entity_name, '') AS entity_name,
      coalesce(lab.y, 0) AS y, typ.pattern, coalesce(typ.n_attempts, 0) AS n_attempts
    FROM a LEFT JOIN lab ON lab.acct_id = a.id
    LEFT JOIN ents e ON e.bank_id = a.bank AND e.acct = a.acct
    LEFT JOIN typ ON typ.k = a.bank || ':' || a.acct
    """)
    con.execute(f"COPY accounts TO '{(out / 'accounts.parquet').as_posix()}' (FORMAT parquet)")

    # Per-account features, from the owning institution's view: it sees every
    # txn its own account sends or receives, but nothing about the counterparty
    # beyond its id and bank.
    con.execute("""
    CREATE TABLE legs AS
    SELECT src AS acct, dst AS cp, src_inst AS inst, dst_inst AS cp_inst, ts, usd, fmt, cur, fx_switch,
           'out' AS dir FROM txns
    UNION ALL
    SELECT dst, src, dst_inst, src_inst, ts, usd, fmt, cur, fx_switch, 'in' FROM txns
    """)
    con.execute(f"""
    CREATE TABLE features AS
    WITH base AS (
      SELECT acct,
        count(*) FILTER (WHERE dir='out') AS n_out,
        count(*) FILTER (WHERE dir='in')  AS n_in,
        coalesce(sum(usd) FILTER (WHERE dir='out'),0) AS usd_out,
        coalesce(sum(usd) FILTER (WHERE dir='in'),0)  AS usd_in,
        coalesce(max(usd),0) AS usd_max,
        coalesce(median(usd),0) AS usd_median,
        count(DISTINCT cp) FILTER (WHERE dir='out') AS deg_out,
        count(DISTINCT cp) FILTER (WHERE dir='in')  AS deg_in,
        avg((cp_inst <> inst)::INT) AS cross_inst_share,
        avg((cp = acct)::INT) AS self_share,
        avg((fmt='ACH')::INT) AS fmt_ach, avg((fmt='Cheque')::INT) AS fmt_cheque,
        avg((fmt='Credit Card')::INT) AS fmt_card, avg((fmt='Cash')::INT) AS fmt_cash,
        avg((fmt='Wire')::INT) AS fmt_wire, avg((fmt='Bitcoin')::INT) AS fmt_btc,
        avg((fmt='Reinvestment')::INT) AS fmt_reinvest,
        count(DISTINCT cur) AS n_currencies, avg(fx_switch::INT) AS fx_share,
        avg((usd = round(usd, -2))::INT) AS round_share,
        avg((hour(ts) BETWEEN 0 AND 5)::INT) AS night_share,
        count(DISTINCT CAST(ts AS DATE)) AS active_days,
        epoch(max(ts) - min(ts)) / 3600.0 AS span_hours
      FROM legs GROUP BY acct),
    bidir AS (
      SELECT acct, count(*) AS n_bidir FROM (
        SELECT acct, cp FROM legs WHERE dir='out' AND cp<>acct
        INTERSECT SELECT acct, cp FROM legs WHERE dir='in' AND cp<>acct) GROUP BY acct),
    daily AS (
      SELECT acct, max(n_cp_out) AS max_daily_fanout, max(n_cp_in) AS max_daily_fanin FROM (
        SELECT acct, CAST(ts AS DATE) d,
          count(DISTINCT cp) FILTER (WHERE dir='out') n_cp_out,
          count(DISTINCT cp) FILTER (WHERE dir='in') n_cp_in
        FROM legs GROUP BY 1,2) GROUP BY acct),
    passthru AS (
      -- hours between an inbound leg and the next outbound leg
      SELECT acct, median(gap_h) AS med_in_out_gap_h, avg((gap_h < 24)::INT) AS share_out_within_24h FROM (
        SELECT acct, dir,
          epoch(lead(ts) OVER (PARTITION BY acct ORDER BY ts) - ts)/3600.0 AS gap_h,
          lead(dir) OVER (PARTITION BY acct ORDER BY ts) AS next_dir
        FROM legs) WHERE dir='in' AND next_dir='out' GROUP BY acct)
    SELECT a.acct_id, a.inst, a.y, a.pattern, a.entity_id,
      b.* EXCLUDE (acct),
      coalesce(bi.n_bidir,0) AS n_bidir,
      coalesce(bi.n_bidir,0) / greatest(b.deg_out + b.deg_in, 1) AS bidir_ratio,
      d.max_daily_fanout, d.max_daily_fanin,
      coalesce(p.med_in_out_gap_h, -1) AS med_in_out_gap_h,
      coalesce(p.share_out_within_24h, 0) AS share_out_within_24h,
      b.usd_out / greatest(b.usd_in, 1) AS out_in_ratio,
      least(b.usd_out, b.usd_in) / greatest(b.usd_out, b.usd_in, 1) AS passthrough_ratio,
      b.deg_in / greatest(b.deg_out, 1) AS fan_in_out_ratio
    FROM accounts a JOIN base b ON b.acct = a.acct_id
    LEFT JOIN bidir bi ON bi.acct = a.acct_id
    LEFT JOIN daily d ON d.acct = a.acct_id
    LEFT JOIN passthru p ON p.acct = a.acct_id
    """)
    con.execute(f"COPY features TO '{(out / 'features.parquet').as_posix()}' (FORMAT parquet)")
    s = con.execute("""select count(*), sum(y), avg(y), count(distinct inst) from features""").fetchone()
    per = con.execute("select inst, count(*), sum(y), round(avg(y)*100,3) from features group by 1 order by 1").fetchall()
    print(f"{variant}: {s[0]:,} accounts, {s[1]:,} laundering-involved ({s[2]*100:.2f}%), {s[3]} institutions")
    for r in per:
        print(f"  inst {r[0]}: {r[1]:>9,} accts  {r[2]:>6,} pos  {r[3]}%")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["HI-Small"]))
