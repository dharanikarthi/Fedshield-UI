import os
"""NFPC master table: static + behavioural features, label confidence (red-herring
shield), bank partition, and suspect-registry features.

    build_master(mode)          -> DataFrame (one row per labelled account)
    registry_features(m, ...)   -> DataFrame of cross-bank registry features
"""
import numpy as np
import pandas as pd

from fedshield.config import NFPC_DIR, NFPC_TXN_END, NFPC_TXN_START, WORK

OUT = WORK / "nfpc"
YN = ["pan_available", "aadhaar_available", "passport_available", "mobile_banking_flag",
      "internet_banking_flag", "atm_card_flag", "demat_flag", "credit_card_flag", "fastag_flag",
      "nomination_flag", "cheque_allowed", "cheque_availed", "kyc_compliant", "rural_branch",
      "joint_account_flag", "nri_flag"]
# Blocked: post-flag / label-derived columns (see config.NFPC_LEAKAGE) plus the two
# demographic update dates the benchmark found to be injected leakage.
BLOCKED = {"account_status", "freeze_date", "unfreeze_date", "mule_flag_date", "alert_reason",
           "flagged_by_branch", "address_last_update_date", "passbook_last_update_date",
           "name", "address", "phone_number", "branch_address", "branch_city"}


def _days(ref, col):
    return (ref - pd.to_datetime(col, errors="coerce")).dt.days.astype("float32")


def static_table(ref):
    """ref: Series of reference timestamps aligned to accounts (cutoff or REF date)."""
    d = NFPC_DIR
    acc = pd.read_parquet(d / "accounts.parquet")
    link = pd.read_parquet(d / "customer_account_linkage.parquet")
    cust = pd.read_parquet(d / "customers.parquet")
    demo = pd.read_parquet(d / "demographics.parquet")
    prod = pd.read_parquet(d / "product_details.parquet")
    br = pd.read_parquet(d / "branch.parquet")
    sch = pd.read_parquet(d / "accounts-additional.parquet")
    m = (acc.merge(link, on="account_id", how="left").merge(cust, on="customer_id", how="left")
            .merge(demo, on="customer_id", how="left").merge(prod, on="customer_id", how="left")
            .merge(br, on="branch_code", how="left").merge(sch, on="account_id", how="left"))
    m["acct"] = m["account_id"].str[5:].astype(int)
    r = m["acct"].map(ref).fillna(pd.Timestamp(NFPC_TXN_END))
    m["account_age_days"] = _days(r, m["account_opening_date"])
    m["customer_age_years"] = _days(r, m["date_of_birth"]) / 365.25
    m["relationship_days"] = _days(r, m["relationship_start_date"])
    m["days_since_kyc"] = _days(r, m["last_kyc_date"])
    mob = _days(r, m["last_mobile_update_date"])
    m["days_since_mobile_update"] = mob.where(mob >= 0)          # future changes are unknown as-of
    m["mobile_changed_after_open"] = (m["days_since_mobile_update"] < m["account_age_days"]).astype("int8")
    for c in YN:
        if c in m:
            m[c] = (m[c] == "Y").astype("int8")
    m["gender_f"] = (m["gender"] == "F").astype("int8")
    m["pin_mismatch_perm"] = (m["customer_pin"] != m["permanent_pin"]).astype("int8")
    m["pin_mismatch_branch"] = (m["customer_pin"] != m["branch_pin"]).astype("int8")
    m["pin_region_mismatch"] = ((m["customer_pin"] // 1000) != (m["branch_pin"].fillna(0) // 1000)).astype("int8")
    m["n_accounts_customer"] = m.groupby("customer_id")["account_id"].transform("count").astype("int8")
    m["product_family_code"] = m["product_family"].map({"S": 0, "K": 1, "O": 2}).astype("float32")
    for s in ["PMJDY", "PMSBY", "PMJJBY", "APY", "SCSS", "SSA"]:
        m[f"scheme_{s.lower()}"] = (m["scheme_code"] == s).astype("int8")
    m["branch_type_code"] = m["branch_type"].map({"urban": 0, "semi-urban": 1, "rural": 2}).astype("float32")
    m["balance_gap_daily_monthly"] = (m["daily_avg_balance"] - m["monthly_avg_balance"]).astype("float32")
    keep = [c for c in m.columns if c not in BLOCKED and m[c].dtype != object
            and c not in ("customer_pin", "permanent_pin", "branch_pin", "branch_pin_code", "currency_code")]
    keep = list(dict.fromkeys(["acct", "account_id"] + keep + ["branch_state", "customer_id"]))
    return m[keep]


def label_confidence(lab, tx):
    """Red-herring shield: down-weight mule labels whose evidence is implausible.
    Returns (confidence in (0,1], reason codes)."""
    fd = pd.to_datetime(lab["mule_flag_date"], errors="coerce")
    conf = np.ones(len(lab))
    reason = np.array([""] * len(lab), dtype=object)
    mule = lab["is_mule"].to_numpy() == 1
    lo, hi = pd.Timestamp(NFPC_TXN_START), pd.Timestamp(NFPC_TXN_END)
    first = pd.to_datetime(tx["first_ts"]).to_numpy()
    rules = [
        ("flag_outside_window", mule & ~fd.between(lo, hi).to_numpy(), 0.5),
        ("flag_before_activity", mule & (fd.to_numpy() < first), 0.5),
        ("no_alert_reason", mule & lab["alert_reason"].isna().to_numpy(), 0.7),
        ("routine_investigation", mule & (lab["alert_reason"] == "Routine Investigation").to_numpy(), 0.85),
        ("no_transactions", mule & np.isnan(tx["n_txn"].to_numpy(dtype=float)), 0.5),
    ]
    for code, mask, f in rules:
        conf[mask] *= f
        reason[mask] = np.where(reason[mask] == "", code, reason[mask] + "|" + code)
    return np.maximum(conf, 0.25), reason


def derived(m):
    c = lambda a: m[a].astype("float32")
    m["passthrough_ratio"] = (np.minimum(c("sum_credit"), c("sum_debit")) /
                              np.maximum(np.maximum(c("sum_credit"), c("sum_debit")), 1))
    m["credit_debit_ratio"] = c("sum_debit") / c("sum_credit").clip(lower=1)
    m["fan_in_out_ratio"] = c("n_cp_in") / c("n_cp_out").clip(lower=1)
    m["txn_per_cp"] = c("n_txn") / c("n_cp").clip(lower=1)
    span = (pd.to_datetime(m["last_ts"]) - pd.to_datetime(m["first_ts"])).dt.days.astype("float32")
    m["span_days"] = span
    m["txn_velocity"] = c("n_txn") / span.clip(lower=1)
    m["active_day_density"] = c("active_days") / span.clip(lower=1)
    m["burst_ratio"] = c("max_txn_day") / c("median_txn_active_day").clip(lower=1)
    m["dormancy_wake_intensity"] = c("txn_after_dormancy") / (c("n_txn") / c("active_months").clip(lower=1)).clip(lower=1)
    m["mobile_spike_ratio"] = (c("n_post_mobile_60d") + 1) / (c("n_pre_mobile_60d") + 1)
    m["volume_to_balance"] = (c("sum_credit") / c("active_months").clip(lower=1)) / m["monthly_avg_balance"].abs().clip(lower=100)
    m["balance_drain"] = 1 - (c("bal_min").clip(lower=0) / c("bal_mean").abs().clip(lower=1)).clip(0, 1)
    m["geo_spread"] = c("lat_std").fillna(0) + c("lon_std").fillna(0)
    m["ip_per_txn"] = c("n_ip") / c("n_txn").clip(lower=1)
    return m


def build_master(mode="full"):
    lab = pd.read_parquet(NFPC_DIR / "train_labels.parquet")
    lab["acct"] = lab["account_id"].str[5:].astype(int)
    tx = pd.read_parquet(OUT / f"txn_feats_{mode}.parquet")
    if mode == "asof":
        from fedshield.features.nfpc import asof_cutoffs
        cut = asof_cutoffs().set_index("acct")["cutoff"]
    else:
        cut = pd.Series(dtype="datetime64[ns]")
    st = static_table(cut)
    m = lab.merge(st, on=["acct", "account_id"], how="left").merge(tx, on="acct", how="left")
    m = derived(m)
    conf, why = label_confidence(m, m)
    m["label_conf"], m["rh_reason"] = conf.astype("float32"), why
    return m


def partition(m, scheme="state", n_banks=3, seed=42):
    """state: regional banks (states assigned greedily to hit 40/35/25% sizes).
    pattern: mules skewed by alert_reason family so banks see different typologies."""
    rng = np.random.RandomState(seed)
    if scheme == "state":
        sizes = m.groupby("branch_state").size().sort_values(ascending=False)
        target = np.array([0.40, 0.35, 0.25])[:n_banks] * len(m)
        load, assign = np.zeros(n_banks), {}
        for s, n in sizes.items():
            b = int(np.argmax(target - load))
            assign[s] = b
            load[b] += n
        return m["branch_state"].map(assign).fillna(0).astype(int).to_numpy()
    fam = {"Dormant Account Reactivation": 0, "High-Value Activity on New Account": 0,
           "Post-Contact-Update Spike": 0, "Income-Transaction Mismatch": 0,
           "Structuring Transactions Below Threshold": 1, "Round Amount Pattern": 1,
           "MCC-Amount Distribution Anomaly": 1, "Salary Cycle Anomaly": 1,
           "Rapid Movement of Funds": 2, "Unusual Fund Flow Pattern": 2,
           "Layered Transaction Pattern": 2, "Branch Cluster Investigation": 2,
           "Geographic Anomaly Detected": 2}
    bank = rng.choice(n_banks, size=len(m), p=[0.40, 0.35, 0.25])
    mule = m["is_mule"].to_numpy() == 1
    f = m["alert_reason"].map(fam)
    has = mule & f.notna().to_numpy()
    affinity = rng.rand(len(m)) < 0.75
    bank[has & affinity] = f[has & affinity].astype(int).to_numpy()
    return bank


def _ip_registry(lab, scope, mode):
    """IP links to confirmed mules, out-of-core in DuckDB (the IP table is ~110M rows)."""
    import duckdb
    path = OUT / f"ips_{mode}_shared.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["reg_ip_shared_mule"]).rename_axis("acct")
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{os.environ.get('FEDSHIELD_DUCKDB_MEM', '2GB')}'; SET threads=4; "
              f"SET temp_directory='{(OUT.parent / 'duckdb_tmp').as_posix()}'; SET preserve_insertion_order=false")
    c.register("lab", lab[["acct", "bank", "conf_mule"]])
    part = "ip" if scope == "fed" else "ip, bank"
    q = f"""
      WITH s AS (SELECT i.acct, i.ip, coalesce(l.bank, -1) bank, coalesce(l.conf_mule, 0) cm
                 FROM '{path.as_posix()}' i LEFT JOIN lab l USING (acct)),
           t AS (SELECT acct, sum(cm) OVER (PARTITION BY {part}) - cm AS other FROM s)
      SELECT acct, sum(CASE WHEN other > 0 THEN 1 ELSE 0 END)::FLOAT reg_ip_shared_mule FROM t GROUP BY acct"""
    r = c.execute(q).df().set_index("acct")
    c.close()
    return r


def registry_features(m, bank, confirmed_mask, scope, mode="full"):
    """Counterparty / device links to confirmed mules, excluding the account's own label.

    scope='local': only confirmations from the account's own bank are visible.
    scope='fed'  : the consortium registry (all banks' confirmations, hashed).
    Counterparties are weighted by rarity (1/#accounts touching them) — common
    merchants carry no signal, rare shared counterparties do.
    """
    edges = pd.read_parquet(OUT / f"edges_{mode}.parquet", columns=["acct", "cp", "is_credit"])
    lab = pd.DataFrame({"acct": m["acct"].to_numpy(), "bank": bank,
                        "conf_mule": confirmed_mask.astype(int)})
    bank_of = dict(zip(lab.acct, lab.bank))
    e = edges.drop_duplicates(["acct", "cp"])
    e = e.merge(lab, on="acct", how="left")
    e["bank"] = e["bank"].fillna(-1)
    e["conf_mule"] = e["conf_mule"].fillna(0)
    deg = e.groupby("cp")["acct"].transform("size")
    e["rarity"] = 1.0 / deg
    out = {}
    src = e
    tot = src.groupby("cp")["conf_mule"].transform("sum") if scope == "fed" else         src.groupby(["cp", "bank"])["conf_mule"].transform("sum")
    other = tot - src["conf_mule"]                           # exclude own label
    src = src.assign(hit=(other > 0).astype(int), hits=other)
    src = src.assign(rare_hit=src["hit"] * src["rarity"])
    g = src.groupby("acct").agg(reg_cp_flagged=("hit", "sum"), reg_cp_rare=("rare_hit", "sum"),
                                reg_cp_links=("hits", "sum"), n=("hit", "size"))
    g["reg_cp_share"] = g["reg_cp_flagged"] / g["n"].clip(lower=1)
    out["cp"] = g.drop(columns="n")
    out["ip"] = _ip_registry(lab, scope, mode)
    f = m[["acct"]].merge(out["cp"], left_on="acct", right_index=True, how="left") \
                   .merge(out["ip"], left_on="acct", right_index=True, how="left").fillna(0)
    return f.drop(columns="acct").reset_index(drop=True).astype("float32")
