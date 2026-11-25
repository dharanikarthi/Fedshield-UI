"""NFPC submission for test_accounts:  account_id,is_mule,suspicious_start,suspicious_end

is_mule      : FedShield final-model probability (submission.csv) and 0/1 at the
               validation-chosen threshold (submission_binary.csv)
window       : from the daily activity table — the account's peak 7-day burst,
               widened while the rolling 7-day volume stays >= 30% of that peak.
               Start = first transaction in the window, end = last. Only filled for
               accounts predicted as mules (blank otherwise).

    python -m fedshield.experiments.submission
"""
import pickle

import duckdb
import numpy as np
import pandas as pd

from fedshield.config import NFPC_DIR, RESULTS, WORK
from fedshield.experiments.nfpc import splits_for
from fedshield.features.nfpc_master import OUT, build_master, derived, partition, registry_features, static_table

SUB_DIR = RESULTS / "submission"


def test_master(m_lab):
    test = pd.read_parquet(NFPC_DIR / "test_accounts.parquet")[["account_id"]]
    test["acct"] = test["account_id"].str[5:].astype(int)
    st = static_table(pd.Series(dtype="datetime64[ns]"))
    tx = pd.read_parquet(OUT / "txn_feats_full.parquet")
    t = test.merge(st, on=["acct", "account_id"], how="left").merge(tx, on="acct", how="left")
    t = derived(t)
    t["is_mule"] = 0
    t["split"] = "submit"
    return t


def windows(accts, ratio=0.30):
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4; SET preserve_insertion_order=false")
    con.register("want", pd.DataFrame({"acct": np.asarray(accts, dtype=np.int64)}))
    d = con.execute(f"""SELECT d.acct, d.d, d.credit + d.debit vol, d.first_ts, d.last_ts
                        FROM '{(WORK / 'nfpc' / 'daily_full.parquet').as_posix()}' d JOIN want USING (acct)
                        ORDER BY acct, d""").df()
    out = {}
    for a, g in d.groupby("acct", sort=False):
        s = g.set_index(pd.to_datetime(g["d"]))["vol"].asfreq("D", fill_value=0.0)
        r = s.rolling(7, min_periods=1).sum().to_numpy()
        k = int(np.argmax(r))
        lo, hi = max(k - 6, 0), k
        while lo > 0 and r[lo - 1] >= ratio * r[k]:
            lo -= 1
        while hi < len(r) - 1 and r[hi + 1] >= ratio * r[k]:
            hi += 1
        days = s.index[lo], s.index[hi]
        inw = g[(pd.to_datetime(g["d"]) >= days[0]) & (pd.to_datetime(g["d"]) <= days[1])]
        out[a] = (pd.Timestamp(inw["first_ts"].min()), pd.Timestamp(inw["last_ts"].max()))
    return out


def main():
    with open(WORK / "models" / "nfpc_full.pkl", "rb") as fh:
        art = pickle.load(fh)
    model, feats, thr = art["model"], art["features"], art["threshold"]
    print("building labelled + test master ...", flush=True)
    m = build_master("full")
    m["split"] = splits_for(m)
    m["bank"] = partition(m, "state")
    state_bank = m.groupby("branch_state")["bank"].agg(lambda s: s.mode().iat[0]).to_dict()
    t = test_master(m)
    t["bank"] = t["branch_state"].map(state_bank).fillna(0).astype(int)
    allm = pd.concat([m, t], ignore_index=True)
    reg = [f for f in feats if f.startswith("reg_")]
    if reg:   # same registry as training: confirmations = training-split mules, consortium scope
        confirmed = ((allm["split"] == "train") & (allm["is_mule"] == 1)).to_numpy()
        rf = registry_features(allm, allm["bank"].to_numpy(), confirmed, "fed", "full")
        allm = pd.concat([allm.reset_index(drop=True), rf], axis=1)
    sub = allm[allm["split"] == "submit"].reset_index(drop=True)
    missing = [f for f in feats if f not in sub.columns]
    assert not missing, f"features missing for test accounts: {missing[:10]}"
    p = model.predict_proba(sub[feats].astype("float32").to_numpy())
    flag = p >= thr
    print(f"  {len(sub):,} test accounts, {int(flag.sum()):,} predicted mules (threshold {thr:.3f})", flush=True)
    win = windows(sub.loc[flag, "acct"])
    fmt = lambda ts: "" if ts is None or pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M:%S")
    starts = [fmt(win.get(a, (None, None))[0]) if f else "" for a, f in zip(sub["acct"], flag)]
    ends = [fmt(win.get(a, (None, None))[1]) if f else "" for a, f in zip(sub["acct"], flag)]
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame({"account_id": sub["account_id"], "suspicious_start": starts, "suspicious_end": ends})
    base.insert(1, "is_mule", np.round(p, 6))
    base.to_csv(SUB_DIR / "submission.csv", index=False)
    b2 = base.copy()
    b2["is_mule"] = flag.astype(int)
    b2.to_csv(SUB_DIR / "submission_binary.csv", index=False)
    w = [(e - s).days for s, e in win.values()]
    print(f"  windows: {len(win):,}  median {np.median(w):.0f} days  p90 {np.percentile(w, 90):.0f} days")
    print(f"saved {SUB_DIR / 'submission.csv'} and submission_binary.csv")


if __name__ == "__main__":
    main()
