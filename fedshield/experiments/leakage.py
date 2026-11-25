"""NFPC leakage check: point-in-time (as-of) features vs full-window features.

For a sample (every labelled mule + 12,000 non-mules) the transaction features are
recomputed using ONLY transactions strictly before a per-account cutoff:
  mules with a flag date inside the transaction window -> cutoff = flag date
  everyone else -> cutoff drawn from the same flag-date distribution (so truncation
                   itself carries no label information)
The same model is trained on (a) full-window and (b) as-of features for the same
accounts and split. If (b) holds up, the headline score is not explained by
post-flag activity.

    python -m fedshield.experiments.leakage
"""
import glob
import json
import time

import numpy as np
import pandas as pd

from fedshield.config import NFPC_SHARDS, RESULTS, SEED, WORK
from fedshield.features.nfpc import FEATURE_SQL, asof_cutoffs
from fedshield.features.nfpc_v2 import WINDOW_SQL, connect, mcc_stats

OUT = WORK / "nfpc" / "asof_sample.parquet"


def sample_accounts(n_neg=12000):
    sc = pd.read_parquet(WORK / "models" / "nfpc_full_scores.parquet", columns=["acct", "is_mule", "split"])
    rng = np.random.RandomState(SEED + 1001)   # NOT SEED: the train/test split was drawn with SEED
    neg = sc[sc.is_mule == 0]
    neg = neg.iloc[rng.choice(len(neg), n_neg, replace=False)]
    return pd.concat([sc[sc.is_mule == 1], neg])


def extract(acc):
    con = connect()
    mcc_stats(con)
    cut = asof_cutoffs()
    cut = cut[cut.acct.isin(acc.acct)]
    import fedshield.config as C
    a = pd.read_parquet(C.NFPC_DIR / "accounts.parquet", columns=["account_id", "last_mobile_update_date"])
    a["acct"] = a["account_id"].str[5:].astype(int)
    a["mdate"] = pd.to_datetime(a["last_mobile_update_date"], errors="coerce")
    a = a[a.acct.isin(acc.acct)]
    # mobile update after the cutoff is not yet known at the cutoff
    a = a.merge(cut, on="acct")
    con.register("mobile_upd", a.loc[a["mdate"].notna() & (a["mdate"] < a["cutoff"]), ["acct", "mdate"]])
    con.register("cutoffs", cut)
    parts, t0 = [], time.time()
    shards = sorted(glob.glob(str(NFPC_SHARDS / "shard=*")), key=lambda p: int(p.split("=")[-1]))
    for i, s in enumerate(shards):
        con.execute(f"""CREATE OR REPLACE TEMP TABLE shard AS
                        SELECT x.* FROM read_parquet('{s}/*.parquet') x
                        JOIN cutoffs c ON c.acct = x.acct AND x.ts < c.cutoff""")
        if con.execute("SELECT count(*) FROM shard").fetchone()[0]:
            v1 = FEATURE_SQL.format(cutoff_join="")
            parts.append(con.execute(f"""SELECT a.*, b.* EXCLUDE (acct) FROM ({v1}) a
                                         LEFT JOIN ({WINDOW_SQL.format(lo=0, hi=10**9)}) b USING (acct)""").df())
        con.execute("DROP TABLE shard")
        print(f"  shard {i + 1}/{len(shards)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT)
    return df


def fit_eval(X_tr, y_tr, X_te, y_te):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                          scale_pos_weight=(y_tr == 0).sum() / max(y_tr.sum(), 1), tree_method="hist",
                          n_jobs=4, random_state=SEED, verbosity=0)
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    imp = pd.Series(m.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    return {"auprc": float(average_precision_score(y_te, p)), "auc": float(roc_auc_score(y_te, p)),
            "top": [[k, round(float(v), 4)] for k, v in imp.head(10).items()]}


def main():
    t0 = time.time()
    acc = sample_accounts()
    asof = pd.read_parquet(OUT) if OUT.exists() else extract(acc)
    full = pd.read_parquet(WORK / "models" / "nfpc_full_features.parquet")
    full = full[full.acct.isin(acc.acct)].set_index("acct")
    from fedshield.features.nfpc_master import derived
    drop = [c for c in ("days_since_kyc",) if c in full.columns]       # post-flag re-KYC: label proxy
    reg = [c for c in full.columns if c.startswith("reg_")]
    asx = asof.set_index("acct").reindex(full.index)
    txn_cols = [c for c in asx.columns if c in full.columns]
    # as-of table: transaction columns from pre-cutoff data, derived ratios RECOMPUTED from them
    a = full.copy()
    a[txn_cols] = asx[txn_cols]
    probe = pd.concat([a, asx[[c for c in ("first_ts", "last_ts") if c in asx.columns]]], axis=1)
    der = derived(probe.copy())
    der_cols = [c for c in der.columns if c in full.columns and c not in txn_cols and not der[c].equals(probe[c])]
    a[der_cols] = der[der_cols]
    lab = acc.set_index("acct").reindex(full.index)
    tr = (lab.split != "test").to_numpy()
    y = lab.is_mule.to_numpy().astype(int)
    R = {"accounts": int(len(full)), "mules": int(y.sum()), "base_rate": float(y.mean()),
         "txn_features_recomputed": len(txn_cols), "derived_recomputed": der_cols, "dropped": drop,
         "accounts_with_no_pre_cutoff_txns": int(asx[txn_cols[0]].isna().sum())}
    keep = lambda f, extra=(): f.drop(columns=drop + list(extra))
    R["full_window"] = fit_eval(keep(full)[tr], y[tr], keep(full)[~tr], y[~tr])
    R["as_of"] = fit_eval(keep(a)[tr], y[tr], keep(a)[~tr], y[~tr])
    R["full_window_no_registry"] = fit_eval(keep(full, reg)[tr], y[tr], keep(full, reg)[~tr], y[~tr])
    R["as_of_no_registry"] = fit_eval(keep(a, reg)[tr], y[tr], keep(a, reg)[~tr], y[~tr])
    st = full.drop(columns=drop + reg + txn_cols + der_cols)
    R["static_only"] = fit_eval(st[tr], y[tr], st[~tr], y[~tr])
    with_kyc = full.drop(columns=reg + txn_cols + der_cols)
    R["static_only_with_kyc"] = fit_eval(with_kyc[tr], y[tr], with_kyc[~tr], y[~tr])
    R["seconds"] = time.time() - t0
    (RESULTS / "nfpc_leakage.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "top"})
                      for k, v in R.items()}, indent=1))
    print("as-of top features:", R["as_of"]["top"][:6])


if __name__ == "__main__":
    main()
