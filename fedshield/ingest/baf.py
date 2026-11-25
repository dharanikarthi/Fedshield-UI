"""Bank Account Fraud (NeurIPS 2022) adapter — onboarding/KYC fraud.

No institution field exists, so banks are formed from the application's
`payment_type` (product mix differs by institution), giving a non-IID split.
Protocol (from the BAF paper): train on months 0-5, test on months 6-7.
Protected attribute for the fairness audit: customer_age >= 50.

    python -m fedshield.ingest.baf [Base]
"""
import sys

import duckdb
import pandas as pd

from fedshield.config import BAF_DIR, BAF_WORK

CATS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
BANK_OF_PAYMENT = {"AA": 0, "AB": 1, "AC": 2, "AD": 2, "AE": 2}


def main(variant="Base"):
    BAF_WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    df = con.execute(
        f"select * from read_csv_auto('{(BAF_DIR / f'{variant}.csv').as_posix()}')").df()
    df["bank"] = df["payment_type"].map(BANK_OF_PAYMENT).fillna(2).astype("int8")
    df["age_50plus"] = (df["customer_age"] >= 50).astype("int8")
    df["split"] = (df["month"] >= 6).map({False: "train", True: "test"})
    # BAF encodes missing as -1 in several numeric columns; keep the sentinel and add flags.
    for c in ["prev_address_months_count", "current_address_months_count",
              "intended_balcon_amount", "bank_months_count", "session_length_in_minutes"]:
        df[f"{c}_missing"] = (df[c] < 0).astype("int8")
    df = pd.get_dummies(df, columns=CATS, dtype="int8")
    df = df.rename(columns={"fraud_bool": "y"})
    out = BAF_WORK / f"{variant.replace(' ', '_')}.parquet"
    df.to_parquet(out, index=False)
    g = df.groupby(["bank", "split"])["y"].agg(["size", "sum", "mean"])
    print(g.to_string())
    print(f"-> {out}  ({df.shape[1]} cols)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["Base"]))
