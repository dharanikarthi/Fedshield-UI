"""Write each bank's private training partition for the distributed (Docker) demo."""
import json
import pickle
import sys

import pandas as pd

from fedshield.config import WORK
from fedshield.experiments.common import class_weights

mode = sys.argv[1] if len(sys.argv) > 1 else "full"
art = pickle.load(open(WORK / "models" / f"nfpc_{mode}.pkl", "rb"))
feats = pd.read_parquet(WORK / "models" / f"nfpc_{mode}_features.parquet")
sc = pd.read_parquet(WORK / "models" / f"nfpc_{mode}_scores.parquet")
df = sc[["acct", "is_mule", "bank", "split", "label_conf"]].merge(feats, on="acct")
out = WORK / "partitions"
out.mkdir(exist_ok=True)
for b, name in enumerate(art["bank_names"]):
    d = df[(df.bank == b) & (df.split == "train")].reset_index(drop=True)
    part = d[art["features"]].copy()
    part["y"] = d["is_mule"].to_numpy()
    part["w"] = class_weights(part["y"].to_numpy(), d["label_conf"].to_numpy())
    part.to_parquet(out / f"{name}.parquet", index=False)
    print(name, part.shape, int(part.y.sum()), "mules")
(out / "features.json").write_text(json.dumps(art["features"]))
