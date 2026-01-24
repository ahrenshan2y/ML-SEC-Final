# core/ml/feature_sensitivity.py
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ml.adversarial_attack import OFFLINE_FEATURE_COLS, NAME2IDX, flip_value, REALISTIC_ATTACKABLE


def load_offline_matrix(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["class"].astype(int).to_numpy()
    X = df[OFFLINE_FEATURE_COLS].astype(float).to_numpy()
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts/feature_sensitivity.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    X, y = load_offline_matrix(Path(args.csv))

    pred_clean = model.predict(X)
    acc_clean = float((pred_clean == y).mean())
    print(f"Clean acc = {acc_clean:.4f}")

    rows = []
    for fname in REALISTIC_ATTACKABLE:
        idx = NAME2IDX[fname]
        X_flip = X.copy()
        X_flip[:, idx] = np.vectorize(flip_value)(X_flip[:, idx])

        pred = model.predict(X_flip)
        acc = float((pred == y).mean())
        drop = acc_clean - acc

        rows.append({
            "feature": fname,
            "idx": idx,
            "acc_after_flip": acc,
            "acc_drop": drop
        })

        print(f"{fname:18s} -> acc={acc:.4f} drop={drop:.4f}")

    df_out = pd.DataFrame(rows).sort_values("acc_drop", ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved sensitivity ranking: {out}")


if __name__ == "__main__":
    main()
