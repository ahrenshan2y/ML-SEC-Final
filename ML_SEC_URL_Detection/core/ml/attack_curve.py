# core/ml/attack_curve.py
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.adversarial_attack import OFFLINE_FEATURE_COLS, batch_attack

def load_offline_matrix(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["class"].astype(int).to_numpy()
    X = df[OFFLINE_FEATURE_COLS].astype(float).to_numpy()
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--out", default="artifacts/attack_curve.csv")
    ap.add_argument("--plot", default="artifacts/attack_curve.png")
    args = ap.parse_args()

    model = joblib.load(args.model)
    X, y = load_offline_matrix(Path(args.csv))

    rows = []

    # flip=0 baseline
    pred_clean = model.predict(X)
    acc_clean = float((pred_clean == y).mean())
    rows.append({"flip_k": 0, "accuracy": acc_clean})
    print(f"[k=0] acc={acc_clean:.4f}")

    for k in range(1, args.kmax + 1):
        X_adv, flipped_list, success = batch_attack(model, X, y, k=k)
        pred_adv = model.predict(X_adv)
        acc_adv = float((pred_adv == y).mean())
        rows.append({"flip_k": k, "accuracy": acc_adv, "attack_success_rate": float(success.mean())})
        print(f"[k={k}] acc={acc_adv:.4f} | success_rate={success.mean():.4f}")

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved curve CSV: {out_csv}")

    # plot
    ks = [r["flip_k"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    plt.figure()
    plt.plot(ks, accs, marker="o")
    plt.xlabel("Flip strength k")
    plt.ylabel("Accuracy")
    plt.title("Adversarial strength curve (k-flip)")
    out_png = Path(args.plot)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
