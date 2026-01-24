# core/ml/robustness_eval.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .adversarial_attack import batch_attack, OFFLINE_FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=3)  # k-flip attack
    
    args = ap.parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(Path(args.csv))

    if "class" not in df.columns:
        raise SystemExit("CSV must contain label column named 'class'.")

    missing = [c for c in OFFLINE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing feature columns: {missing}")

    X = df[OFFLINE_FEATURE_COLS].astype(float).values
    y = df["class"].astype(int).values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clean_acc = float(np.mean(model.predict(X_test) == y_test))

    X_adv, _, success = batch_attack(model, X_test, y_test, k=args.k)
    adv_acc = float(np.mean(model.predict(X_adv) == y_test))

    print("===== Robustness Evaluation =====")
    print(f"Clean Accuracy       : {clean_acc:.4f}")
    print(f"Adversarial Accuracy : {adv_acc:.4f}")
    print(f"Accuracy Drop        : {(clean_acc - adv_acc):.4f}")

if __name__ == "__main__":
    main()
