# core/ml/robustness_eval.py
import argparse
import os
from pathlib import Path

import joblib
from joblib import parallel_backend
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import config_context

from .adversarial_attack import batch_attack, OFFLINE_FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=3)  # k-flip attack
    
    args = ap.parse_args()
    
    # 使用joblib的sequential后端来禁用并行处理
    with parallel_backend('sequential'):
        model = joblib.load(args.model)
        # 确保模型使用单线程
        if hasattr(model, 'n_jobs'):
            model.n_jobs = 1
        
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

        # 在sequential后端上执行批量攻击
        X_adv, _, success = batch_attack(model, X_test, y_test, k=args.k)
        
        adv_acc = float(np.mean(model.predict(X_adv) == y_test))

        print("===== Robustness Evaluation =====")
        print(f"Clean Accuracy       : {clean_acc:.4f}")
        print(f"Adversarial Accuracy : {adv_acc:.4f}")
        print(f"Accuracy Drop        : {(clean_acc - adv_acc):.4f}")

if __name__ == "__main__":
    main()
