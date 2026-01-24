# core/ml/adv_train.py
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .adversarial_attack import batch_attack, OFFLINE_FEATURE_COLS


def parse_k_schedule(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k_schedule", default="1,2,3")
    ap.add_argument("--adv_ratio", type=float, default=1.0)
    ap.add_argument("--only_success_adv", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.csv))
    if "class" not in df.columns:
        raise SystemExit("CSV must contain label column named 'class'.")

    missing = [c for c in OFFLINE_FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing feature columns: {missing}")

    X = df[OFFLINE_FEATURE_COLS].astype(float).values
    y = df["class"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # base pipeline
    def make_model():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])

    model = make_model()
    model.fit(X_train, y_train)

    k_list = parse_k_schedule(args.k_schedule)
    if len(k_list) < args.rounds:
        # 不够就用最后一个补齐
        k_list = k_list + [k_list[-1]] * (args.rounds - len(k_list))
    else:
        k_list = k_list[:args.rounds]

    # rounds of adversarial training
    for rid in range(args.rounds):
        k = k_list[rid]

        # evaluate current
        clean_acc = float(np.mean(model.predict(X_test) == y_test))
        X_adv_eval, _, _ = batch_attack(model, X_test, y_test, k=k)
        adv_acc = float(np.mean(model.predict(X_adv_eval) == y_test))

        print(f"[Round {rid+1}/{args.rounds}] k={k} | clean_acc={clean_acc:.4f} | adv_acc={adv_acc:.4f}")

        # generate adversarial examples on training set
        X_adv, _, success = batch_attack(model, X_train, y_train, k=k)

        if args.only_success_adv:
            X_adv_use = X_adv[success]
            y_adv_use = y_train[success]
        else:
            X_adv_use = X_adv
            y_adv_use = y_train

        # subsample by adv_ratio
        if args.adv_ratio < 1.0 and len(X_adv_use) > 0:
            n = int(len(X_adv_use) * args.adv_ratio)
            idx = np.random.default_rng(args.seed + rid).choice(len(X_adv_use), size=n, replace=False)
            X_adv_use = X_adv_use[idx]
            y_adv_use = y_adv_use[idx]

        # mix clean + adv
        if len(X_adv_use) > 0:
            X_mix = np.vstack([X_train, X_adv_use])
            y_mix = np.concatenate([y_train, y_adv_use])
        else:
            X_mix = X_train
            y_mix = y_train

        # re-train from scratch each round (更稳定复现你当时的日志行为)
        model = make_model()
        model.fit(X_mix, y_mix)

    # final reports (k=3 用最后一个k)
    final_clean_pred = model.predict(X_test)
    print("\n=== Final Clean Report ===")
    print(classification_report(y_test, final_clean_pred, digits=4))

    k_final = k_list[-1]
    X_adv_final, _, _ = batch_attack(model, X_test, y_test, k=k_final)
    final_adv_pred = model.predict(X_adv_final)
    print(f"\n=== Final Adversarial Report (k={k_final}) ===")
    print(classification_report(y_test, final_adv_pred, digits=4))

    model_path = outdir / "model_offline_advtrained.joblib"
    meta_path = outdir / "meta_offline_advtrained.json"

    joblib.dump(model, model_path)
    meta = {
        "csv": str(args.csv),
        "mode": "offline_advtrained",
        "rounds": args.rounds,
        "k_schedule": k_list,
        "adv_ratio": args.adv_ratio,
        "only_success_adv": bool(args.only_success_adv),
        "seed": args.seed,
        "test_size": args.test_size,
        "feature_dim": int(X.shape[1]),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved meta : {meta_path}")


if __name__ == "__main__":
    main()
