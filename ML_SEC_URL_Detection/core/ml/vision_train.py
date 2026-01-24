# core/ml/train_vision.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="core/artifacts/vision_cache/features.csv")
    ap.add_argument("--outdir", default="core/artifacts")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fcsv = Path(args.features_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fcsv)
    vf_cols = [c for c in df.columns if c.startswith("vf_")]
    if not vf_cols:
        raise SystemExit(f"No vf_ columns in {fcsv}. Did you run vision_build_cache?")

    X = df[vf_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("=== Vision classification report ===")
    print(classification_report(y_test, pred, digits=4))

    model_path = outdir / "model_vision.joblib"
    meta_path = outdir / "meta_vision.json"
    joblib.dump(pipe, model_path)

    meta = {
        "mode": "vision",
        "features_csv": str(fcsv),
        "feature_dim": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": args.seed,
        "vf_cols": vf_cols,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved meta : {meta_path}")


if __name__ == "__main__":
    main()
