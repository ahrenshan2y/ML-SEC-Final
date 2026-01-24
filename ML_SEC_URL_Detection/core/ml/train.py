from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Union, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_dataset(csv_path: Union[str, Path]) -> Tuple[Union[np.ndarray, np.ndarray], np.ndarray, str, str]:

    df = pd.read_csv(csv_path)

    possible_y = ["label", "target", "y", "class", "Class", "CLASS"]
    y_col = next((c for c in possible_y if c in df.columns), None)
    if y_col is None:
        raise ValueError(f"Cannot find label column in {df.columns}. Expected one of {possible_y}")

    possible_url = ["url", "URL", "Url", "link"]
    url_col = next((c for c in possible_url if c in df.columns), None)
    if url_col is not None:
        urls = df[url_col].astype(str).values
        y = df[y_col].astype(int).values
        return urls, y, url_col, y_col

    drop_cols = set([y_col, "Index", "index", "Id", "id"])
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = df[y_col].astype(int).values

    return X, y, "<features>", y_col


def featurize(
    urls: np.ndarray,
    mode: str,
    timeout: float = 4.0,
    maxn: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert urls -> feature matrix X, and return valid sample mask (which urls successfully extracted features).
    - For offline: basically won't fail (pure string features)
    - For online: may fail, failed samples will be skipped
    """
    if maxn is not None:
        urls = urls[:maxn]

    if mode == "offline":
        from src.feature_offline import URLFeatureExtractor  
        X = [URLFeatureExtractor(u).extract() for u in urls]
        X = np.asarray(X, dtype=float)
        ok_mask = np.ones(len(urls), dtype=bool)
        return X, ok_mask

    if mode == "online":
        from src.feature_online import FeatureExtractionOnline

        if cache_path is not None and cache_path.exists():
            dfc = pd.read_csv(cache_path)

            feat_cols = [c for c in dfc.columns if c not in ("url", "label")]
            X = dfc[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            ok_mask = np.ones(X.shape[0], dtype=bool)
            return X.astype(float), ok_mask

        feats_list = []
        ok_mask = np.zeros(len(urls), dtype=bool)

        for i, u in enumerate(urls):
            try:
                f = FeatureExtractionOnline(u, timeout=timeout).getFeaturesList()
                if len(f) != 30:
                    raise ValueError(f"Expected 30 features, got {len(f)}")
                feats_list.append([float(x) for x in f])
                ok_mask[i] = True
            except Exception:

                ok_mask[i] = False

        X = np.asarray(feats_list, dtype=float)

        if cache_path is not None:
            ok_urls = urls[ok_mask]
            df_cache = pd.DataFrame(X)
            df_cache.insert(0, "url", ok_urls)
            df_cache.to_csv(cache_path, index=False, encoding="utf-8")

        return X, ok_mask

    raise ValueError("mode must be offline or online")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/phishing.csv")
    ap.add_argument("--mode", choices=["offline", "online"], default="offline")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--timeout", type=float, default=4.0, help="online fetching timeout (seconds)")
    ap.add_argument("--maxn", type=int, default=None, help="limit number of urls for faster run")
    ap.add_argument("--cache_feats", action="store_true", help="cache online features to csv to speed up reruns")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, y, url_col, y_col = load_dataset(csv_path)

    if isinstance(data, np.ndarray) and data.ndim == 2:
        X = data
        urls = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

    else:

        urls = data

        cache_path = None
        if args.cache_feats and args.mode == "online":
            cache_path = outdir / f"online_features_cache_maxn{args.maxn or 'all'}.csv"

        X_all, ok_mask = featurize(
            urls,
            args.mode,
            timeout=args.timeout,
            maxn=args.maxn,
            cache_path=cache_path,
        )

        if args.maxn is not None:
            y = y[:args.maxn]

        if ok_mask is not None and ok_mask.shape[0] == len(y):
            y_ok = y[ok_mask]
        else:
            y_ok = y

        if X_all.shape[0] == 0:
            raise SystemExit("No valid samples after feature extraction. Check URLs / network / timeout.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_ok, test_size=args.test_size, random_state=args.seed, stratify=y_ok
        )

        n_total = int(len(y))
        n_ok = int(len(y_ok))
        if args.mode == "online":
            print(f"[online] total urls={n_total}, succeeded={n_ok}, failed={n_total - n_ok}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, pred, digits=4))

    model_path = outdir / f"model_{args.mode}.joblib"
    meta_path = outdir / f"meta_{args.mode}.json"

    joblib.dump(pipe, model_path)

    meta = {
        "mode": args.mode,
        "csv": str(csv_path),
        "url_col": url_col,
        "label_col": y_col,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_dim": int(X_train.shape[1]),
        "seed": args.seed,
        "has_urls": urls is not None,
        "online_timeout": args.timeout if args.mode == "online" else None,
        "maxn": args.maxn,
        "cache_feats": bool(args.cache_feats),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved meta : {meta_path}")


if __name__ == "__main__":
    main()
