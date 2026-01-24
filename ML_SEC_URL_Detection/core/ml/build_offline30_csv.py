# core/ml/build_offline30_csv.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.feature_offline import FeatureExtractionOffline30, OFFLINE30_COLS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV with columns: url,label (or url,class)")
    ap.add_argument("--out_csv", required=True, help="Output CSV with: Index + 30 cols + class")
    ap.add_argument("--maxn", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path, encoding="utf-8", on_bad_lines="skip")
    cols_lower = [c.lower().strip() for c in df.columns]

    if "url" not in cols_lower:
        raise ValueError(f"Missing 'url' column in {df.columns}")

    url_col = df.columns[cols_lower.index("url")]
    if "label" in cols_lower:
        y_col = df.columns[cols_lower.index("label")]
    elif "class" in cols_lower:
        y_col = df.columns[cols_lower.index("class")]
    else:
        raise ValueError(f"Missing 'label' or 'class' column in {df.columns}")

    if args.maxn and args.maxn > 0:
        df = df.head(args.maxn).copy()

    out_rows = []
    for i, row in enumerate(df.itertuples(index=False), start=0):
        u = str(getattr(row, url_col))
        y = int(getattr(row, y_col))
        feats = FeatureExtractionOffline30(u).extract()

        rec = {"Index": i}
        rec.update({k: v for k, v in zip(OFFLINE30_COLS, feats)})
        rec["class"] = y
        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=["Index"] + OFFLINE30_COLS + ["class"])
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(out_df)}, feature_dim: {len(OFFLINE30_COLS)}")


if __name__ == "__main__":
    main()
