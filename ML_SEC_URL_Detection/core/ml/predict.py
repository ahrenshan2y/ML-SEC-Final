# core/ml/predict.py
from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd

from src.feature_online import FeatureExtractionOnline

FEATURE_COLS_30 = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport"
]


def _label_to_text(pred: int) -> str:
    # 你定义：1 = phishing, 0 = benign
    return "phishing" if pred == 1 else "benign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["online", "offline"], default="online")

    # online：输入 URL -> 在线抓取/解析 -> 得到 30 维 -> 喂给模型
    ap.add_argument("--url", default=None)

    # offline：从 CSV 取一行 30维（用于实验复现/对齐）
    ap.add_argument("--csv", default=None)
    ap.add_argument("--row", type=int, default=None)

    args = ap.parse_args()
    model = joblib.load(args.model)

    if args.mode == "online":
        if not args.url:
            raise SystemExit("online mode requires --url")

        feats = FeatureExtractionOnline(args.url).getFeaturesList()
        if len(feats) != 30:
            raise SystemExit(f"Online feature length must be 30, got {len(feats)}")

        x = np.array(feats, dtype=float).reshape(1, -1)

    else:
        if not args.csv or args.row is None:
            raise SystemExit("offline mode requires --csv <path> and --row <index>")

        df = pd.read_csv(args.csv)
        missing = [c for c in FEATURE_COLS_30 if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV missing columns: {missing}")

        row = df.iloc[int(args.row)]
        feats = [float(row[c]) for c in FEATURE_COLS_30]
        x = np.array(feats, dtype=float).reshape(1, -1)

    pred = int(model.predict(x)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0].tolist()

    print("feature_dim =", x.shape[1])
    print({
        "mode": args.mode,
        "url": args.url if args.mode == "online" else None,
        "pred": pred,
        "pred_text": _label_to_text(pred),
        "proba": proba
    })


if __name__ == "__main__":
    main()
