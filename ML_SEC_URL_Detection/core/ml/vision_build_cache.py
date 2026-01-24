# core/ml/vision_build_cache.py
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

import cv2

from core.src.vision_features import extract_vision_features_from_image


def infer_url_label_columns(df: pd.DataFrame):
    # 你的数据集当前是 30维特征 + class，通常没有 url 列
    # 所以 vision 模块需要 url 数据集。
    # 如果你没有 url 列，就必须换一个带 url 的数据，或另外准备 urls.csv。
    candidates_url = ["url", "URL", "website", "WebSite", "link"]
    candidates_y = ["label", "Label", "class", "Class", "target", "Target"]
    url_col = next((c for c in candidates_url if c in df.columns), None)
    y_col = next((c for c in candidates_y if c in df.columns), None)
    return url_col, y_col


def make_driver(headless: bool = True):
    opts = Options()
    if headless:
        # new headless for modern Chrome
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1365,768")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--log-level=3")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(15)
    return driver


def screenshot_url(driver, url: str, out_png: Path | None, wait: float = 2.0) -> np.ndarray | None:
    try:
        driver.get(url)
        time.sleep(wait)  # let page render
        png = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if out_png is not None:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            out_png.write_bytes(png)
        return img
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV must contain url + label columns for vision mode.")
    ap.add_argument("--outdir", default="core/artifacts/vision_cache")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--maxn", type=int, default=200, help="Limit samples for quick run. Use -1 for all.")
    ap.add_argument("--wait", type=float, default=2.0)
    ap.add_argument("--save_screenshots", action="store_true", default=False)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    shots_dir = outdir / "screenshots"
    out_features = outdir / "features.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8")

    url_col, y_col = infer_url_label_columns(df)
    if url_col is None:
        raise SystemExit(
            f"Vision cache requires URL column in CSV, but got columns: {list(df.columns)}.\n"
            f"Fix: prepare a new CSV with columns: url,label (recommended), or rename accordingly."
        )
    if y_col is None:
        raise SystemExit("Cannot infer label column. Need one of: label/class/target")

    # subset
    if args.maxn != -1:
        df = df.head(args.maxn).copy()

    driver = make_driver(headless=args.headless)

    rows = []
    try:
        for i, r in tqdm(df.iterrows(), total=len(df)):
            url = str(r[url_col])
            y = int(r[y_col])

            if not url.startswith("http"):
                url = "https://" + url

            out_png = (shots_dir / f"{i}.png") if args.save_screenshots else None
            img = screenshot_url(driver, url, out_png, wait=args.wait)

            feats = extract_vision_features_from_image(img)  # 24 dims
            rows.append([i, url, y] + feats)
    finally:
        driver.quit()

    cols = ["row_id", "url", "label"] + [f"vf_{k:02d}" for k in range(24)]
    out_df = pd.DataFrame(rows, columns=cols)
    out_df.to_csv(out_features, index=False, encoding="utf-8")
    print(f"Saved vision features: {out_features}")
    if args.save_screenshots:
        print(f"Saved screenshots to : {shots_dir}")


if __name__ == "__main__":
    main()
