# core/ml/vision_predict.py
from __future__ import annotations

import argparse
import time
import joblib
import numpy as np
import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from selenium.common.exceptions import WebDriverException, TimeoutException

import cv2
from core.src.vision_features import extract_vision_features_from_image


def make_driver(headless: bool = True, timeout: float = 15.0):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1365,768")

    # 可选：降低一些站点的渲染/安全策略干扰
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--ignore-certificate-errors")
    opts.add_argument("--allow-running-insecure-content")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(timeout)
    return driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="core/artifacts/model_vision.joblib")
    ap.add_argument("--url", required=True)

    ap.add_argument("--no-headless", action="store_true", help="Run Chrome with UI (not headless)")
    ap.add_argument("--timeout", type=float, default=15.0, help="Page load timeout (seconds)")
    ap.add_argument("--wait", type=float, default=2.0, help="Seconds to wait after load before screenshot")

    # ✅ 新增：允许失败而不抛异常（跑批量很关键）
    ap.add_argument("--allow_fail", action="store_true", help="If set, output error json and exit 0")

    args = ap.parse_args()

    model = joblib.load(args.model)
    url = args.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    headless = not args.no_headless

    driver = make_driver(headless=headless, timeout=args.timeout)
    try:
        try:
            driver.get(url)
            time.sleep(args.wait)
            png = driver.get_screenshot_as_png()
        except (TimeoutException, WebDriverException) as e:
            # 典型：ERR_NAME_NOT_RESOLVED / timeout / SSL 等
            err = str(e).splitlines()[0][:300]
            out = {"mode": "vision", "url": url, "pred": None, "proba": None, "error": err}
            print(out)
            if args.allow_fail:
                return
            raise
    finally:
        driver.quit()

    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    feats = extract_vision_features_from_image(img)  # 24 dims
    x = np.array(feats, dtype=float).reshape(1, -1)

    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0].tolist() if hasattr(model, "predict_proba") else None

    print({"mode": "vision", "url": url, "pred": pred, "proba": proba})


if __name__ == "__main__":
    main()
