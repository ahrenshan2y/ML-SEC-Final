# core/src/vision_features.py
from __future__ import annotations

import cv2
import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0

def extract_vision_features_from_image(img_bgr: np.ndarray) -> list[float]:
    """
    Input: BGR image (OpenCV)
    Output: fixed-length feature vector (float list)

    No deep learning, only OpenCV-based layout & appearance features.
    """
    if img_bgr is None or img_bgr.size == 0:
        return [0.0] * 24

    h, w = img_bgr.shape[:2]
    area = float(h * w)

    # 1) Resize for stable statistics (keep aspect by padding or direct resize)
    img = cv2.resize(img_bgr, (800, 450), interpolation=cv2.INTER_AREA)
    h2, w2 = img.shape[:2]
    area2 = float(h2 * w2)

    # 2) Color features (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # mean/std for S,V reflect "flat fake pages" vs complex real pages sometimes
    s_mean = float(np.mean(S))
    s_std = float(np.std(S))
    v_mean = float(np.mean(V))
    v_std = float(np.std(V))

    # 3) Edge / complexity features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = _safe_div(float(np.sum(edges > 0)), area2)

    # 4) Text-like density approximation:
    # Use adaptive threshold + morphology; counts of connected components
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    # remove tiny noise
    kernel = np.ones((3, 3), np.uint8)
    thr2 = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr2, connectivity=8)
    # stats: [x, y, w, h, area]
    # ignore background label=0
    comp_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    comp_count = float(len(comp_areas))
    comp_area_mean = float(np.mean(comp_areas)) if comp_count > 0 else 0.0
    comp_area_std = float(np.std(comp_areas)) if comp_count > 0 else 0.0
    comp_density = _safe_div(comp_count, area2 / 10000.0)  # per 10k pixels (scaled)

    # 5) Layout concentration: how much content is in top region
    top = thr2[: int(h2 * 0.33), :]
    mid = thr2[int(h2 * 0.33): int(h2 * 0.66), :]
    bot = thr2[int(h2 * 0.66):, :]

    top_ratio = _safe_div(float(np.sum(top > 0)), float(np.sum(thr2 > 0)) + 1.0)
    mid_ratio = _safe_div(float(np.sum(mid > 0)), float(np.sum(thr2 > 0)) + 1.0)
    bot_ratio = _safe_div(float(np.sum(bot > 0)), float(np.sum(thr2 > 0)) + 1.0)

    # 6) Large rectangle candidates (form-like boxes)
    # Find contours on threshold image, look for medium/large rectangles
    contours, _ = cv2.findContours(thr2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0.0
    big_rect_count = 0.0
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        a = ww * hh
        if a < 200:  # ignore tiny
            continue
        rect_count += 1.0
        if a > 8000:
            big_rect_count += 1.0

    # 7) Dominant color ratio (simple quantization)
    # many phishing pages have large flat background blocks
    small = cv2.resize(img, (200, 112), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    # quantize
    q = (pixels // 32).astype(np.int32)  # 0..7 per channel
    codes = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(codes, return_counts=True)
    dom_ratio = float(np.max(counts)) / float(np.sum(counts)) if len(counts) else 0.0
    color_unique = float(len(vals))

    feats = [
        # image basic
        float(w), float(h), float(w2), float(h2),

        # color
        s_mean, s_std, v_mean, v_std,
        dom_ratio, color_unique,

        # edges / complexity
        edge_density,

        # components
        comp_count, comp_density, comp_area_mean, comp_area_std,

        # layout ratios
        top_ratio, mid_ratio, bot_ratio,

        # form-like
        rect_count, big_rect_count,

        # threshold ink ratio
        _safe_div(float(np.sum(thr2 > 0)), area2),

        # grayscale stats
        float(np.mean(gray)), float(np.std(gray)),
        float(np.mean(edges > 0)),
    ]

    # fixed length 24
    if len(feats) != 24:
        # safety
        feats = (feats + [0.0] * 24)[:24]
    return feats
