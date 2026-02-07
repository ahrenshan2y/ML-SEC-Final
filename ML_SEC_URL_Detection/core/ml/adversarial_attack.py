# core/ml/adversarial_attack.py
from __future__ import annotations
import numpy as np
from sklearn import config_context
from joblib import parallel_backend
import warnings

# 抑制joblib相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')

OFFLINE_FEATURE_COLS = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport"
]
NAME2IDX = {n: i for i, n in enumerate(OFFLINE_FEATURE_COLS)}

REALISTIC_ATTACKABLE = ["HTTPS", "Favicon", "ServerFormHandler", "IframeRedirection", "AnchorURL"]  #high-weight features in phishing classification

DEFAULT_ATTACKABLE_IDX = [NAME2IDX[n] for n in REALISTIC_ATTACKABLE if n in NAME2IDX]


def flip_value(v: float) -> float:
    # 1 <-> -1, 0 -> 1
    if v > 0:
        return -1.0
    if v < 0:
        return 1.0
    return 1.0


def feature_flip_attack_one(
    model,
    x: np.ndarray,
    y_true: int,
    k: int = 3,
    attackable_idx: list[int] | None = None,
):
    if attackable_idx is None:
        attackable_idx = DEFAULT_ATTACKABLE_IDX

    x_adv = x.astype(float).copy()
    flipped: list[int] = []

    # already wrong => treat as "success"
    pred = int(model.predict(x_adv.reshape(1, -1))[0])
    if pred != y_true:
        return x_adv, flipped, True

    use_decision = hasattr(model, "decision_function")

    for _ in range(k):
        best_i = None
        best_score = None

        for i in attackable_idx:
            if i in flipped:
                continue

            cand = x_adv.copy()
            cand[i] = flip_value(cand[i])

            if use_decision:
                s = float(model.decision_function(cand.reshape(1, -1))[0])
                score = -s if y_true == 1 else s
            else:
                # 使用sequential后端确保predict_proba使用单线程
                with parallel_backend('sequential'):
                    proba = model.predict_proba(cand.reshape(1, -1))[0]
                cls = list(model.classes_)
                idx = cls.index(y_true)
                score = -float(proba[idx])

            if best_score is None or score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        x_adv[best_i] = flip_value(x_adv[best_i])
        flipped.append(best_i)

        pred = int(model.predict(x_adv.reshape(1, -1))[0])
        if pred != y_true:
            return x_adv, flipped, True

    return x_adv, flipped, False


def batch_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    attackable_idx: list[int] | None = None,
):
    X_adv = X.astype(float).copy()
    flipped_list: list[list[int]] = []
    success = np.zeros(len(y), dtype=bool)

    # 使用sequential后端禁用并行处理
    with parallel_backend('sequential'):
        for j in range(len(y)):
            xa, flipped, ok = feature_flip_attack_one(
                model, X[j], int(y[j]), k=k, attackable_idx=attackable_idx
            )
            X_adv[j] = xa
            flipped_list.append(flipped)
            success[j] = ok

    return X_adv, flipped_list, success


