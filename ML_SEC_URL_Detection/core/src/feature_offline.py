# core/src/feature_offline.py
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# phishing.csv 30 dimensional features
OFFLINE30_COLS = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport"
]

SHORTENER_RE = re.compile(
    r"(bit\.ly|goo\.gl|shorte\.st|ow\.ly|t\.co|tinyurl\.com|is\.gd|cli\.gs|lnkd\.in|db\.tt|adf\.ly|bitly\.com)",
    re.IGNORECASE
)

# Keywords suspicious for phishing
SUSPICIOUS_TOKENS = {
    "login", "verify", "update", "secure", "account", "bank", "confirm",
    "signin", "password", "billing", "webscr", "support", "service",
    "authentication", "authorize", "wallet", "payment"
}

# Simple IP address regex
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "http://" + u
    return u


def _host(u: str) -> str:
    try:
        return urlparse(u).netloc or ""
    except Exception:
        return ""


def _path(u: str) -> str:
    try:
        return urlparse(u).path or ""
    except Exception:
        return ""


def _query(u: str) -> str:
    try:
        return urlparse(u).query or ""
    except Exception:
        return ""


def _scheme(u: str) -> str:
    try:
        return urlparse(u).scheme or ""
    except Exception:
        return ""


def _tokenize(u: str) -> list[str]:
    return [t for t in re.split(r"[\W_]+", u.lower()) if t]


@dataclass
class FeatureExtractionOffline30:
    """
    Offline approximation of the original 30 features.
    - No network calls
    - Uses URL lexical/structural heuristics
    - Output order matches OFFLINE30_COLS exactly
    """

    url: str

    def extract(self) -> list[float]:
        u = _normalize_url(self.url)
        host = _host(u)
        path = _path(u)
        query = _query(u)
        scheme = _scheme(u).lower()
        host_no_port = host.split(":")[0].lower()
        tokens = set(_tokenize(host + " " + path + " " + query))

        feats = {
            "UsingIP": self.UsingIP(u, host_no_port),
            "LongURL": self.LongURL(u),
            "ShortURL": self.ShortURL(u),
            "Symbol@": self.SymbolAt(u),
            "Redirecting//": self.Redirecting(u),
            "PrefixSuffix-": self.PrefixSuffix(host_no_port),
            "SubDomains": self.SubDomains(host_no_port),
            "HTTPS": self.HTTPS(scheme),
            "DomainRegLen": self.DomainRegLen(host_no_port, tokens),
            "Favicon": self.Favicon(u, host_no_port, tokens),
            "NonStdPort": self.NonStdPort(host),
            "HTTPSDomainURL": self.HTTPSDomainURL(host_no_port),
            "RequestURL": self.RequestURL(u, tokens),
            "AnchorURL": self.AnchorURL(u, tokens),
            "LinksInScriptTags": self.LinksInScriptTags(u, tokens),
            "ServerFormHandler": self.ServerFormHandler(u, tokens),
            "InfoEmail": self.InfoEmail(u, tokens),
            "AbnormalURL": self.AbnormalURL(u, host_no_port),
            "WebsiteForwarding": self.WebsiteForwarding(u),
            "StatusBarCust": self.StatusBarCust(u, tokens),
            "DisableRightClick": self.DisableRightClick(u, tokens),
            "UsingPopupWindow": self.UsingPopupWindow(u, tokens),
            "IframeRedirection": self.IframeRedirection(u, tokens),
            "AgeofDomain": self.AgeofDomain(host_no_port, tokens),
            "DNSRecording": self.DNSRecording(host_no_port),
            "WebsiteTraffic": self.WebsiteTraffic(host_no_port, tokens),
            "PageRank": self.PageRank(host_no_port, tokens),
            "GoogleIndex": self.GoogleIndex(host_no_port, tokens),
            "LinksPointingToPage": self.LinksPointingToPage(u),
            "StatsReport": self.StatsReport(host_no_port),
        }

        return [float(feats[name]) for name in OFFLINE30_COLS]

    # 1 UsingIP
    def UsingIP(self, u: str, host_no_port: str) -> int:
        return -1 if IP_RE.match(host_no_port or "") else 1

    # 2 LongURL
    def LongURL(self, u: str) -> int:
        if len(u) < 54:
            return 1
        if 54 <= len(u) <= 75:
            return 0
        return -1

    # 3 ShortURL
    def ShortURL(self, u: str) -> int:
        return -1 if SHORTENER_RE.search(u) else 1

    # 4 Symbol@
    def SymbolAt(self, u: str) -> int:
        return -1 if "@" in u else 1

    # 5 Redirecting//
    def Redirecting(self, u: str) -> int:
        return -1 if u.rfind("//") > 6 else 1

    # 6 PrefixSuffix-
    def PrefixSuffix(self, host_no_port: str) -> int:
        return -1 if "-" in (host_no_port or "") else 1

    # 7 SubDomains
    def SubDomains(self, host_no_port: str) -> int:
        dot_count = host_no_port.count(".")
        if dot_count <= 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8 HTTPS
    def HTTPS(self, scheme: str) -> int:
        return 1 if scheme == "https" else -1

    # 9 DomainRegLen (Offline cannot perform whois, use heuristic approximation: suspicious tokens/new domain features)
    def DomainRegLen(self, host_no_port: str, tokens: set[str]) -> int:
        # Heuristic approximation: if domain is long/mixed with digits/suspicious tokens >= 2 -> -1, otherwise 0/1
        if not host_no_port:
            return -1
        digit_ratio = sum(ch.isdigit() for ch in host_no_port) / max(1, len(host_no_port))
        suspicious = len(tokens.intersection(SUSPICIOUS_TOKENS))
        if digit_ratio > 0.3 or suspicious >= 2 or len(host_no_port) >= 35:
            return -1
        if suspicious == 1 or len(host_no_port) >= 25:
            return 0
        return 1

    # 10 Favicon (Offline cannot parse HTML, heuristic: if appears to be fake login/payment page -> -1, otherwise 0/1)
    def Favicon(self, u: str, host_no_port: str, tokens: set[str]) -> int:
        # Heuristic approximation: appearance of login/verify/bank etc., and not https -> more suspicious
        if len(tokens.intersection(SUSPICIOUS_TOKENS)) >= 1 and "https://" not in u.lower():
            return -1
        return 0  # unknown -> 0 is reasonable

    # 11 NonStdPort
    def NonStdPort(self, host: str) -> int:
        return -1 if ":" in (host or "") else 1

    # 12 HTTPSDomainURL
    def HTTPSDomainURL(self, host_no_port: str) -> int:
        return -1 if "https" in (host_no_port or "") else 1

    # 13 RequestURL (Original definition: external link resource ratio. Offline approximation: whether URL contains many external domain redirects/resource traces)
    def RequestURL(self, u: str, tokens: set[str]) -> int:
        # Heuristic approximation: if query contains many url=/redirect=/next=, usually redirect/resource reference
        q = _query(u).lower()
        flags = ["url=", "redirect=", "next=", "target=", "dest=", "destination="]
        cnt = sum(f in q for f in flags)
        if cnt >= 2:
            return -1
        if cnt == 1:
            return 0
        return 1

    # 14 AnchorURL (Original definition: unsafe anchor ratio. Offline approximation: whether it is a 'login/redirect' type URL)
    def AnchorURL(self, u: str, tokens: set[str]) -> int:
        # Heuristic approximation: contains many suspicious tokens or contains fragment/javascript
        if "#javascript" in u.lower():
            return -1
        suspicious = len(tokens.intersection(SUSPICIOUS_TOKENS))
        if suspicious >= 2:
            return -1
        if suspicious == 1:
            return 0
        return 1

    # 15 LinksInScriptTags (Offline approximation: whether obviously loading scripts/tracking parameters)
    def LinksInScriptTags(self, u: str, tokens: set[str]) -> int:
        q = _query(u).lower()
        # Common tracking/script parameters
        markers = ["utm_", "gclid", "fbclid", "script", "js=", "callback="]
        cnt = sum(m in q for m in markers)
        if cnt >= 2:
            return -1
        if cnt == 1:
            return 0
        return 1

    # 16 ServerFormHandler (Offline approximation: whether path looks like 'form submission/login handling')
    def ServerFormHandler(self, u: str, tokens: set[str]) -> int:
        path = _path(u).lower()
        # Common form handling endpoints
        if any(x in path for x in ["/login", "/signin", "/verify", "/submit", "/auth", "/session"]):
            return 0 if "https://" in u.lower() else -1
        return 1

    # 17 InfoEmail (Offline approximation: whether URL contains mailto/email related content)
    def InfoEmail(self, u: str, tokens: set[str]) -> int:
        low = u.lower()
        if "mailto:" in low or "email=" in low:
            return -1
        return 1

    # 18 AbnormalURL (Offline approximation: empty host/abnormal characters)
    def AbnormalURL(self, u: str, host_no_port: str) -> int:
        if not host_no_port:
            return -1
        if re.search(r"[^\x00-\x7F]", u):  # Non-ASCII (possibly IDN spoofing)
            return 0
        return 1

    # 19 WebsiteForwarding (Offline approximation: whether redirect type parameters)
    def WebsiteForwarding(self, u: str) -> int:
        q = _query(u).lower()
        if any(k in q for k in ["redirect=", "next=", "url=", "target=", "dest="]):
            return 0
        return 1

    # 20 StatusBarCust (Offline cannot analyze JavaScript, return 0)
    def StatusBarCust(self, u: str, tokens: set[str]) -> int:
        return 0

    # 21 DisableRightClick (Offline cannot analyze JavaScript, return 0)
    def DisableRightClick(self, u: str, tokens: set[str]) -> int:
        return 0

    # 22 UsingPopupWindow (Offline cannot analyze JavaScript, return 0)
    def UsingPopupWindow(self, u: str, tokens: set[str]) -> int:
        return 0

    # 23 IframeRedirection (Offline cannot analyze HTML, return 0)
    def IframeRedirection(self, u: str, tokens: set[str]) -> int:
        return 0

    # 24 AgeofDomain (Offline cannot perform whois, use heuristic: suspicious tokens/mixed digits)
    def AgeofDomain(self, host_no_port: str, tokens: set[str]) -> int:
        digit_ratio = sum(ch.isdigit() for ch in host_no_port) / max(1, len(host_no_port))
        suspicious = len(tokens.intersection(SUSPICIOUS_TOKENS))
        if digit_ratio > 0.3 or suspicious >= 2:
            return -1
        if suspicious == 1:
            return 0
        return 1

    # 25 DNSRecording (Offline does not resolve DNS, return 0)
    def DNSRecording(self, host_no_port: str) -> int:
        return 0

    # 26 WebsiteTraffic (Offline has no Alexa/traffic data, return 0)
    def WebsiteTraffic(self, host_no_port: str, tokens: set[str]) -> int:
        return 0

    # 27 PageRank (Offline has no pagerank, return 0)
    def PageRank(self, host_no_port: str, tokens: set[str]) -> int:
        return 0

    # 28 GoogleIndex (Offline has no search index, return 0)
    def GoogleIndex(self, host_no_port: str, tokens: set[str]) -> int:
        return 0

    # 29 LinksPointingToPage (Offline cannot count backlinks, approximation: whether URL is a landing page)
    def LinksPointingToPage(self, u: str) -> int:
        path = _path(u)
        # Heuristic approximation: no path or very short path -> likely homepage (more legitimate)
        if len(path.strip("/")) == 0:
            return 1
        if len(path) <= 10:
            return 0
        return -1

    # 30 StatsReport (Offline blacklist/suspicious TLD heuristic)
    def StatsReport(self, host_no_port: str) -> int:
        if not host_no_port:
            return 1
        # Suspicious TLDs (examples, can be extended based on your data)
        bad_tlds = (".xyz", ".top", ".icu", ".cyou", ".sbs", ".lol", ".click")
        if host_no_port.endswith(bad_tlds):
            return -1
        return 1

