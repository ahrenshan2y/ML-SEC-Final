# src/feature_online.py
from __future__ import annotations

import ipaddress
import re
import socket
from datetime import date
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    import whois  # python-whois
except Exception:
    whois = None


class FeatureExtractionOnline:
    """
    Fixed & stabilized version of your original online/content feature extractor.
    Network calls are best-effort with timeouts; failures produce safe defaults.
    """

    def __init__(self, url: str, timeout: float = 4.0):
        self.features = []
        self.url = (url or "").strip()
        self.timeout = timeout

        self.domain = ""
        self.urlparse = None
        self.response = None
        self.soup = None
        self.whois_response = None

        # parse URL
        try:
            if not re.match(r"^https?://", self.url, re.IGNORECASE):
                self.url = "http://" + self.url
            self.urlparse = urlparse(self.url)
            self.domain = self.urlparse.netloc
        except Exception:
            self.domain = ""

        # fetch HTML
        try:
            self.response = requests.get(self.url, timeout=self.timeout, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0"
            })
            self.soup = BeautifulSoup(self.response.text, "html.parser")
        except Exception:
            self.response = None
            self.soup = None

        # whois
        if whois is not None and self.domain:
            try:
                self.whois_response = whois.whois(self.domain.split(":")[0])
            except Exception:
                self.whois_response = None

        # build feature list (keep same 30 slots as your original design)
        self.features.append(self.UsingIp())
        self.features.append(self.longUrl())
        self.features.append(self.shortUrl())
        self.features.append(self.symbol())
        self.features.append(self.redirecting())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.HttpsScheme())
        self.features.append(self.DomainRegLen())
        self.features.append(self.Favicon())

        self.features.append(self.NonStdPort())
        self.features.append(self.HTTPSDomainURL())
        self.features.append(self.RequestURL())
        self.features.append(self.AnchorURL())
        self.features.append(self.LinksInScriptTags())
        self.features.append(self.ServerFormHandler())
        self.features.append(self.InfoEmail())
        self.features.append(self.AbnormalURL())
        self.features.append(self.WebsiteForwarding())
        self.features.append(self.StatusBarCust())

        self.features.append(self.DisableRightClick())
        self.features.append(self.UsingPopupWindow())
        self.features.append(self.IframeRedirection())
        self.features.append(self.AgeofDomain())
        self.features.append(self.DNSRecording())

        # the following are very unstable online signals; keep them but degrade gracefully
        self.features.append(self.WebsiteTraffic())
        self.features.append(self.PageRank())
        self.features.append(self.GoogleIndex())
        self.features.append(self.LinksPointingToPage())
        self.features.append(self.StatsReport())

    def getFeaturesList(self):
        return self.features

    # 1. UsingIp
    def UsingIp(self):
        try:
            ipaddress.ip_address(self.urlparse.hostname or "")
            return -1
        except Exception:
            return 1

    # 2. longUrl
    def longUrl(self):
        if len(self.url) < 54:
            return 1
        if 54 <= len(self.url) <= 75:
            return 0
        return -1

    # 3. shortUrl
    def shortUrl(self):
        match = re.search(
            r"bit\.ly|goo\.gl|shorte\.st|ow\.ly|t\.co|tinyurl|is\.gd|cli\.gs|"
            r"lnkd\.in|db\.tt|adf\.ly|bitly\.com|tinyurl\.com",
            self.url,
            re.IGNORECASE
        )
        return -1 if match else 1

    # 4. Symbol @
    def symbol(self):
        return -1 if "@" in self.url else 1

    # 5. Redirecting //
    def redirecting(self):
        return -1 if self.url.rfind("//") > 6 else 1

    # 6. prefixSuffix "-"
    def prefixSuffix(self):
        try:
            return -1 if "-" in (self.domain or "") else 1
        except Exception:
            return -1

    # 7. SubDomains
    def SubDomains(self):
        dot_count = len(re.findall(r"\.", self.domain or ""))
        if dot_count <= 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8. HTTPS scheme
    def HttpsScheme(self):
        try:
            return 1 if (self.urlparse.scheme.lower() == "https") else -1
        except Exception:
            return 1

    # 9. DomainRegLen
    def DomainRegLen(self):
        try:
            if not self.whois_response:
                return -1
            expiration_date = self.whois_response.expiration_date
            creation_date = self.whois_response.creation_date

            if isinstance(expiration_date, list) and expiration_date:
                expiration_date = expiration_date[0]
            if isinstance(creation_date, list) and creation_date:
                creation_date = creation_date[0]
            if not expiration_date or not creation_date:
                return -1

            age_months = (expiration_date.year - creation_date.year) * 12 + (expiration_date.month - creation_date.month)
            return 1 if age_months >= 12 else -1
        except Exception:
            return -1

    # 10. Favicon
    def Favicon(self):
        try:
            if not self.soup:
                return -1
            for link in self.soup.find_all("link", href=True):
                href = link["href"]
                if "icon" in (link.get("rel") or []) or "icon" in (href.lower()):
                    # accept if same domain or relative
                    if href.startswith("/") or (self.domain and self.domain in href) or (self.url in href):
                        return 1
            return -1
        except Exception:
            return -1

    # 11. NonStdPort
    def NonStdPort(self):
        try:
            return -1 if ":" in (self.domain or "") else 1
        except Exception:
            return -1

    # 12. HTTPSDomainURL
    def HTTPSDomainURL(self):
        try:
            return -1 if "https" in (self.domain or "").lower() else 1
        except Exception:
            return -1

    # 13. RequestURL
    def RequestURL(self):
        try:
            if not self.soup:
                return 0
            i, success = 0, 0
            tags = []
            tags += self.soup.find_all("img", src=True)
            tags += self.soup.find_all("audio", src=True)
            tags += self.soup.find_all("embed", src=True)
            tags += self.soup.find_all("iframe", src=True)

            for t in tags:
                src = t.get("src", "")
                if not src:
                    continue
                i += 1
                # same domain or relative counts as success
                if src.startswith("/") or (self.domain and self.domain in src) or (self.url in src):
                    success += 1

            if i == 0:
                return 0
            percentage = (success / float(i)) * 100
            if percentage < 22.0:
                return 1
            elif percentage < 61.0:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 14. AnchorURL
    def AnchorURL(self):
        try:
            if not self.soup:
                return -1
            i, unsafe = 0, 0
            for a in self.soup.find_all("a", href=True):
                href = (a["href"] or "").lower()
                i += 1
                if ("#" in href) or ("javascript" in href) or ("mailto" in href):
                    unsafe += 1
                else:
                    if not (href.startswith("/") or (self.domain and self.domain in href) or (self.url in href)):
                        unsafe += 1

            if i == 0:
                return -1
            percentage = (unsafe / float(i)) * 100
            if percentage < 31.0:
                return 1
            elif percentage < 67.0:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 15. LinksInScriptTags
    def LinksInScriptTags(self):
        try:
            if not self.soup:
                return 0
            i, success = 0, 0
            for link in self.soup.find_all("link", href=True):
                href = link["href"]
                i += 1
                if href.startswith("/") or (self.domain and self.domain in href) or (self.url in href):
                    success += 1
            for script in self.soup.find_all("script", src=True):
                src = script["src"]
                i += 1
                if src.startswith("/") or (self.domain and self.domain in src) or (self.url in src):
                    success += 1

            if i == 0:
                return 0
            percentage = (success / float(i)) * 100
            if percentage < 17.0:
                return 1
            elif percentage < 81.0:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 16. ServerFormHandler
    def ServerFormHandler(self):
        try:
            if not self.soup:
                return -1
            forms = self.soup.find_all("form", action=True)
            if len(forms) == 0:
                return 1
            for f in forms:
                action = (f.get("action") or "").strip().lower()
                if action in ("", "about:blank"):
                    return -1
                if not (action.startswith("/") or (self.domain and self.domain in action) or (self.url in action)):
                    return 0
            return 1
        except Exception:
            return -1

    # 17. InfoEmail
    def InfoEmail(self):
        try:
            if not self.response:
                return -1
            return -1 if re.search(r"mailto:", self.response.text, re.IGNORECASE) else 1
        except Exception:
            return -1

    # 18. AbnormalURL
    def AbnormalURL(self):
        # Conservative: if whois exists and domain missing => abnormal
        try:
            if not self.domain:
                return -1
            return 1
        except Exception:
            return -1

    # 19. WebsiteForwarding
    def WebsiteForwarding(self):
        try:
            if not self.response:
                return -1
            hlen = len(self.response.history)
            if hlen <= 1:
                return 1
            elif hlen <= 4:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 20. StatusBarCust
    def StatusBarCust(self):
        try:
            if not self.response:
                return -1
            return 1 if re.search(r"onmouseover", self.response.text, re.IGNORECASE) else -1
        except Exception:
            return -1

    # 21. DisableRightClick
    def DisableRightClick(self):
        try:
            if not self.response:
                return -1
            return 1 if re.search(r"event\.button\s*==\s*2", self.response.text) else -1
        except Exception:
            return -1

    # 22. UsingPopupWindow
    def UsingPopupWindow(self):
        try:
            if not self.response:
                return -1
            return 1 if re.search(r"alert\(", self.response.text) else -1
        except Exception:
            return -1

    # 23. IframeRedirection
    def IframeRedirection(self):
        try:
            if not self.response:
                return -1
            return 1 if re.search(r"<iframe|frameborder", self.response.text, re.IGNORECASE) else -1
        except Exception:
            return -1

    # 24. AgeofDomain
    def AgeofDomain(self):
        try:
            if not self.whois_response:
                return -1
            creation_date = self.whois_response.creation_date
            if isinstance(creation_date, list) and creation_date:
                creation_date = creation_date[0]
            if not creation_date:
                return -1
            today = date.today()
            age = (today.year - creation_date.year) * 12 + (today.month - creation_date.month)
            return 1 if age >= 6 else -1
        except Exception:
            return -1

    # 25. DNSRecording
    def DNSRecording(self):
        # If domain resolves, consider OK
        try:
            host = (self.domain or "").split(":")[0]
            if not host:
                return -1
            socket.gethostbyname(host)
            return 1
        except Exception:
            return -1

    # 26. WebsiteTraffic (degraded: return 0 if unknown)
    def WebsiteTraffic(self):
        return 0

    # 27. PageRank (degraded)
    def PageRank(self):
        return 0

    # 28. GoogleIndex (degraded)
    def GoogleIndex(self):
        return 0

    # 29. LinksPointingToPage
    def LinksPointingToPage(self):
        try:
            if not self.response:
                return -1
            n = len(re.findall(r"<a\s+href=", self.response.text, re.IGNORECASE))
            if n == 0:
                return 1
            elif n <= 2:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 30. StatsReport
    def StatsReport(self):
        try:
            if not self.domain:
                return 1
            ip_address = socket.gethostbyname(self.domain.split(":")[0])
            # keep your original blacklist idea, but if lookup fails -> safe
            ip_match = re.search(r"10\.10\.10\.10", ip_address)
            return -1 if ip_match else 1
        except Exception:
            return 1
