"""Small security helpers."""
import re

_CRED_RE = re.compile(r"(://)([^:/@\s]+):([^@/\s]+)@")


def mask_url(url) -> str:
    """Hide user:password in URLs (e.g. rtsp://admin:pw@host) before logging."""
    return _CRED_RE.sub(r"\1***:***@", str(url))
