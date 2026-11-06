"""The browser's IP address for rate limits, behind Railway's proxy and Cloudflare.

Production traffic runs browser -> Cloudflare -> Railway's edge proxy -> uvicorn. deploy.py starts uvicorn
with ``--proxy-headers --forwarded-allow-ips $FORWARDED_ALLOW_IPS``. With ``*`` (the documented cloud
setting, since Railway's proxy is the only ingress) uvicorn rewrites ``request.client`` to the leftmost
X-Forwarded-For entry. The browser writes that entry itself, so keying a limiter on it lets anyone rotate
past the limit or burn someone else's bucket.

So the limiter reads the address the trusted proxy saw instead: the rightmost X-Forwarded-For entry
(appended by Railway's edge) when every peer is trusted, otherwise the peer uvicorn already resolved. When
that address is a Cloudflare edge, Cloudflare sets CF-Connecting-IP (overwriting any value the browser
sent), so that header names the browser. CF-Connecting-IP from anyone else is ignored: a request sent
straight to Railway could carry any value.
"""

import ipaddress
import os
from functools import lru_cache

from starlette.requests import HTTPConnection

# https://www.cloudflare.com/ips-v4 and /ips-v6. A new range missing here only makes requests through
# it share the Cloudflare edge's bucket; it never trusts a forged header.
CLOUDFLARE_RANGES = (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "141.101.64.0/18",
    "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20", "197.234.240.0/22", "198.41.128.0/17",
    "162.158.0.0/15", "104.16.0.0/13", "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32", "2405:8100::/32",
    "2a06:98c0::/29", "2c0f:f248::/32",
)


@lru_cache
def _cloudflare():
    return tuple(ipaddress.ip_network(r) for r in CLOUDFLARE_RANGES)


def parse_ip(value: str | None):
    """An IP address from a header entry (bare, ``v4:port`` or ``[v6]:port``), or None."""
    value = (value or "").strip()
    if value.startswith("[") and "]" in value:
        value = value[1 : value.index("]")]
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped or ip


def is_cloudflare(ip) -> bool:
    return ip is not None and any(ip in net for net in _cloudflare() if net.version == ip.version)


def trusts_every_peer() -> bool:
    # Mirrors uvicorn: only the exact value "*" trusts every peer.
    return os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1").strip() == "*"


def connecting_ip(request: HTTPConnection):
    """The address that opened the connection to our trusted proxy (or to uvicorn directly)."""
    peer = request.client.host if request.client else None
    if trusts_every_peer():
        hops = [h for value in request.headers.getlist("x-forwarded-for") for h in value.split(",") if h.strip()]
        if hops:
            return parse_ip(hops[-1])
    return parse_ip(peer)


def client_ip(request: HTTPConnection) -> str:
    """A rate-limit key for the browser; ``unknown`` when no address can be read."""
    connecting = connecting_ip(request)
    if is_cloudflare(connecting):
        forwarded = parse_ip(request.headers.get("cf-connecting-ip"))
        if forwarded is not None:
            return str(forwarded)
    if connecting is not None:
        return str(connecting)
    return (request.client.host if request.client else None) or "unknown"
