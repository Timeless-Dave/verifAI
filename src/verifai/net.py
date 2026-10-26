"""Guards for URLs a user pasted. VerifAI only fetches public http(s) addresses."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

Resolver = Callable[..., list]


def public_url_problem(url: str, *, resolve: Resolver = socket.getaddrinfo) -> str | None:
    """Return a reason to refuse the fetch, or None if the URL may be requested.

    DNS failure is not a refusal: the liveness check should report that the
    host could not be reached. Private, loopback, and link-local addresses
    are refused so a pasted answer cannot make this process call the local
    network.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "The URL could not be parsed."
    if parsed.scheme not in {"http", "https"}:
        return "Only http and https URLs are fetched."
    if parsed.username or parsed.password:
        return "URLs with an embedded username or password are not fetched."
    host = parsed.hostname
    if not host:
        return "The URL has no host."
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith((".localhost", ".local")):
        return f"{host} is a local name. VerifAI only fetches public URLs."
    try:
        literal = ipaddress.ip_address(lowered)
    except ValueError:
        literal = None
    if literal is not None:
        if _nonpublic(literal):
            return f"{host} is not a public address. VerifAI will not fetch it."
        return None
    try:
        infos = resolve(host, None)
    except socket.gaierror:
        return None
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if _nonpublic(ip):
            return f"{host} resolves to a non-public address ({ip}). VerifAI will not fetch it."
    return None


def _nonpublic(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
