# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Network policy for caller-supplied media fetches.

A media URL is fetched from inside the serving network, so its hostname is not
the security boundary: a public-looking name can resolve inward, and a public
URL can redirect inward. Resolve each host before connecting and reject every
answer that belongs to a reserved or internal network.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

ALLOWED_SCHEMES = frozenset({"http", "https"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class UnsafeMediaURLError(ValueError):
    """A media URL would make vLLM connect to a non-public network."""


def _addresses(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    return [ipaddress.ip_address(info[4][0]) for info in infos if len(info[4])]


def _is_blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return any(address in network for network in BLOCKED_NETWORKS)


def assert_safe_media_url(url: str) -> None:
    """Resolve an HTTP(S) media host and reject internal addresses.

    IPv4 spellings accepted by the platform (hex, decimal, and mixed forms) are
    normalized by the resolver and checked in their final address form. Every
    DNS answer is checked, and IPv4-mapped IPv6 is checked as IPv4.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeMediaURLError("Media fetch URL has no host.")
    addresses = _addresses(hostname)
    if not addresses:
        raise UnsafeMediaURLError("Media fetch host has no resolvable address.")
    if any(_is_blocked(address) for address in addresses):
        raise UnsafeMediaURLError(
            "Media fetch host resolves to a blocked address range."
        )


def redirect_target(url: str, location: str) -> str:
    """Resolve a redirect Location header against the current URL."""
    return location if urlsplit(location).netloc else urljoin(url, location)


def is_media_redirect(response: Any) -> bool:
    """True when an HTTP response is a redirect the media fetch should follow."""
    status = getattr(response, "status", None)
    status_code = getattr(response, "status_code", None)
    return status in REDIRECT_STATUSES or status_code in REDIRECT_STATUSES
