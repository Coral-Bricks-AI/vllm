# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Network guard for caller-supplied media fetches.

vLLM fetches media from inside the serving network. A caller-supplied URL is
therefore a request-forgery primitive unless every resolved address is public and
every redirect hop is checked. Hostnames are resolved here rather than by the HTTP
client because a public-looking name (``localtest.me``, ``*.nip.io``) can resolve
inward, and the serving resolver—not the URL text—is what the fetch would use.
"""
from __future__ import annotations

import ipaddress
import socket
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

# Only these schemes can reach the network through this fetcher.
ALLOWED_SCHEMES = frozenset({"http", "https"})


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
    """Reject a media URL before a socket is opened.

    Resolves the hostname and refuses the request if any answer is loopback,
    link-local (including instance/task metadata), RFC1918, CGNAT, unique-local,
    or another reserved range. IPv4 spellings that the platform accepts (hex,
    decimal, and mixed forms) are normalized by the resolver and therefore checked
    in their final address form.
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
    for address in addresses:
        if _is_blocked(address):
            raise UnsafeMediaURLError(
                "Media fetch host resolves to a blocked address range."
            )


def redirect_target(url: str, location: str) -> str:
    """Resolve a redirect Location header against its request URL."""
    return location if urlsplit(location).netloc else urljoin(url, location)
