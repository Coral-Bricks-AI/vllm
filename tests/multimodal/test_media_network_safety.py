# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ipaddress
import socket
from unittest.mock import patch

import pytest

from vllm.utils.media_network_safety import (
    BLOCKED_NETWORKS,
    UnsafeMediaURLError,
    assert_safe_media_url,
    is_media_redirect,
    redirect_target,
)


def _resolve(address: str):
    if ":" in address:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, 0, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]


@pytest.mark.parametrize("url,address", [
    ("http://10.0.0.1/x.png", "10.0.0.1"),
    ("http://192.168.0.1/x.png", "192.168.0.1"),
    ("http://172.16.0.1/x.png", "172.16.0.1"),
    ("http://127.0.0.1/x.png", "127.0.0.1"),
    ("http://169.254.169.254/latest/meta-data", "169.254.169.254"),
    ("http://169.254.170.2/v4/credentials", "169.254.170.2"),
    ("http://100.64.0.1/x.png", "100.64.0.1"),
    ("http://240.0.0.1/x.png", "240.0.0.1"),
    ("http://0x0a.0.0.1/x.png", "10.0.0.1"),
    ("http://2130706433/x.png", "127.0.0.1"),
])
def test_direct_and_encoded_internal_addresses_are_refused(url, address):
    with patch("socket.getaddrinfo", return_value=_resolve(address)):
        with pytest.raises(UnsafeMediaURLError):
            assert_safe_media_url(url)


@pytest.mark.parametrize("url,address", [
    ("https://localtest.me/x.png", "127.0.0.1"),
    ("https://10-0-0-1.nip.io/x.png", "10.0.0.1"),
])
def test_public_names_resolving_inward_are_refused(url, address):
    with patch("socket.getaddrinfo", return_value=_resolve(address)):
        with pytest.raises(UnsafeMediaURLError):
            assert_safe_media_url(url)


def test_ipv6_loopback_ula_and_v4_mapped_are_refused():
    for address in ("::1", "fe80::1", "fd00::1", "::ffff:10.0.0.1"):
        with patch("socket.getaddrinfo", return_value=_resolve(address)):
            with pytest.raises(UnsafeMediaURLError):
                assert_safe_media_url(f"http://[{address}]/x.png")


def test_every_answer_is_checked():
    records = _resolve("8.8.8.8") + _resolve("127.0.0.1")
    with patch("socket.getaddrinfo", return_value=records):
        with pytest.raises(UnsafeMediaURLError):
            assert_safe_media_url("https://multi-record.example/x.png")


@pytest.mark.parametrize("url,address", [
    ("https://8.8.8.8/x.png", "8.8.8.8"),
    ("https://2606:4700::1111/x.png", "2606:4700::1111"),
])
def test_public_addresses_pass(url, address):
    with patch("socket.getaddrinfo", return_value=_resolve(address)):
        assert_safe_media_url(url)


@pytest.mark.parametrize("url", [
    "data:image/png;base64,iVBORw0KGgo=",
    "file:///etc/passwd",
    "ftp://internal/x.png",
])
def test_non_http_schemes_are_untouched_by_this_guard(url):
    assert_safe_media_url(url)


def test_blocked_networks_cover_reserved_and_internal_ranges():
    addresses = (
        "0.0.0.1", "10.0.0.1", "100.64.0.1", "127.0.0.1", "169.254.169.254",
        "172.16.0.1", "192.0.0.1", "192.168.0.1", "198.18.0.1", "240.0.0.1",
        "::1", "fe80::1", "fd00::1",
    )
    assert all(
        any(ipaddress.ip_address(value) in network for network in BLOCKED_NETWORKS)
        for value in addresses
    )


def test_relative_redirect_location_resolves_against_current_url():
    assert redirect_target("https://example.com/a/b", "c") == "https://example.com/a/c"
    assert redirect_target("https://example.com/a", "https://other.example/z") == \
        "https://other.example/z"


class _SyncResponse:
    def __init__(self, *, status=200, location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self.content = b"public-image"
        self.closed = False

    @property
    def is_redirect(self):
        return is_media_redirect(self)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def test_sync_media_fetch_checks_each_redirect_hop(monkeypatch):
    from vllm.connections import HTTPConnection

    conn = HTTPConnection()
    calls = []

    def get_response(url, **kwargs):
        calls.append((url, kwargs.get("allow_redirects")))
        if len(calls) == 1:
            return _SyncResponse(status=302, location="http://10.0.0.1/x.png")
        return _SyncResponse()

    monkeypatch.setattr(conn, "get_response", get_response)
    with pytest.raises(UnsafeMediaURLError):
        conn.get_media_bytes("https://public.example/x.png")
    assert calls[0][1] is False


def test_sync_media_fetch_follows_checked_redirects(monkeypatch):
    from vllm.connections import HTTPConnection

    conn = HTTPConnection()
    monkeypatch.setattr(
        conn,
        "get_response",
        lambda url, **kwargs: (_SyncResponse(status=302, location="/safe")
                               if url.endswith("x.png") else _SyncResponse()),
    )
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))],
    )
    assert conn.get_media_bytes("https://public.example/x.png") == b"public-image"


def test_sync_media_fetch_refuses_excessive_redirects(monkeypatch):
    from vllm.connections import HTTPConnection

    conn = HTTPConnection()
    monkeypatch.setattr(
        conn,
        "get_response",
        lambda url, **kwargs: _SyncResponse(status=302, location="/loop"),
    )
    with pytest.raises(UnsafeMediaURLError, match="redirect limit"):
        conn.get_media_bytes("https://public.example/x.png")


def test_generic_get_bytes_still_uses_caller_redirect_policy(monkeypatch):
    from vllm.connections import HTTPConnection

    conn = HTTPConnection()
    monkeypatch.setattr(
        conn,
        "get_response",
        lambda url, **kwargs: _SyncResponse(),
    )
    assert conn.get_bytes("http://10.0.0.1/model.bin", allow_redirects=False) == b"public-image"
