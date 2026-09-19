"""Pytest plugin: allow local test transport; never reach a real cloud provider."""

import socket

_original = None
_connect = None
_connect_ex = None


def pytest_configure(config):
    global _original, _connect, _connect_ex
    _original = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if name not in {"localhost", "127.0.0.1", "::1", None}:
            raise RuntimeError("Offline eval blocked external DNS: " + str(name))
        return _original(host, port, *args, **kwargs)

    socket.getaddrinfo = resolve
    _connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(address):
        if isinstance(address, tuple) and address[0] not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("Offline eval blocked external socket")

    def connect(sock, address):
        check(address)
        return _connect(sock, address)

    def connect_ex(sock, address):
        check(address)
        return _connect_ex(sock, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


def pytest_unconfigure(config):
    if _original is not None:
        socket.getaddrinfo = _original
    if _connect is not None:
        socket.socket.connect = _connect
    if _connect_ex is not None:
        socket.socket.connect_ex = _connect_ex
