#!/usr/bin/env python3
"""Checks the operator dashboard's authentication (TASK-25) from outside.

Against a running `olmoe dashboard`, it checks that:
  - every control endpoint, the live page and its feed answer 401 without a
    session, with a wrong token, and with a forged or revoked session cookie;
  - a POST from a page served elsewhere is refused (403) even with the token;
  - signing in with the token gives a session cookie that works until sign-out;
  - a node agent with the wrong secret is refused, and so is a frame with a
    bad signature from an agent that did prove the secret.
Only the standard library is needed.

Usage:
  python3 tools/ref/check_dashboard.py --base http://127.0.0.1:8080 --token-file dashboard.token \\
      --node 127.0.0.1:47310 --secret-file node.secret
"""
import argparse
import hashlib
import hmac
import http.client
import json
import os
import socket
import struct
import sys
import urllib.parse

GETS = ["/api/state", "/live", "/telemetry"]
POSTS = ["/api/load", "/api/stop", "/api/nodes/approve", "/api/nodes/forget", "/api/logout"]
FAILED = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def request(base, method, path, body=None, headers=None):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    conn.request(method, path, body=data, headers=h)
    res = conn.getresponse()
    out = res.status, res.getheader("Set-Cookie"), res.read()
    conn.close()
    return out


def check_http(base, token):
    for name, headers in [("no credentials", {}),
                          ("a wrong token", {"Authorization": "Bearer " + "0" * 64}),
                          ("a forged session", {"Cookie": "tf_session=" + os.urandom(32).hex()})]:
        codes = [request(base, "GET", p, headers=headers)[0] for p in GETS]
        codes += [request(base, "POST", p, {}, headers)[0] for p in POSTS]
        check(all(c == 401 for c in codes), f"every control endpoint refuses {name} (got {sorted(set(codes))})")
    status, _, _ = request(base, "POST", "/api/login", {"token": "not-the-token"})
    check(status == 401, "a wrong token cannot sign in")
    status, _, _ = request(base, "POST", "/api/stop", {}, {"Authorization": "Bearer " + token, "Origin": "https://example.com"})
    check(status == 403, f"a cross-site POST is refused even with the token (got {status})")
    status, cookie, _ = request(base, "POST", "/api/login", {"token": token})
    check(status == 200 and cookie is not None, "the token signs in and sets a cookie")
    attrs = [a.strip().lower() for a in (cookie or "").split(";")]
    check("httponly" in attrs and "samesite=strict" in attrs, "the session cookie is HttpOnly and SameSite=Strict")
    session = cookie.split(";")[0] if cookie else ""
    status, _, body = request(base, "GET", "/api/state", headers={"Cookie": session})
    check(status == 200 and b'"nodes"' in body, "the session reads the state")
    status, _, _ = request(base, "POST", "/api/logout", {}, {"Cookie": session})
    check(status == 200, "sign-out succeeds")
    status, _, _ = request(base, "GET", "/api/state", headers={"Cookie": session})
    check(status == 401, "the session no longer works after sign-out")
    status, _, _ = request(base, "GET", "/api/state", headers={"Authorization": "Bearer " + token})
    check(status == 200, "the token works as a Bearer header")


# One node-link frame: magic, kind, from_node, pad, len, pad, seq; then body and a 32-byte MAC.
HEAD = struct.Struct("<IBBHI4xQ")
MAGIC = 0x4E4B4654
HELLO, CHALLENGE, PROOF, WELCOME, REPORT, REFUSE = 1, 2, 3, 4, 5, 8


class Link:
    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.s = socket.create_connection((host, int(port)), timeout=10)
        self.seq = 0
        self.key = None
        self.buf = b""

    def send(self, kind, body, key=None):
        self.seq += 1
        head = HEAD.pack(MAGIC, kind, 1, 0, len(body), self.seq)
        k = key or self.key
        mac = hmac.new(k, head + body, hashlib.sha256).digest() if k else bytes(32)
        self.s.sendall(head + body + mac)

    def recv(self):
        while True:
            if len(self.buf) >= HEAD.size:
                _, kind, _, _, n, _ = HEAD.unpack(self.buf[:HEAD.size])
                total = HEAD.size + n + 32
                if len(self.buf) >= total:
                    body = self.buf[HEAD.size:HEAD.size + n]
                    self.buf = self.buf[total:]
                    return kind, body
            chunk = self.s.recv(65536)
            if not chunk:
                return None
            self.buf += chunk

    def closed(self):
        try:
            while True:
                if self.recv() is None:
                    return True
        except (ConnectionError, socket.timeout):
            return True


def mac(secret, *parts):
    return hmac.new(secret, b"".join(parts), hashlib.sha256).digest()


def check_nodes(addr, secret):
    # A wrong secret: the dashboard's proof does not verify, and our proof is refused.
    link = Link(addr)
    nn = os.urandom(32)
    link.send(HELLO, nn + b"check-wrong-secret")
    kind, body = link.recv()
    wrong = b"x" * 64
    nd, proof = body[:32], body[32:]
    check(kind == CHALLENGE and proof != mac(wrong, b"tf-dashboard", nn, nd), "with a wrong secret, the dashboard's proof does not verify")
    link.send(PROOF, mac(wrong, b"tf-node", nd, nn))
    check(link.closed(), "the dashboard refuses an agent with a wrong secret")

    # The right secret, then a frame signed with another key.
    link = Link(addr)
    nn = os.urandom(32)
    link.send(HELLO, nn + b"check-tamper")
    kind, body = link.recv()
    nd, proof = body[:32], body[32:]
    check(kind == CHALLENGE and proof == mac(secret, b"tf-dashboard", nn, nd), "with the right secret, the dashboard's proof verifies")
    link.send(PROOF, mac(secret, b"tf-node", nd, nn))
    link.key = mac(secret, b"tf-key", nn, nd)
    kind, _ = link.recv()
    check(kind == WELCOME, "the dashboard accepts an agent with the right secret")
    link.send(REPORT, b'{"models":[],"rdma":[],"rank":null}')
    link.send(REPORT, b'{"models":[],"rdma":[],"rank":null}', key=os.urandom(32))
    check(link.closed(), "the dashboard drops the link on a frame with a bad signature")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--secret-file", required=True)
    args = ap.parse_args()
    token = open(args.token_file).read().strip()
    secret = open(args.secret_file).read().strip().encode()
    check_http(args.base, token)
    check_nodes(args.node, secret)
    # The fake agents show up as nodes; leave the list as it was.
    for name in ["check-wrong-secret", "check-tamper"]:
        request(args.base, "POST", "/api/nodes/forget", {"name": name}, {"Authorization": "Bearer " + token})
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
