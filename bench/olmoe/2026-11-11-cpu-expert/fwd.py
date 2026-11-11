"""Relays TCP connections from LISTEN_PORT (all interfaces) to TARGET_HOST:TARGET_PORT (default 127.0.0.1, WSL2 through localhost)."""
import socket
import sys
import threading

listen_port, target_port = int(sys.argv[1]), int(sys.argv[2])
target_host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"


def pump(src, dst):
    try:
        while True:
            data = src.recv(1 << 16)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    try:
        dst.shutdown(socket.SHUT_WR)
    except OSError:
        pass


srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", listen_port))
srv.listen(1)
print(f"fwd: listening on {listen_port}, relaying to {target_host}:{target_port}", flush=True)
while True:
    a, addr = srv.accept()
    print(f"fwd: connection from {addr}", flush=True)
    b = socket.create_connection((target_host, target_port))
    for s in (a, b):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t1 = threading.Thread(target=pump, args=(a, b))
    t2 = threading.Thread(target=pump, args=(b, a))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    a.close()
    b.close()
    print("fwd: connection closed", flush=True)
