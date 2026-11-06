# Direct desktop-to-Spark path (2026-11-05)

TASK-21. Until now every cross-machine run went through Tailscale's DERP relay in NYC. **Plain TCP from `gx10-03` to the desktop's campus address is a direct path, about 48 times faster than the relay end to end. Tailscale itself still cannot go direct,** because UDP flows between the machines get cut after a few packets.

## What blocks Tailscale

| Check | Result |
|---|---|
| `tailscale ping`, both directions | always `via DERP(nyc)`, 33 ms; "direct connection not established" |
| `tailscale netcheck`, both ends | UDP works, `MappingVariesByDestIP: false`, so neither NAT should stop a direct connection |
| ICMP ping, `gx10-03` to desktop | 1.6 ms average, no loss |
| Desktop to the Spark's private address | no reply (no route) |
| Raw UDP hole punch, 1 packet/s for 30 s | exactly 3 packets arrive per side (about 2 ms round trip), then nothing |
| Same, 10 packets/s for 15 s | exactly 3 per side |
| Same, 100 packets/s for 5 s | exactly 3 per side |
| Same, 10 packets/s for 20 s, default route and Wi-Fi source | exactly 3 per side on both |

The cutoff follows packet count, not time, and no error comes back. That looks like a firewall on the campus network classifying each UDP flow and then dropping it. This is an inference from the behavior; we cannot see the device. A subnet route would not help, since it runs over the same UDP. We did not try other UDP ports or disguised traffic: if this is policy, opening UDP is a request for campus IT.

## Direct TCP

`gx10-03` connects out to the desktop's campus address, because the Spark sits behind NAT and accepts nothing from outside. So rank 0 (router) runs on `gx10-03` and rank 1 (expert) on the desktop, the reverse of earlier runs. Both machines have 1 Gb/s links. `dispatch` (ReleaseFast): aarch64 with `-Dcpu=cortex_a78` on the Spark, native Windows x86_64 on the desktop. The relay rows use the same roles and programs, connecting to the desktop's Tailscale address. Co-tenant on `gx10-03`: vLLM resident, GPU 0%, load average about 0.5.

| `dispatch router`, 64 slots | Direct TCP | Relay |
|---|---|---|
| `--packets 4096`, total | 720, 884, 1,017, 1,114, 1,197 ms | 49,318 ms |
| Per token (pipelined), median | 248 µs | 12,040 µs |
| Both directions, median | 0.49 Gb/s | 0.01 Gb/s |
| `--latency`, one way, p50 | 772 µs (1,024 round trips) | 18,474 µs (64 round trips) |
| `--latency`, min / p99 | 612 / 1,640 µs | 17,996 / 20,152 µs |
| `--oneway` | 0.57 Gb/s (65,536 packets, twice) | 0.01 Gb/s (4,096 packets) |
| `--moe`, synthetic two-process MoE | PASS, 33 ms | PASS, 500 ms |

All packets verified (CRC, generation, payload), and both processes exited 0 in every run. Larger rings help a little: with `--slots 256`, `--oneway` reaches 0.78 Gb/s and `--packets 4096` takes 893 ms.

The relay's total varies by run: 49 s here, against 27.4 s on Oct 27 with the roles reversed and the desktop side in WSL2.

## OLMoE-1B-7B over direct TCP

The same run as [`../olmoe/2026-11-05.md`](../olmoe/2026-11-05.md), F32 payload, CPU, twice. `olmoe` does not build for Windows, so the expert ran in WSL2, with a small Python TCP forwarder on Windows relaying the Spark's connection in (outside the repo, `TCP_NODELAY`, two threads). The forwarder is in this path and adds some latency. Rank 0 holds 3.94 GB on `gx10-03`; rank 1 holds experts 32 to 63 on the desktop.

| Path | Router | Logits max err | Top-1 | Experts chosen | Time per prompt (5 / 21 / 17 tokens) |
|---|---|---|---|---|---|
| **Direct TCP, run 1** | `gx10-03` | 3.77e-5 | 43/43 | 5,504/5,504 | 2.73 / 5.07 / 3.96 s |
| **Direct TCP, run 2** | `gx10-03` | 3.77e-5 | 43/43 | 5,504/5,504 | 2.27 / 5.06 / 3.94 s |
| Relay (TASK-10) | desktop | 3.77e-5 | 43/43 | 5,504/5,504 | 6.94 / 31.7 / 24.3 s |

The link carried 39.7 MB each way per run. The longest prompt is now close to the one-process CPU time on the desktop (5.57 s), so the network no longer dominates. The 5-token prompt takes longer than its size suggests, and we did not look into why.

## Limits

- **The direction is fixed:** the Spark must dial the desktop. Connections the other way have no route.
- **The source address changes on the way:** TCP connections arrive from a private `192.168.13.x` address, while UDP arrives from the lab's public address. So a campus device rewrites or proxies TCP. It may also explain the close-time behavior seen on earlier paths, though we did not test that. Teardown here was clean in every run.
- **This is not an approved route:** the TCP path works, but no one in charge of the network has been asked about it. The engine's TCP transport has no authentication or encryption, so an expert should listen on a campus-facing address only during a run.
- **Cross-machine RDMA is unrelated:** the desktop has no RDMA hardware, so desktop-to-Spark stays TCP at no more than 1 Gb/s.
