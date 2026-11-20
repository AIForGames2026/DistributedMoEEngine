# Native Windows dashboard and node agent (2026-11-20)

TASK-43. `olmoe.exe dashboard` and `olmoe.exe node` built and run on Windows itself. A Windows PC can host the operator dashboard, and join one as an expert node, without WSL2, `--advertise` or an SSH tunnel. It follows TASK-40, which brought the engine's other modes to Windows.

## Machine

Windows 11 Home 10.0.26200, i7-12700K, RTX 5070 Ti (driver 617.42), Zig 0.17.0. `olmoe.exe` was built ReleaseFast with `-Dgpu=true`, and OLMoE-1B-7B Q8_0 read from `E:\models\olmoe` (a hard drive).

## What was built

- **Engine processes** (`src/win_child.zig`, `proc.Child` on Windows):
  - `CreateProcessW` sends output to the log file and reads input from `NUL`.
  - Only those two handles are inherited, so a rank never holds the dashboard's sockets.
  - Each process runs in a job object that ends it, and anything it starts, when stopped or when the dashboard or agent exits. This stands in for Linux's process group and parent-death signal.
  - A stop ends the job with exit code 143, what SIGTERM gives on Linux. The engine sets no SIGTERM handler, so it is the same end.
  - Arguments are quoted as `CommandLineToArgvW` splits them back; a unit test covers spaces, quotes and backslashes.
- **Machine facts** (`proc.zig`):
  - The models folder is listed with `FindFirstFileW`.
  - Memory comes from `GlobalMemoryStatusEx`.
  - The GPU name is the first NVIDIA display adapter.
  - No RDMA devices are reported.
- **Secrets** (`auth.zig`, the user's choice): a token or secret file must be inside the user's profile folder. That folder's default access list admits only the user, SYSTEM and Administrators.
  - The folder is checked before a new secret is written, so none is ever left outside it.
  - Short 8.3 names (`C:\Users\LUKEHI~1`) are expanded before comparing.
  - This is weaker than Linux's `chmod 600` rule: a file whose access list was widened by hand is still accepted.
- **Shared helpers:**
  - `sock.zig` gains a two-way socket pair, accepting with the peer's address, and waiting on one socket.
  - `plat.zig` gains a mutex, random bytes (`RtlGenRandom`), full paths with long names, folders, file replacement, the program's own path (the default `--engine`), and wall-clock milliseconds.
  - The dashboard, the agent and `auth.zig` use these in place of their libc calls; on Linux they make the same calls as before.
- **Log files:** `win32.openFile` now lets others keep writing a file it opens. The agent watches a rank's log for "listening on port" while the rank writes it. With read-only sharing, Windows refused the open, so the first load timed out after 60 s.

## Checks

| Check | Result |
|---|---|
| Unit tests, Windows | 131 pass, none skipped (auth, proc and `win_child` now run here; the signed-link tests use loopback socket pairs) |
| A token file in the repo folder (outside the profile) | refused, and no file written |
| Dashboard and agent start | the agent passes the signed handshake; its report shows 65,278 MB of memory, "GeForce RTX 5070 Ti" and the model |
| Sign in, approve, restart both | the approval is kept in the nodes file across the restart |
| Load OLMoE on 2 processes with `--gpu`, f32 payloads | running: the router from the dashboard, rank 1 from the agent |
| `check_serve.py requests` through the dashboard's `/v1`, without prefix reuse | PASS: curl and `openai`, streamed and not, agree on all 4 conversations, about 59 tokens/s |
| Stop from the dashboard | the router and rank 1 end; no engine process is left |
| Agent killed while its rank runs | the rank ends within 2 s; the dashboard reports "node win-agent went offline, so its rank is gone" |
| Dashboard page | served; the sign-in page loads |

## Linux unchanged

- **WSL2:** 135 tests pass. A script ran the same flow: dashboard and agent, sign in, approve, load OLMoE on 2 processes, a chat through `/v1`, then stop. The load ran, the chat answered, and no engine process was left.
- **`gx10-02`:** with `-Drdma=true -Dgpu=true`, 135 tests pass and `olmoe` builds.
