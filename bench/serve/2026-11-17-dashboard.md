# Operator dashboard, first checks (2026-11-17)

TASK-25: `olmoe dashboard` (`src/dashboard.zig`, page in `web/dashboard.html` and `web/dashboard.js`) and its node agent `olmoe node` (`src/node_agent.zig`), with operator sessions and the signed node link in `src/auth.zig` and child processes in `src/proc.zig`.

All of this ran on the desktop (i7-12700K, WSL2 Ubuntu, CPU only), with the dashboard and two agents on the same machine, each agent in its own directory under its own name (`node-a`, `node-b`). The Sparks were not reachable from the desktop that afternoon (they had dropped out of its Tailscale network), so the run across machines is still to do. Build: commit `ac58b75` plus this change, `zig build olmoe` (Debug), no RDMA or GPU.

Model: OLMoE-1B-7B-0924-Instruct Q8_0, three processes over TCP (rank 0 is the router started by the dashboard, ranks 1 and 2 by the agents), `--payload f32`.

## Operator flow, in the Claude browser pane

1. Sign in with the token from `dashboard.token` (made on the first run, mode 0600).
2. Both agents are listed as waiting for approval, with their memory and models; approve them. Approvals are saved to `dashboard.nodes` and survived restarts of the dashboard.
3. Tick both under Model, set the payload to f32 and turn off prefix reuse, Load. The dashboard started rank 1 on `node-b` and rank 2 on `node-a` through their agents, waited for both to listen, started the router, and reported "loaded on 3 processes" about 4 s later. The live diagram (the router's own page, passed through at `/live`) showed in the dashboard.
4. A test request from the page through the dashboard's `/v1`: 153 tokens streamed, first token 17 s, 1.3 tokens/s. That speed comes from a Debug build on the desktop CPU, split three ways over TCP with every process using every core, not from the dashboard: the router's own log gives the same times.
5. Stop: the router and both ranks ended, and no engine process was left.

From the shell, with the token as a Bearer header:

- Reload with different nodes while a model runs: the old processes stopped and the new set came up.
- Removing a node that the loaded model uses: 409.
- An unknown node or a model name such as `../../etc/passwd.gguf`: 400 with the reason, nothing started.
- Killing rank 2 (`kill -9`) while loaded: within 1 s the run was marked failed ("the rank on node node-a exited"), and the router and rank 1 were stopped.

## Authentication

`tools/ref/check_dashboard.py` against the running dashboard, 16 of 16 checks passed:

- `/api/state`, `/live`, `/telemetry` and every control POST (`/api/load`, `/api/stop`, `/api/nodes/approve`, `/api/nodes/forget`, `/api/logout`) answer 401 with no credentials, with a wrong token, and with a forged session cookie.
- A wrong token cannot sign in. A cross-site POST is refused (403) even with the right token.
- The right token signs in; the cookie is HttpOnly and SameSite=Strict, reads the state, and stops working after sign-out. The token also works as a Bearer header.
- A node agent with the wrong secret cannot verify the dashboard's proof, and the dashboard refuses its proof. With the right secret the agent is welcomed, and a frame signed with another key drops the link.

An `olmoe node` started with a wrong secret file also refused to link ("could not prove it holds the node secret"). `zig build test` passes 114 tests, including the new ones for sessions, cookies, secret files, the handshake (one secret and two), forged, replayed and reflected frames, and child processes.

## Same tokens as olmoe directly

With the same three-process load (f32 payload, prefix reuse off), `tools/ref/check_serve.py requests --base http://127.0.0.1:8095/v1 --max-tokens 32` sent its 4 conversations greedily through the dashboard's `/v1` with curl and the `openai` package, streamed and not: all four ways gave the same tokens for every conversation. `check_serve.py compare` then fed each saved prompt to `olmoe local` on the same machine: 4 of 4 conversations generated the same 32 tokens. So the dashboard passes requests through unchanged, and the split it starts gives exactly what `olmoe` gives in one process.

## Across machines (14:36 to 14:57)

The dashboard and router on `gx10-02` (ReleaseFast build of the same tree), and `olmoe node` on the desktop (WSL, ReleaseFast, CPU) as the second machine, over Tailscale (35 ms round trip). WSL sits behind NAT, so the agent was started with `--advertise 127.0.0.1` and an SSH tunnel carried `gx10-02`'s port 47401 back to the desktop's rank; the agent's own link went straight to the dashboard on port 47310. The page and API reached the operator's browser through the same tunnel.

- **Operator flow, from the page:** sign in, approve `desktop-wsl` (it linked from the desktop's Tailscale address), tick it, Load with f32 payload and no prefix reuse: loaded on 2 processes (router on `gx10-02` holding experts 0 to 31, rank 1 on the desktop holding 32 to 63) within a few seconds, live diagram shown. A chat from the page streamed 83 tokens, first token 22.5 s, 0.6 tokens/s; every token crosses the 35 ms link once per layer (16 layers), so this is the link, not the engine.
- **Same tokens as olmoe directly:** `check_serve.py requests --max-tokens 16` through the dashboard's `/v1`: curl and `openai`, streamed and not, agreed on 4 of 4 conversations, and `check_serve.py compare` with `olmoe local` on `gx10-02` gave the same 16 tokens for all 4.
- **Authentication:** `check_dashboard.py` against `gx10-02`'s dashboard, 16 of 16 passed.
- **Failure:** `kill -9` of the desktop's rank at 14:56:17.71; the dashboard marked the run failed at 14:56:18.01 ("the rank on node desktop-wsl exited with 137"), 0.3 s later, and stopped the router.
- Afterwards no engine process and none of the ports (8095, 18095, 47310, 47401) were left on `gx10-02` or the desktop.

RDMA through the dashboard is not tested yet: the desktop has no RDMA device, and the other Sparks belong to other lab members.

## Limits

- Limits found so far: sessions live in memory, so restarting the dashboard signs everyone out (approvals are kept); an expert rank still accepts the first connection to its port, as before; and the router's local port (127.0.0.1) skips the API key for other users of the same machine.
