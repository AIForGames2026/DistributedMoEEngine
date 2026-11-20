# Windows tray launcher and settings in the dashboard (2026-11-20)

TASK-44. `tfswarm-operator.exe` runs the operator dashboard, or a node agent, on Windows with no terminal. It follows TASK-43, which brought the dashboard and agent to Windows but still started them from PowerShell with command-line options.

## Machine

Windows 11 Home 10.0.26200, i7-12700K, RTX 5070 Ti (driver 617.42), Zig 0.17.0. `tfswarm.exe` was built ReleaseFast with `-Dgpu=true`, and OLMoE-1B-7B Q8_0 read from `E:\models\olmoe` (a hard drive). The tests used `TFSWARM_HOME` folders under `%TEMP%`, so the real `%LOCALAPPDATA%\TFSwarm` was not touched.

## What was built

- **Launcher** (`src/launcher.zig`, `res/windows/`):
  - A Windows-subsystem program with no console window and no libc, with an icon and a manifest. The engine was renamed from `olmoe` to `tfswarm` in the same change, and the launcher is `tfswarm-operator.exe`.
  - It runs `tfswarm setup`, `tfswarm dashboard --settings` or `tfswarm node --settings` from `tfswarm.exe` beside it. Each runs in a job object (`win_child.zig`), so quitting, or the launcher dying, ends it and every engine process under it.
  - Clicking the tray icon opens the dashboard. The menu copies the admin token, opens Settings or the logs folder, restarts, runs setup again, and quits. The same commands are accepted as posted `WM_COMMAND` messages, which the tests used.
  - One copy runs per settings folder. A second launch asks the first to open the dashboard and exits.
  - It watches its child: exit 75 from the dashboard means "start me again"; any other exit shows a notification pointing at the log. For a node it reads the agent's log to show the link state in the tooltip.
- **Settings** (`src/settings.zig`): a JSON file in `%LOCALAPPDATA%\TFSwarm` (`~/.config/tfswarm` on Linux, or `TFSWARM_HOME`). The token, node secret, API key, approved nodes and logs sit beside it. `--settings FILE` gives the dashboard and agent their options from it, with later options still overriding.
- **Setup page** (`tfswarm setup`, `src/setup.zig`, `web/setup.*`):
  - It serves only on 127.0.0.1, on a free port, and prints a one-time link. Every other request needs the cookie that link sets.
  - It checks the models folder, including `.gguf` files that have no `.tfsum` index yet.
  - For a node, "Test the link" runs the signed handshake with the pasted secret before saving. The secret is written with `auth.saveSecret`: owner-only on Linux, and only inside the profile folder on Windows.
- **Dashboard** (`src/dashboard.zig`, `web/dashboard.*`):
  - `POST /api/login-link` hands a program holding the admin token a one-time link; a request carrying a browser `Origin` is refused. `GET /signin/CODE` signs in once within 60 s.
  - A Settings view shows the admin token, node secret and API key to copy, and edits the settings file. The models folder and the API key apply at once. Who can open the page and the ports apply after a restart, which the page asks for (exit 75).

## Checks

| Check | Result |
|---|---|
| Unit tests, Windows | 132 pass (the settings file round trip is new) |
| Launcher with an empty settings folder | opened the setup page; `tfswarm.exe` and the launcher run with no window |
| Setup, "Host the dashboard" | settings saved, setup exited, the dashboard started, and the launcher signed in through a one-time link |
| Tray: copy the admin token | the clipboard held the token |
| Tray: Settings | a second one-time sign-in, landing on the Settings view |
| Second launch | exited at once and opened the dashboard of the first |
| Tray: restart | a new dashboard process, opened signed in again |
| Settings page: node port changed, API key on | saved to the file; `/v1` needed the key at once; the restart bar appeared |
| Settings page: restart | the dashboard exited with 75; the launcher started it on node port 47311 and signed in again |
| Sign-in links | a used or made-up link went to `/#link-expired`; a wrong token, or a correct one sent with an `Origin` header, got 401 |
| Setup page errors, as a node | the right messages for a wrong secret, an unreachable address and a short secret |
| Second launcher set up as a node | "Test the link" passed; the agent linked as `win-node` |
| Approve, then load OLMoE on 2 processes with `--gpu`, f32 payloads | running after about 30 s; rank 1 held by the launcher's agent |
| `check_serve.py requests` through the dashboard's `/v1`, without prefix reuse | PASS: curl and `openai`, streamed and not, agree on all 4 conversations, 56 to 61 tokens/s |
| Quit the node launcher | its rank ended; the dashboard reported "node win-node went offline, so its rank is gone" |
| Quit the dashboard launcher | no `tfswarm.exe` or launcher left |

The node's secret was sent to the setup page's save request from PowerShell, not pasted into the page. The browser pane used for the test cannot paste from the system clipboard; the page's own fields and errors were checked in it.

## After the rename

The same change renamed the engine from `olmoe` to `tfswarm`, the launcher to `tfswarm-operator.exe`, the PoC skeleton to `tfswarm-poc`, and the settings folder to `TFSwarm`. Afterwards, on Windows, the launcher ran setup, started the dashboard and signed in under the new names, and quit with nothing left. `check_serve.py requests` passed against `tfswarm local --serve`, whose `/v1` responses now carry a `tfswarm` object and `owned_by: tfswarm`. 132 tests pass on Windows, and 136 in WSL2 and on `gx10-02`, where the WSL2 setup script passed again.

## Linux

- **WSL2:** 136 tests pass. A script ran `tfswarm setup` (403 without its cookie, the models check, the save; the settings file is mode 600). It then started the dashboard and an agent from the settings, used a one-time link once and was refused the second time, and saw the dashboard exit with 75 on a restart. No process was left.
- **`gx10-02`:** with `-Drdma=true -Dgpu=true`, 136 tests pass and `tfswarm` builds.

## Not covered

- **Model indexes:** a model's `.tfsum` index is still made with `tfload index` on Linux. The setup page reports `.gguf` files that lack one.
- **Out of scope:** starting at sign-in and an installer.
