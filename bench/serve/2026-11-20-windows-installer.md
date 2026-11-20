# Windows installer, updates and uninstall (2026-11-20)

TASK-45. `TFSwarm-Setup-VERSION.exe` installs the tray launcher from TASK-44 with a CPU or a CUDA engine, updates it from GitHub releases, and removes it again.

## Machine

Windows 11 Home 10.0.26200, i7-12700K, RTX 5070 Ti (driver 617.42), Zig 0.17.0, CUDA 13.1, Inno Setup 7.1.1 (installed for this user). The installers were built by `.\build.ps1 -Installer`, ReleaseFast. The tests installed into a scratch folder with `/DIR`, used `TFSWARM_HOME` folders under `%TEMP%`, and uninstalled at the end.

## What was built

- **Installer** (`installer/tfswarm.iss`):
  - It installs for the current user into `%LOCALAPPDATA%\Programs\TFSwarm` with no admin rights, or for everyone into Program Files, as the user picks on its first page.
  - Its engine page installs the CPU or the CUDA build of `tfswarm.exe` (2.8 MB and 12.5 MB; the installer is 4.6 MB). CUDA is picked when `nvcuda.dll` is present, choosing it without a driver asks first, and an upgrade keeps the last choice. `/ENGINE=cpu` or `/ENGINE=cuda` chooses from the command line.
  - It adds a Start menu entry, and on request a desktop shortcut and a start at sign-in. Its last page starts the launcher, whose setup page asks whether the PC hosts the dashboard or joins one.
  - Before installing or uninstalling it asks every running launcher to quit, which ends its dashboard or node agent.
  - Uninstalling asks whether to delete `%LOCALAPPDATA%\TFSwarm` (settings, token, node secret, logs); `/DELETESETTINGS` does so quietly.
- **Updates** (`src/update.zig`, `src/launcher.zig`):
  - The launcher reads GitHub's newest release through WinHTTP (the system's proxy settings apply) a minute after it starts and then daily, unless its menu has turned this off (`check_updates` in the settings file). "Check for updates" asks at once.
  - A newer release (by version number; a release beats a pre-release of the same number) shows a notification and an "Update to" item. That asks before downloading, checks the installer's size and SHA-256 against the digest GitHub publishes, stops the dashboard or agent, and runs the installer with `/SILENT /RELAUNCH`, which starts the launcher again when done.
  - `launcher.log` in the settings folder records what the icon said, keeping the end of the previous launch's log.
  - The version comes from `-Dversion`, or `git describe` of the latest `v` tag.
- **GPU default:** the dashboard reports whether its engine has CUDA. On the CUDA engine the Load form starts with GPU ticked, and Settings has "Load models on the GPU by default"; on the CPU engine both are greyed out.
- **Release workflow** (`.github/workflows/release.yml`): a `v*` tag builds both engines and the installer on a Windows runner (CUDA 13.1 from `Jimver/cuda-toolkit`, Inno Setup from the image or Chocolatey) and attaches the installer to that tag's release.

## Checks

| Check | Result |
|---|---|
| Unit tests, Windows | 134 pass (version comparison and release parsing are new) |
| Wizard pages | per user or everyone; folder; engine (CUDA picked, this PC has a driver); shortcuts; ready page naming the engine; cancelling left nothing installed |
| Quiet install, `/ENGINE=cpu`, version 0.3.9 | the CPU build's `tfswarm.exe`, byte for byte; Start menu entry; uninstall entry with version 0.3.9 and engine `cpu` |
| Update from a local feed with a wrong digest | the launcher asked, downloaded, refused with "InstallerDigestMismatch. Nothing was changed.", and restarted the node agent; no file written |
| Update from a local feed offering 0.4.0 | asked, downloaded, ran the installer quietly; the launcher came back as 0.4.0 with the node agent running, the CPU engine kept, and the uninstall entry at 0.4.0. The log showed both launches |
| Installer run by hand over a running launcher, `/ENGINE=cuda` | closed the launcher, put the CUDA build in place, and started nothing |
| Installed CUDA engine | `tfswarm gpu-formats` PASS |
| Uninstall with a launcher running, answering No to deleting settings | launcher closed; files, shortcut and uninstall entry gone; settings kept |
| Uninstall cancelled at its first question | launcher still running, files in place |
| Quiet uninstall with `/DELETESETTINGS` | launcher closed, everything removed including the settings folder |
| Check against GitHub (no release yet) | "No release yet" |
| Menu: check automatically, toggled | `check_updates` false in the settings file |
| CUDA build's dashboard | `gpu_build` and `gpu_default` true; Load form ticked; saving Settings with GPU off made the default false |
| CPU build's dashboard | `gpu_build` false; GPU greyed out on the Load form and in Settings |

The engine choice and the update prompt were answered with keystrokes sent to the installer's and launcher's windows, and checked from screenshots of them.

## Linux

- **WSL2:** 138 tests pass and `tfswarm` builds.
- **`gx10-02`:** with `-Drdma=true -Dgpu=true`, 138 tests pass and `tfswarm` builds.

## Not covered

- **Release workflow:** runs only when a `v*` tag is pushed, so it has not run yet. Pushing a tag such as `v0.4.0` builds the first installer release, which installed launchers will then offer.
- **Install for everyone:** needs an admin prompt, so it was not run here; the per-user install was.
- **Code signing:** the installer is unsigned, so SmartScreen warns the first time it runs.
- **Out of scope:** firewall rules (Windows asks the first time the dashboard or a node listens), and a Linux package.
