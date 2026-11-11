#!/bin/bash
# Diagnostic: the f32 CPU run with fwd.py relaying to the WSL2 VM address (172.26.67.168) instead of localhost.
FWD_TO=172.26.67.168
source <(sed -n '/^out=/,/^}$/p' "$(dirname "$0")/run.sh")
run cpu-vmip t22-cpu
