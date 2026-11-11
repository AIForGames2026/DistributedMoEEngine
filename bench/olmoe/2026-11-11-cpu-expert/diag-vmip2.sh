#!/bin/bash
# The repeat and the GPU comparison with fwd.py relaying to the WSL2 VM address.
FWD_TO=172.26.67.168
source <(sed -n '/^out=/,/^}$/p' "$(dirname "$0")/run.sh")
run cpu-vmip2 t22-cpu
run gpu-vmip t22-gpu --gpu
