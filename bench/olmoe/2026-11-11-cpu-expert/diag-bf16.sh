#!/bin/bash
# Diagnostic: the CPU run with a BF16 payload (one packet per expert call instead of two).
source <(sed -n '/^out=/,/^}$/p' "$(dirname "$0")/run.sh" | sed 's/--payload ${PAYLOAD:-f32}/--payload bf16/')
run cpu-bf16 t22-cpu
