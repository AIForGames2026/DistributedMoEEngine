#!/bin/bash
# TASK-33: three processes on two machines over TCP. Rank 0 (router) and rank 1 on gx10-02 with the GPU;
# rank 2 on the desktop's CPU in WSL2, which gx10-02 dials on the campus path (TASK-21) through fwd.py.
# Run from Git Bash on the desktop with FWD_TO set to the WSL2 VM's address (`wsl hostname -I`).
out="$(cd "$(dirname "$0")" && pwd)"
wout="/mnt/$(echo "${out:1:1}" | tr A-Z a-z)${out:2}"
DESK=184.171.155.166
SPARK="ssh -o HostName=100.122.124.37 gx10-02"
E='source ~/lukehinojosa/env.sh; cd ~/lukehinojosa'
O=src/engine/zig-out-gen/bin/olmoe

$SPARK "echo \"\$(hostname): \$(uname -r), driver \$(nvidia-smi --query-gpu=driver_version --format=csv,noheader), \$($E; zig version), engine \$($E; strings $O | grep -m1 -o 'v0\.2\.0-alpha[^ ]*n33')\"" > "$out/tcp-versions.txt"
MSYS_NO_PATHCONV=1 wsl.exe bash -c "echo \"desktop WSL2: \$(uname -r), \$(lscpu | grep 'Model name' | sed 's/  */ /g'), \$(nproc) threads, engine \$(strings ~/tfs-scratch/n33-cpu/bin/olmoe | grep -m1 -o 'v0\.2\.0-alpha[^ ]*n33')\"" >> "$out/tcp-versions.txt"

# run NAME MODEL ROUTER-ARGS...: MODEL is relative to ~/lukehinojosa on gx10-02 and to ~ in WSL2.
run() {
    local name=$1 model=$2
    shift 2
    local port=$(( 7700 + RANDOM % 50 ))
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "cd ~ && ~/tfs-scratch/n33-cpu/bin/olmoe expert ~/$model --port $port --rank 2 --ranks 3 > '$wout/tcp-$name.rank2-desktop.log' 2>&1" &
    local ep=$!
    python "$out/../2026-11-11-cpu-expert/fwd.py" $((port + 50)) $port $FWD_TO > "$out/tcp-$name.fwd.log" 2>&1 &
    local fp=$!
    $SPARK "$E; nohup $O expert $model --port $((port + 100)) --rank 1 --ranks 3 --gpu > /tmp/lh_n33_r1.log 2>&1 < /dev/null &"
    sleep 5
    $SPARK "$E; $O router $model --peer 127.0.0.1:$((port + 100)) --peer $DESK:$((port + 50)) --payload f32 --gpu $* > /tmp/lh_n33_r0.log 2>&1; echo rc=\$? >> /tmp/lh_n33_r0.log; sleep 1"
    scp -q -o HostName=100.122.124.37 gx10-02:/tmp/lh_n33_r0.log "$out/tcp-$name.rank0-gx10-02.log"
    scp -q -o HostName=100.122.124.37 gx10-02:/tmp/lh_n33_r1.log "$out/tcp-$name.rank1-gx10-02.log"
    $SPARK 'rm -f /tmp/lh_n33_r0.log /tmp/lh_n33_r1.log'
    wait $ep
    kill $fp 2>/dev/null
    grep -hE "PASS|FAIL|tokens/s|expert FFNs|expert calls|rc=" "$out/tcp-$name".rank*.log
}

echo "== OLMoE: forward and 32 generated tokens per prompt"
run olmoe models/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden models/olmoe_generate.tfref --generate 32
echo "== Qwen3-30B-A3B: forward"
run qwen3 models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf --golden models/qwen3-30b-a3b/qwen3_forward.tfref
echo "== Qwen3-30B-A3B with YaRN: 32 generated tokens per prompt"
run qwen3-yarn models/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf --golden models/qwen3-30b-a3b/qwen3_yarn_generate.tfref --yarn 4 --generate 32
echo TCP_DONE
