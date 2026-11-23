#!/bin/bash
# TASK-19's final benchmarks on gx10-02, every block at one commit, built into zig-out-final.
# Runs on gx10-02 from anywhere: run.sh COMMIT [block ...]  (setup correct speed storage kv transport soak)
# Each block writes to out/BLOCK and a done marker, so a rerun skips finished blocks.
COMMIT=${1:?commit}
shift
BLOCKS=${*:-setup correct speed storage kv transport soak}
source ~/lukehinojosa/env.sh
L=~/lukehinojosa
W=$L/bench-final
S=$L/src-final
B=$L/zig-out-final/bin
O=$B/tfswarm
M=$L/models
OUT=$W/out
mkdir -p "$OUT"
START=$(date +%s)
DEADLINE=${DEADLINE:-$(( START + 23 * 3600 + 30 * 60 ))} # the soak ends here; collecting takes the rest
QD=$M/qwen3-30b-a3b
Q8=$QD/Qwen3-30B-A3B-Q8_0.gguf
GD=$M/glm53-flash
GLM=$GD/GLM-5.3-Flash-Q4_K-00002-of-00002.gguf
A=192.168.197.12

stamp()
{
    echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/progress.txt"
}

# run NAME ARGS...: one engine run, timed, its log in the current block's folder.
run()
{
    local name=$1
    shift
    /usr/bin/time -v "$@" > "$R/$name.log" 2>&1
    echo "exit $?" >> "$R/$name.log"
    echo "$name: $(grep -hE "time to first token|PASS|FAIL|error|recall|expert cache: [0-9]+ expert calls" "$R/$name.log" | grep -v "^expert" | cut -c1-220 | tail -n 3 | paste -sd ' ')" >> "$R/summary.txt"
}

setup()
{
    {
        echo "commit $COMMIT, built ReleaseFast with -Drdma=true -Dgpu=true"
        echo "$(hostname): $(uname -r), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), zig $(zig version)"
        nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit --format=csv
        lscpu | grep -E "Model name|^CPU\(s\)|MHz"
        free -g | head -2
        df -h ~ | tail -1
        nvme list 2>/dev/null | grep -m1 nvme
        cat /sys/block/nvme0n1/device/model 2>/dev/null
        ibv_devinfo 2>/dev/null | grep -E "hca_id|state|active_mtu|link_layer"
        python3 --version
    } > "$R/versions.txt" 2>&1
    (cd "$S" && zig build test -Doptimize=ReleaseFast -Drdma=true -Dgpu=true -During="$TFS_ROOT/tools" > "$R/unit-tests.log" 2>&1; echo "exit $?" >> "$R/unit-tests.log")
    "$O" gpu-formats > "$R/gpu-formats.log" 2>&1
    echo "exit $?" >> "$R/gpu-formats.log"
    # The drive: fio, then the engine's own io_uring reader on the same 16 GiB file.
    local f=$W/read.dat
    "$B/uring_bench" mk "$f" 16384 > "$R/mk.log" 2>&1
    sync
    for i in 1 2 3
    do
        fio --name=seq --filename="$f" --rw=read --bs=1M --iodepth=16 --direct=1 --ioengine=io_uring --runtime=30 --time_based --output-format=json > "$R/fio-seq-$i.json" 2>&1
        fio --name=rand --filename="$f" --rw=randread --bs=4k --iodepth=32 --direct=1 --ioengine=io_uring --runtime=30 --time_based --output-format=json > "$R/fio-rand4k-$i.json" 2>&1
        "$B/uring_bench" read "$f" --bs 1024 --qd 16 > "$R/uring-$i.log" 2>&1
    done
    rm -f "$f"
}

correct()
{
    bash "$S/bench/olmoe/2026-11-20-shared-parts/record.sh" "$O" compare "$R/record" > "$R/record.txt" 2>&1
    bash "$S/bench/olmoe/2026-11-20-shared-parts/check.sh" "$O" "$R/qwen2moe" > "$R/qwen2moe.txt" 2>&1
    bash "$S/bench/olmoe/2026-11-18-families/check.sh" "$O" "$R/families" > "$R/families.txt" 2>&1
    run olmoe-golden "$O" local $M/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden $M/olmoe_generate.tfref --gpu --generate 64
    run qwen3-golden "$O" local $Q8 --golden $QD/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32
    run qwen36-golden "$O" local $M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf --golden $M/qwen36-35b-a3b/qwen35_generate.tfref --gpu --generate 32
    run glm53-golden "$O" local $GLM --golden $GD/glm53_golden.tfref --gpu --cache 5000 --generate 16
}

# models as NAME:FILE:CONTEXTS, CONTEXTS separated by commas.
MODELS="olmoe:$M/olmoe-1b-7b-0924-instruct-q8_0.gguf:512,4000
qwen2moe:$M/qwen2moe-a2.7b/Qwen1.5-MoE-A2.7B-Chat-Q8_0.gguf:512,4096
qwen3:$Q8:512,4096,16384
qwen3-q4km:$M/qwen3-30b-a3b-gguf/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf:512,4096
qwen36:$M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf:512,4096,16384
mixtral:$M/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf:512,4096,16384
moonlight:$M/moonlight-16b/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf:512,4096
gptoss:$M/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf:512,4096,16384"

speed()
{
    # The same token ids for every family, so no model needs its own tokenizer here.
    python3 -c "
import struct
open('$W/synth.tokens', 'wb').write(struct.pack('<16384I', *[(i * 7919) % 30000 + 100 for i in range(16384)]))
"
    local T="--tokens $W/synth.tokens --generate 128 --chunk 512"
    local line name file ctxs c rep
    for line in $MODELS
    do
        name=${line%%:*}
        file=${line#*:}
        ctxs=${file##*:}
        file=${file%:*}
        for c in ${ctxs//,/ }
        do
            for rep in 1 2 3
            do
                run $name-gpu-$c-$rep "$O" local "$file" $T --context $c --gpu
            done
        done
        # Four ranks over shared memory, with each activation format, to show what the split costs.
        for p in fp8 f32
        do
            run $name-shm4-$p-512 "$O" shm "$file" $T --context 512 --ranks 4 --gpu --payload $p --max-step 512
        done
    done
    for rep in 1 2
    do
        run olmoe-cpu-512-$rep "$O" local $M/olmoe-1b-7b-0924-instruct-q8_0.gguf $T --context 512
        run qwen3-cpu-512-$rep "$O" local $Q8 $T --context 512
    done
    run glm53-gpu-512 "$O" local $GLM --tokens $GD/repo.tokens --context 512 --generate 64 --chunk 64 --gpu --cache 5000
}

storage()
{
    local line name file n
    # Each model's experts are loaded in halves, as one of two ranks would.
    for line in olmoe:$M/olmoe-1b-7b-0924-instruct-q8_0.gguf:64 qwen3:$Q8:128 qwen36:$M/qwen36-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf:256 mixtral:$M/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf:8 moonlight:$M/moonlight-16b/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf:64 gptoss:$M/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf:32 qwen2moe:$M/qwen2moe-a2.7b/Qwen1.5-MoE-A2.7B-Chat-Q8_0.gguf:60
    do
        name=${line%%:*}
        file=${line#*:}
        n=${file##*:}
        file=${file%:*}
        for i in 1 2
        do
            "$B/tfload" load "$file" --experts 0-$(( n / 2 - 1 )) > "$R/tfload-$name-lo-$i.log" 2>&1
            "$B/tfload" load "$file" --experts $(( n / 2 ))-$(( n - 1 )) > "$R/tfload-$name-hi-$i.log" 2>&1
        done
    done
    # GLM-5.3-Flash is larger than memory: hit rate and speed against the expert cache's size, with and without lookahead.
    local G="--tokens $GD/repo.tokens --context 512 --generate 64 --chunk 64 --gpu"
    for slots in 1000 2500 4000 5500
    do
        run glm53-cache$slots "$O" local $GLM $G --cache $slots
        run glm53-cache$slots-ahead "$O" local $GLM $G --cache $slots --lookahead 2 --extra 2
    done
    # Qwen3 streamed through a cache a quarter and half its experts' size, against all in memory in the speed block.
    local Q="--tokens $QD/repo.tokens --context 512 --generate 128 --chunk 512 --gpu"
    for slots in 1536 3072
    do
        run qwen3-cache$slots "$O" local $Q8 $Q --cache $slots
        run qwen3-cache$slots-ahead "$O" local $Q8 $Q --cache $slots --lookahead 2 --extra 2
    done
}

kv()
{
    local Q="--tokens $QD/repo.tokens --chunk 2048 --gpu --yarn 4"
    local f rep
    for f in f32 bf16 fp8
    do
        run golden-olmoe-$f "$O" local $M/olmoe-1b-7b-0924-instruct-q8_0.gguf --golden $M/olmoe_generate.tfref --gpu --generate 64 --kv-format $f
        run golden-qwen3-$f "$O" local $Q8 --golden $QD/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --kv-format $f
    done
    for rep in 1 2
    do
        run 32k-f32-$rep "$O" local $Q8 $Q --context 32640 --generate 128 --kv-format f32 --save $W/kv32.bin
        for f in bf16 fp8
        do
            run 32k-$f-$rep "$O" local $Q8 $Q --context 32640 --generate 128 --kv-format $f --compare $W/kv32.bin
        done
    done
    rm -f $W/kv32.bin
    # The KV middle on NVMe must give exactly the all-in-memory run.
    mkdir -p $W/kvtmp
    for f in f32 fp8
    do
        run tier64k-$f-ram "$O" local $Q8 $Q --context 65408 --generate 32 --kv-format $f --save $W/t.bin
        run tier64k-$f-nvme "$O" local $Q8 $Q --context 65408 --generate 32 --kv-format $f --kv-offload $W/kvtmp --kv-window 8192 --compare $W/t.bin
        rm -f $W/t.bin
    done
    rm -rf $W/kvtmp
    # A saved prefix restores to exactly a cold run.
    local P=$W/prefixes
    rm -rf $P
    mkdir -p $P
    run prefix-16k-cold "$O" local $Q8 $Q --context 16384 --generate 8 --save $W/p16.bin
    run prefix-12k-save "$O" local $Q8 $Q --context 12000 --generate 8 --prefix-cache $P
    run prefix-16k-partial "$O" local $Q8 $Q --context 16384 --generate 8 --prefix-cache $P --compare $W/p16.bin
    run prefix-16k-full "$O" local $Q8 $Q --context 16384 --generate 8 --prefix-cache $P --compare $W/p16.bin
    ls -l $P > "$R/prefix-files.txt"
    rm -rf $P $W/p16.bin
    run 128k-f32 "$O" local $Q8 $Q --context 131040 --generate 32 --kv-format f32 --save $W/kv128.bin
    run 128k-fp8 "$O" local $Q8 $Q --context 131040 --generate 32 --kv-format fp8 --compare $W/kv128.bin
    rm -f $W/kv128.bin
}

# pair NAME EXPERT_ARGS -- ROUTER_ARGS: one dispatch expert and router on gx10-02, on a fresh port.
pair()
{
    local name=$1 p=$(( 48300 + RANDOM % 200 )) e=() r=()
    shift
    while [ "$1" != "--" ]
    do
        e+=("$1")
        shift
    done
    shift
    r=("$@")
    "$B/dispatch" "${e[0]}" --port $p "${e[@]:1}" > "$R/$name-expert.log" 2>&1 &
    local ep=$!
    sleep 2
    "$B/dispatch" "${r[0]}" --port $p "${r[@]:1}" > "$R/$name.log" 2>&1
    echo "exit $?" >> "$R/$name.log"
    wait $ep
    echo "exit $?" >> "$R/$name-expert.log"
}

transport()
{
    local i k
    for i in 1 2 3 4 5
    do
        for k in "" --latency
        do
            "$B/dispatch" shm --packets 4096 --slots 64 $k > "$R/shm${k}-$i.log" 2>&1
            "$B/dispatch" rdma --dev rocep1s0f0 --packets 4096 --slots 64 $k > "$R/rdma${k}-$i.log" 2>&1
            pair rdma2${k}-$i rdma-expert --dev rocep1s0f0 --packets 4096 --slots 64 $k -- rdma-router --host $A --dev rocep1s0f0 --packets 4096 --slots 64 $k
            pair tcp${k}-$i expert --packets 4096 --slots 64 $k -- router --host 127.0.0.1 --packets 4096 --slots 64 $k
        done
    done
    # The engine split four ways over shared memory and over RDMA, against Qwen3's golden.
    for p in fp8 f32
    do
        run qwen3-shm4-$p "$O" shm $Q8 --golden $QD/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --ranks 4 --payload $p
        run qwen3-rdma4-$p "$O" rdma $Q8 --golden $QD/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --ranks 4 --payload $p --dev rocep1s0f0
    done
}

# Qwen3 served by four ranks over RDMA, chatting in a loop until the deadline, sampled every minute.
soak()
{
    local p=$(( 48600 + RANDOM % 100 )) left
    left=$(( DEADLINE - $(date +%s) ))
    if [ $left -lt 7200 ]
    then
        left=7200
    fi
    echo "soak for $left s from $(date -u +%FT%TZ)" > "$R/plan.txt"
    "$O" rdma-expert $Q8 --port $p --dev rocep1s0f0 --rank 1 --ranks 4 --gpu > "$R/r1.log" 2>&1 &
    local e1=$!
    "$O" rdma-expert $Q8 --port $((p + 1)) --dev rocep1s0f1 --rank 2 --ranks 4 --gpu > "$R/r2.log" 2>&1 &
    local e2=$!
    "$O" rdma-expert $Q8 --port $((p + 2)) --dev rocep1s0f0 --rank 3 --ranks 4 --gpu > "$R/r3.log" 2>&1 &
    local e3=$!
    for i in $(seq 60)
    do
        if [ "$(cat "$R"/r[123].log | grep -c listening)" -ge 3 ]
        then
            break
        fi
        sleep 1
    done
    "$O" rdma-router $Q8 --peer rdma:$A:$p@rocep1s0f0 --peer rdma:192.168.177.12:$((p + 1))@rocep1s0f1 --peer rdma:$A:$((p + 2))@rocep1s0f0 --gpu --serve 127.0.0.1:8093 --ctx 8192 > "$R/r0.log" 2>&1 &
    local r0=$!
    for i in $(seq 300)
    do
        if curl -s -o /dev/null http://127.0.0.1:8093/
        then
            break
        fi
        sleep 1
    done
    local end=$(( $(date +%s) + left ))
    (
        while [ $(date +%s) -lt $end ] && kill -0 $r0 2>/dev/null
        do
            echo "$(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) avail_mb $(free -m | awk 'NR==2{print $7}') gpu $(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits) rss_kb $(ps -o rss= -p $r0,$e1,$e2,$e3 | tr -s ' \n' ' ')" >> "$R/samples.txt"
            sleep 60
        done
    ) &
    local sp=$!
    local n=0
    while [ $(date +%s) -lt $end ] && kill -0 $r0 2>/dev/null
    do
        n=$((n + 1))
        echo "== chat $n $(date -u +%FT%TZ)" >> "$R/chats.txt"
        python3 "$S/tools/ref/record_chat.py" --base http://127.0.0.1:8093 --thinking --max-tokens 500 >> "$R/chats.txt" 2>&1
    done
    echo "chats $n, router alive $(kill -0 $r0 2>/dev/null && echo yes || echo no) at $(date -u +%FT%TZ)" >> "$R/plan.txt"
    kill -TERM $r0 2>/dev/null
    wait $r0
    sleep 3
    kill $e1 $e2 $e3 2>/dev/null
    wait $sp
}

stamp "start $COMMIT: $BLOCKS"
for b in $BLOCKS
do
    R=$OUT/$b
    if [ -f "$R/done" ]
    then
        stamp "skip $b (done)"
        continue
    fi
    mkdir -p "$R"
    stamp "begin $b"
    $b
    touch "$R/done"
    stamp "end $b"
done
stamp FINAL_DONE
