#!/bin/bash
# Two follow-ups after fixup.sh, on gx10-02: which commit changed the compact KV goldens, and why the loader reads below fio.
# Runs on gx10-02: followup.sh  (waits for FIXUP_DONE; old/src-COMMIT.tar.gz are the older sources; logs to out/followup)
source ~/lukehinojosa/env.sh
L=~/lukehinojosa
W=$L/bench-final
M=$L/models
QD=$M/qwen3-30b-a3b
Q8=$QD/Qwen3-30B-A3B-Q8_0.gguf
OM=$M/olmoe-1b-7b-0924-instruct-q8_0.gguf
R=$W/out/followup
while ! grep -q FIXUP_DONE "$W/out/progress.txt"
do
    sleep 60
done
mkdir -p "$R/kv" "$R/loader"
echo "$(date -u +%FT%TZ) begin followup" >> "$W/out/progress.txt"

# build COMMIT STEP: unpacks and builds one older commit into old/out-COMMIT.
build()
{
    local c=$1 step=$2
    cd "$W/old" || return 1
    rm -rf src-$c
    tar xzf src-$c.tar.gz
    cd src-$c || return 1
    zig build $step -Doptimize=ReleaseFast -Dgpu=true -During="$TFS_ROOT/tools" --prefix "$W/old/out-$c" > "$R/build-$c.log" 2>&1
    echo "exit $?" >> "$R/build-$c.log"
}

# Each commit from TASK-27 (Nov 16) to now, on the two golden runs that moved: OLMoE bf16 and Qwen3 fp8.
for c in b05d5fb 21172ab 14cf7b8 5449a56 009fc67 a3cc0a2 9d05193 f3873c4
do
    if [ $c = f3873c4 ]
    then
        O=$L/zig-out-final/bin/tfswarm
    elif [ $c = 9d05193 ]
    then
        build $c tfswarm
        O=$W/old/out-$c/bin/tfswarm
    else
        build $c olmoe
        O=$W/old/out-$c/bin/olmoe
    fi
    for f in bf16 fp8
    do
        "$O" local $OM --golden $M/olmoe_generate.tfref --gpu --generate 64 --kv-format $f > "$R/kv/$c-olmoe-$f.log" 2>&1
        "$O" local $Q8 --golden $QD/qwen3_yarn_generate.tfref --gpu --yarn 4 --generate 32 --kv-format $f > "$R/kv/$c-qwen3-$f.log" 2>&1
        echo "$c olmoe-$f: $(grep -ohE "[0-9]+/[0-9]+ tokens" "$R/kv/$c-olmoe-$f.log" | paste -sd ' ')   qwen3-$f: $(grep -ohE "[0-9]+/[0-9]+ tokens" "$R/kv/$c-qwen3-$f.log" | paste -sd ' ')" >> "$R/kv/summary.txt"
    done
done

# The loader: the Nov 5 build against today's, then the drive itself on the model files rather than a fresh file.
build 9e39d09 tfload
for i in 1 2 3
do
    for b in old/out-9e39d09/bin/tfload ../zig-out-final/bin/tfload
    do
        n=$(echo $b | cut -d/ -f2)
        "$W/$b" load $OM --experts 0-31 > "$R/loader/tfload-$n-olmoe-$i.log" 2>&1
        "$W/$b" load $Q8 --experts 0-63 > "$R/loader/tfload-$n-qwen3-$i.log" 2>&1
    done
done
U=$L/zig-out-final/bin/uring_bench
for f in $OM $Q8
do
    n=$(basename $f .gguf)
    filefrag $f > "$R/loader/filefrag-$n.txt" 2>&1
    for qd in 16 32 64
    do
        $U read $f --bs 1024 --qd $qd > "$R/loader/uring-$n-qd$qd.log" 2>&1
    done
    $U read $f --bs 4096 --qd 16 > "$R/loader/uring-$n-bs4m.log" 2>&1
    fio --name=seq --filename=$f --rw=read --bs=1M --iodepth=16 --direct=1 --ioengine=io_uring --readonly --runtime=20 --time_based --output-format=json > "$R/loader/fio-$n.json" 2>&1
done
echo "$(date -u +%FT%TZ) FOLLOWUP_DONE" >> "$W/out/progress.txt"
