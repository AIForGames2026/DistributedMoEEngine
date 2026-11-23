#!/bin/bash
# The loader checks followup.sh could not run: the Nov 5 build needs no -Dgpu, and uring_bench needs a file sized to its window.
# Runs on gx10-02 after followup.sh; logs go to out/followup/loader.
source ~/lukehinojosa/env.sh
W=~/lukehinojosa/bench-final
R=$W/out/followup/loader
M=~/lukehinojosa/models
cd $W/old/src-9e39d09 || exit 1
zig build tfload -Doptimize=ReleaseFast -During=$TFS_ROOT/tools --prefix ../out-9e39d09 > $W/out/followup/build-9e39d09b.log 2>&1
for i in 1 2 3
do
    $W/old/out-9e39d09/bin/tfload load $M/olmoe-1b-7b-0924-instruct-q8_0.gguf --experts 0-31 > $R/tfload-out-9e39d09-olmoe-$i.log 2>&1
    $W/old/out-9e39d09/bin/tfload load $M/qwen3-30b-a3b/Qwen3-30B-A3B-Q8_0.gguf --experts 0-63 > $R/tfload-out-9e39d09-qwen3-$i.log 2>&1
done
U=~/lukehinojosa/zig-out-final/bin/uring_bench
f=$W/read.dat
$U mk $f 16384 > /dev/null 2>&1
for a in "--qd 16" "--qd 32" "--qd 64" "--qd 16 --bs 4096" "--qd 8"
do
    n=$(echo $a | tr -d " -")
    $U read $f --bs 1024 $a > $R/uring-$n.log 2>&1
done
rm -f $f
