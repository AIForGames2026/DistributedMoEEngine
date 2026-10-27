#!/usr/bin/env python3
"""TASK-05 platform probe. Writes one JSON report per machine that answers:

  1. compute capability            -> gpu.compute_capability
  2. does tcgen05 / TMEM assemble  -> tcgen05
  3. ConnectX-7 state, RDMA loopback -> rdma
  4. hardware Decompression Engine -> gpu.mem_decompress_algorithm_mask
  5. memory and NVMe bandwidth, free memory -> memory, nvme

Usage (from the repo root):
  python3 tools/probe/probe.py [--name NAME] [--out DIR]
                               [--nvme-bench DIR] [--rdma-loopback]

--nvme-bench DIR  lays out a 1 GiB file in DIR with fio, reads it with
                  O_DIRECT (256 KiB, QD 16), and deletes it.
--rdma-loopback   runs ib_write_lat (7,552 bytes) in loopback on the first
                  ACTIVE RoCE v2 port.
Reads only; nothing on the system is changed except the temporary files.
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = os.environ.get("CUDA_HOME", "/usr/local/cuda")


def run(cmd, timeout=600, **kw):
    """Runs a command; returns (exit code, stdout + stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def meminfo():
    out = {}
    for line in (read("/proc/meminfo", "") or "").splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0]) * 1024
    return out


def host(name):
    mi = meminfo()
    os_name = re.search(r'PRETTY_NAME="([^"]*)"', read("/etc/os-release", "") or "")
    cpu = re.search(r"Model name:\s*(.+)", run(["lscpu"])[1])
    return {
        "name": name,
        "os": os_name.group(1) if os_name else None,
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_model": cpu.group(1).strip() if cpu else None,
        "cpus": os.cpu_count(),
        "mem_total_bytes": mi.get("MemTotal"),
        "mem_available_bytes": mi.get("MemAvailable"),
        "swap_used_bytes": mi.get("SwapTotal", 0) - mi.get("SwapFree", 0),
        "load_average": os.getloadavg(),
        "wsl": "microsoft" in platform.release().lower(),
    }


def compile_c(src, out, extra):
    cc = os.environ.get("CC", "cc")
    code, log = run([cc, "-std=c11", "-O2", "-o", out, src] + extra)
    return code == 0, log


def gpu(tmp):
    exe = os.path.join(tmp, "cuda_probe")
    libs = ["-L" + d for d in (CUDA + "/lib64", CUDA + "/lib64/stubs", "/usr/lib/wsl/lib") if os.path.isdir(d)]
    ok, log = compile_c(os.path.join(HERE, "cuda_probe.c"), exe, ["-I" + CUDA + "/include"] + libs + ["-lcuda"])
    if not ok:
        return {"error": "build failed: " + log[-500:]}
    code, out = run([exe])
    if code != 0:
        return {"error": out[-500:]}
    info = json.loads(out)
    smi = run(["nvidia-smi", "--query-gpu=driver_version,utilization.gpu", "--format=csv,noheader,nounits"])[1]
    if "," in smi:
        driver, util = [s.strip() for s in smi.splitlines()[0].split(",")]
        info["driver_version"] = driver
        info["utilization_percent"] = int(util) if util.isdigit() else None
    return info


def tcgen05(cc):
    ptxas = shutil.which("ptxas") or CUDA + "/bin/ptxas"
    if not os.path.exists(ptxas):
        return {"error": "ptxas not found"}
    with open(os.path.join(HERE, "tcgen05.ptx")) as f:
        ptx = f.read()
    targets = ["sm_100a"]  # control: datacenter Blackwell has TMEM
    if cc:
        targets.append("sm_" + cc.replace(".", "") + "a")
    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        for t in targets:
            path = os.path.join(tmp, t + ".ptx")
            with open(path, "w") as f:
                f.write(re.sub(r"^\.target .*$", ".target " + t, ptx, flags=re.M))
            code, log = run([ptxas, "-arch=" + t, path, "-o", os.devnull])
            result[t] = {"assembles": code == 0, "message": log.splitlines()[0] if log else ""}
    return result


def rdma(loopback):
    ports = []
    for port_dir in sorted(glob.glob("/sys/class/infiniband/*/ports/*")):
        dev = port_dir.split("/")[4]
        gids = []
        for type_path in sorted(glob.glob(port_dir + "/gid_attrs/types/*")):
            idx = type_path.rsplit("/", 1)[1]
            gid = read(port_dir + "/gids/" + idx, "")
            kind = read(type_path, "")
            if kind and gid and gid != "0000:0000:0000:0000:0000:0000:0000:0000":
                gids.append({"index": int(idx), "type": kind, "ipv4": gid.startswith("0000:0000:0000:0000:0000:ffff:")})
        ports.append({
            "device": dev,
            "port": int(port_dir.rsplit("/", 1)[1]),
            "state": read(port_dir + "/state"),
            "rate": read(port_dir + "/rate"),
            "link_layer": read(port_dir + "/link_layer"),
            "roce_v2_ipv4_gid_index": next((g["index"] for g in gids if g["type"] == "RoCE v2" and g["ipv4"]), None),
        })
    out = {"ports": ports}
    if loopback:
        out["loopback"] = rdma_loopback(ports)
    return out


def rdma_loopback(ports):
    port = next((p for p in ports if "ACTIVE" in (p["state"] or "") and p["roce_v2_ipv4_gid_index"] is not None), None)
    if not port or not shutil.which("ib_write_lat"):
        return {"error": "no ACTIVE RoCE v2 port or ib_write_lat missing"}
    args = ["ib_write_lat", "-d", port["device"], "-i", str(port["port"]), "-x", str(port["roce_v2_ipv4_gid_index"]),
            "-s", "7552", "-n", "5000", "-F"]
    server = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    code, out = run(args + ["localhost"], timeout=120)
    server.wait(timeout=30)
    # Columns: bytes iterations t_min t_max t_typical t_avg t_stdev 99% 99.9%
    row = next((l.split() for l in out.splitlines() if l.split()[:1] == ["7552"]), None)
    if code != 0 or not row:
        return {"error": out[-500:]}
    return {"device": port["device"], "bytes": 7552, "t_typical_us": float(row[4]), "t_avg_us": float(row[5]),
            "p99_us": float(row[7]), "p999_us": float(row[8])}


def nvme(bench_dir):
    drives = []
    for d in sorted(glob.glob("/sys/class/nvme/nvme*")):
        dev = d + "/device"
        drives.append({
            "model": read(d + "/model"),
            "firmware": read(d + "/firmware_rev"),
            "link": read(dev + "/current_link_speed"),
            "width": read(dev + "/current_link_width"),
            "max_link": read(dev + "/max_link_speed"),
            "max_width": read(dev + "/max_link_width"),
        })
    st = os.statvfs(bench_dir or "/")
    out = {
        "drives": drives,
        "filesystem": run(["findmnt", "-no", "FSTYPE", "-T", bench_dir or "/"])[1] or None,
        "free_bytes": st.f_bavail * st.f_frsize,
    }
    if bench_dir:
        out["read_bench"] = nvme_bench(bench_dir)
    return out


def nvme_bench(directory):
    if not shutil.which("fio"):
        return {"error": "fio not found"}
    path = os.path.join(directory, "tf_probe_fio.bin")
    try:
        code, out = run(["fio", "--name=probe", "--filename=" + path, "--size=1G", "--rw=read", "--bs=256k",
                         "--iodepth=16", "--direct=1", "--ioengine=io_uring", "--output-format=json"])
        if code != 0:
            return {"error": out[-500:]}
        job = json.loads(out[out.index("{"):])["jobs"][0]["read"]
        return {"bs": "256k", "iodepth": 16, "direct": True, "gb_s": round(job["bw_bytes"] / 1e9, 2),
                "p50_us": round(job["clat_ns"]["percentile"]["50.000000"] / 1000),
                "p99_us": round(job["clat_ns"]["percentile"]["99.000000"] / 1000)}
    finally:
        if os.path.exists(path):
            os.remove(path)


def memory(tmp):
    exe = os.path.join(tmp, "stream")
    ok, log = compile_c(os.path.join(HERE, "stream.c"), exe, ["-pthread"])
    if not ok:
        return {"error": "build failed: " + log[-500:]}
    code, out = run([exe, "256"])
    return json.loads(out) if code == 0 else {"error": out[-500:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "bench", "platform"))
    ap.add_argument("--nvme-bench", metavar="DIR")
    ap.add_argument("--rdma-loopback", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        report = {"probe_version": 1, "date": datetime.date.today().isoformat(), "host": host(args.name)}
        report["gpu"] = gpu(tmp)
        cc = report["gpu"].get("compute_capability")
        report["tcgen05"] = tcgen05(cc)
        report["rdma"] = rdma(args.rdma_loopback)
        report["nvme"] = nvme(args.nvme_bench)
        report["memory"] = memory(tmp)

    g, t = report["gpu"], report["tcgen05"]
    native = t.get("sm_" + (cc or "").replace(".", "") + "a", {})
    report["answers"] = {
        "1_compute_capability": cc,
        "2_tcgen05_tmem": native.get("assembles"),
        "3_rdma_active_ports": sum("ACTIVE" in (p["state"] or "") for p in report["rdma"]["ports"]),
        "3_rdma_loopback_us": report["rdma"].get("loopback", {}).get("t_typical_us"),
        "4_hw_decompress": None if "mem_decompress_algorithm_mask" not in g else g["mem_decompress_algorithm_mask"] != 0,
        "5_cpu_triad_gb_s": report["memory"].get("triad_gb_s"),
        "5_nvme_read_gb_s": report["nvme"].get("read_bench", {}).get("gb_s"),
        "5_mem_available_bytes": report["host"]["mem_available_bytes"],
    }

    text = json.dumps(report, indent=2)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s-%s.json" % (args.name, report["date"]))
    with open(path, "w") as f:
        f.write(text + "\n")
    print("wrote " + os.path.abspath(path), flush=True)


if __name__ == "__main__":
    main()
