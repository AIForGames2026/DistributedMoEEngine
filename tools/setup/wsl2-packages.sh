#!/usr/bin/env bash
# System packages for the WSL2 dev box (Ubuntu 24.04). Run once:
#   sudo bash tools/setup/wsl2-packages.sh
# The GPU driver comes from Windows; this installs only the CUDA toolkit
# (from NVIDIA's WSL repo, which never ships a Linux driver).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo." >&2
    exit 1
fi

cuda_version=13-0 # matches CUDA 13.0 on the DGX Sparks

apt-get update
apt-get install -y build-essential pkg-config git curl python3 \
    liburing-dev rdma-core libibverbs-dev ibverbs-utils fio

keyring=/tmp/cuda-keyring.deb
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -o "$keyring"
dpkg -i "$keyring"
rm -f "$keyring"
apt-get update
apt-get install -y "cuda-toolkit-$cuda_version"

echo
echo "Done. Add to ~/.bashrc:"
echo '  export PATH=/usr/local/cuda/bin:$HOME/tools/zig:$PATH'
