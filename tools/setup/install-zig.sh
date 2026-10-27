#!/usr/bin/env bash
# Installs the Zig version pinned in .zig-version into $PREFIX/zig (default
# ~/tools/zig), verifying the tarball's SHA-256 against ziglang.org's index.
# No sudo needed. Works on x86_64 (WSL2) and aarch64 (DGX Spark).
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
version=$(tr -d '[:space:]' < "$repo/.zig-version")
prefix=${PREFIX:-$HOME/tools}
platform="$(uname -m)-linux"

mkdir -p "$prefix"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL https://ziglang.org/download/index.json -o "$tmp/index.json"
read -r url sum < <(python3 - "$tmp/index.json" "$version" "$platform" <<'EOF'
import json, sys
entry = json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]
print(entry["tarball"], entry["shasum"])
EOF
)

echo "Downloading $url"
curl -fsSL "$url" -o "$tmp/zig.tar.xz"
echo "$sum  $tmp/zig.tar.xz" | sha256sum -c -

rm -rf "$prefix/zig"
mkdir -p "$prefix/zig"
tar xf "$tmp/zig.tar.xz" -C "$prefix/zig" --strip-components=1
"$prefix/zig/zig" version
echo "Add to PATH: export PATH=\"$prefix/zig:\$PATH\""
