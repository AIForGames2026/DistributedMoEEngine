#!/usr/bin/env python3
"""TASK-36: checks that ggml's own dequantization matches gguf-py's on every tensor of a GGUF,
bit for bit. The engine's unit test compares its decoders with gguf-py on
tests/golden/quant_formats.gguf; this closes the loop to ggml itself.

Usage (libggml-base from a llama.cpp build, e.g. cmake -B build && cmake --build build --target ggml-base):
  python3 tools/ref/check_ggml_quants.py path/to/libggml-base.so tests/golden/quant_formats.gguf
"""
import ctypes
import sys

import numpy as np
import gguf

to_float_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)


class Traits(ctypes.Structure):
    """ggml.h's struct ggml_type_traits."""
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


def main():
    lib = ctypes.CDLL(sys.argv[1])
    lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
    bad = 0
    for t in gguf.GGUFReader(sys.argv[2]).tensors:
        traits = lib.ggml_get_type_traits(int(t.tensor_type)).contents
        raw = np.ascontiguousarray(t.data).view(np.uint8)
        n = int(np.prod([int(d) for d in t.shape]))
        out = np.zeros(n, dtype=np.float32)
        to_float_t(traits.to_float)(raw.ctypes.data, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
        with np.errstate(all="ignore"):
            ref = gguf.quants.dequantize(raw.reshape(t.data.shape), t.tensor_type).reshape(-1).astype(np.float32)
        differ = int((out.view(np.uint32) != ref.view(np.uint32)).sum())
        bad += differ > 0
        print(f"{t.name:16s} {traits.type_name.decode():8s} {'same' if differ == 0 else f'{differ} values differ'}")
    if bad:
        sys.exit(f"FAIL: {bad} tensors differ")
    print("OK: ggml and gguf-py agree bit for bit on every tensor")


if __name__ == "__main__":
    main()
