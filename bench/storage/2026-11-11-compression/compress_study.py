"""How well a GGUF file's Q8_0 expert weights compress losslessly, and NVFP4 made from them (TASK-13).

Standard library only: python3 compress_study.py MODEL.gguf
"""
import bisect
import collections
import lzma
import math
import struct
import sys
import zlib


def read_gguf(path):
    f = open(path, 'rb')

    def rd(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

    def rstr():
        (n,) = rd('Q')
        return f.read(n).decode()

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        if t in sizes:
            f.read(sizes[t])
        elif t == 8:
            rstr()
        elif t == 9:
            (at,) = rd('I')
            (n,) = rd('Q')
            for _ in range(n):
                skip(at)

    _, _, n_t, n_kv = rd('4sIQQ')
    align = 32
    for _ in range(n_kv):
        key = rstr()
        (t,) = rd('I')
        if key == 'general.alignment':
            (align,) = rd('I')
        else:
            skip(t)
    tensors = []
    for _ in range(n_t):
        name = rstr()
        (nd,) = rd('I')
        dims = rd('Q' * nd)
        (ty,) = rd('I')
        (off,) = rd('Q')
        tensors.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    return f, base, tensors


def entropy(b):
    c = collections.Counter(b)
    n = len(b)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def ratio(raw, parts):
    return len(raw) / sum(len(lzma.compress(bytes(p))) for p in parts)


# Every positive E4M3 value (bias 7, with subnormals), for nearest rounding.
E4M3 = sorted({(m / 8) * 2.0 ** -6 for m in range(1, 8)} |
              {(1 + m / 8) * 2.0 ** (e - 7) for e in range(1, 16) for m in range(8) if (e, m) != (15, 7)})
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E2M1_MID = [(E2M1[i] + E2M1[i + 1]) / 2 for i in range(7)]


def nvfp4(vals):
    """NVFP4 codes (one per value) and E4M3 scale codes (one per 16 values), with a tensor scale."""
    gs = max(abs(v) for v in vals) / (6 * 448) or 1.0
    codes, scales = [], []
    for i in range(0, len(vals) - 15, 16):
        blk = vals[i:i + 16]
        want = max(abs(v) for v in blk) / 6 / gs
        j = min(bisect.bisect_left(E4M3, want), len(E4M3) - 1)
        if j > 0 and want - E4M3[j - 1] < E4M3[j] - want:
            j -= 1
        scales.append(j + 1 if want > 0 else 0)
        s = E4M3[j] * gs
        for v in blk:
            k = bisect.bisect_left(E2M1_MID, abs(v) / s) if want > 0 else 0
            codes.append(k | (8 if v < 0 and k else 0))
    return codes, scales


def main():
    f, base, tensors = read_gguf(sys.argv[1])
    experts = [t for t in tensors if 'exps' in t[0] and t[2] == 8]
    print('%d Q8_0 expert tensors, for example %s %s' % (len(experts), experts[0][0], experts[0][1]))
    raw, quants, scales, vals = bytearray(), bytearray(), bytearray(), []
    for name, dims, _, off in experts:
        blocks = dims[0] * dims[1] * dims[2] // 32
        for part in range(4):
            f.seek(base + off + 34 * (blocks * part // 4))
            blob = f.read(34 * 8192)
            raw += blob
            for i in range(0, len(blob), 34):
                scales += blob[i:i + 2]
                quants += blob[i + 2:i + 34]
            if part == 0:
                for i in range(0, 34 * 512, 34):
                    d = struct.unpack('<e', blob[i:i + 2])[0]
                    vals += [d * x for x in struct.unpack('32b', blob[i + 2:i + 34])]
    hq, hlo, hhi = entropy(quants), entropy(scales[0::2]), entropy(scales[1::2])
    ideal = len(raw) / (len(quants) * hq / 8 + len(scales) / 2 * (hlo + hhi) / 8)
    print('Q8_0 sample: %.1f MB' % (len(raw) / 1e6))
    print('  8-bit values: %.3f bits of 8; f16 scales: low byte %.2f, high byte %.2f bits of 8' % (hq, hlo, hhi))
    print('  best order-0 ratio %.3f; zlib -9 %.3f; lzma %.3f; lzma on split values and scales %.3f' % (
        ideal, len(raw) / len(zlib.compress(bytes(raw), 9)), ratio(raw, [raw]), ratio(raw, [quants, scales])))
    codes, sc = nvfp4(vals)
    hc, hs = entropy(codes), entropy(sc)
    packed = bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes) - 1, 2))
    print('NVFP4 from the same weights: %d values' % len(codes))
    print('  4-bit codes: %.3f bits of 4; E4M3 scales: %.3f bits of 8' % (hc, hs))
    print('  best order-0 ratio %.3f; lzma on split codes and scales %.3f' % (
        4.5 / (hc + hs / 16), (len(packed) + len(sc)) / (len(lzma.compress(packed)) + len(lzma.compress(bytes(sc))))))


main()
