"""TASK-32: layer 3's mHC input step in float64 from transformers' layer 2 output, against transformers and the engine, plus its gain for small input noise."""
import sys
import numpy as np
import gguf
sys.path.insert(0, r"D:\CLionProjects\DistributedMoEEngine\tools\ref")
from ref_moe import parse

G = r"E:\models\glm53-flash\GLM-5.3-Flash-Q4_K-00002-of-00002.gguf"
r = gguf.GGUFReader(G)
want = {f"blk.3.hc_attn_{s}" for s in ("fn.weight", "base", "scale")} | {"blk.3.attn_norm.weight"}
t = {}
for x in r.tensors:
    if x.name in want or x.name.startswith("blk.3.hc_attn"):
        t[x.name] = gguf.quants.dequantize(x.data, x.tensor_type).astype(np.float64).reshape([int(v) for v in reversed(x.shape)])
        print(x.name, x.tensor_type.name, t[x.name].shape)
cfg_eps, hc_eps, iters, N, D = 1e-5, 1e-6, 20, 4, 4096
ref = parse(open(r"E:\models\glm53-flash\glm53_layer3.tfref", "rb").read())
gold = parse(open(r"E:\models\glm53-flash\glm53_golden.tfref", "rb").read())
fn = t["blk.3.hc_attn_fn.weight"]
base = t["blk.3.hc_attn_base.weight"].reshape(-1)
scale = t["blk.3.hc_attn_scale.weight"].reshape(-1)
wn = t["blk.3.attn_norm.weight"].reshape(-1)


def attn_in(h):
    flat = h.reshape(h.shape[0], -1)
    flat = flat / np.sqrt((flat ** 2).mean(-1, keepdims=True) + cfg_eps)
    y = flat @ fn.T
    pre = 1 / (1 + np.exp(-(y[:, :N] * scale[0] + base[:N]))) + hc_eps
    col = (pre[:, :, None] * h).sum(1)
    return col / np.sqrt((col ** 2).mean(-1, keepdims=True) + cfg_eps) * wn


for p in range(3):
    h = gold[f"prompt{p}.layer2.hidden"].numpy().astype(np.float64).reshape(-1, N, D)
    want_in = ref[f"prompt{p}.layer3.attn_in"].numpy().astype(np.float64)
    mine = np.fromfile(rf"E:\models\glm53-flash\l3\dump.attn_in.3.{p}.f32", np.float32).reshape(-1, D)
    got = attn_in(h)
    rng = np.random.default_rng(0)
    noisy = attn_in(h + rng.standard_normal(h.shape) * 1e-7 * np.abs(h).max())
    print(f"prompt {p}: numpy vs transformers {np.abs(got - want_in).max():.2e}; engine vs numpy {np.abs(mine - got).max():.2e}; "
          f"input noise 1e-7 of max gives {np.abs(noisy - got).max():.2e}; largest {np.abs(want_in).max():.2e}")
