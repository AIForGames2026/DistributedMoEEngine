"""TASK-32: layer 3's attention and MoE inputs and outputs, engine (TF_DUMP_LAYER=3 files) against transformers (--dump-layer 3)."""
import sys
import numpy as np
sys.path.insert(0, r"D:\CLionProjects\DistributedMoEEngine\tools\ref")
from ref_moe import parse

ref = parse(open(r"E:\models\glm53-flash\glm53_layer3.tfref", "rb").read())
D = 4096
for p in range(3):
    print(f"prompt {p}")
    mine = {}
    for name in ("attn_in", "attn_out", "mlp_in", "routed_out", "mlp_out"):
        mine[name] = np.fromfile(rf"E:\models\glm53-flash\l3\dump.{name}.3.{p}.f32", np.float32).reshape(-1, D)
    mine["shared_out"] = mine["mlp_out"] - mine["routed_out"]
    for name in ("attn_in", "attn_out", "mlp_in", "routed_out", "shared_out", "mlp_out"):
        r = ref[f"prompt{p}.layer3.{name}"].numpy().reshape(-1, D)
        m = mine[name]
        err = np.abs(m - r).max()
        big = np.abs(r).max()
        print(f"  {name:10s} max err {err:.2e}, largest {big:.2e}, relative {err / big:.2e}")
