"""TASK-32: per-layer hidden-state spread between the 20-thread and 7-thread transformers references, prompt 0."""
import sys
sys.path.insert(0, r"D:\CLionProjects\DistributedMoEEngine\tools\ref")
from ref_moe import parse
a = parse(open(r"E:\models\glm53-flash\glm53_golden.tfref", "rb").read())
b = parse(open(r"E:\models\glm53-flash\glm53_golden_t7.tfref", "rb").read())
for l in (0, 1, 2, 3, 4, 10, 18, 21, 29, 44):
    x, y = a[f"prompt0.layer{l}.hidden"], b[f"prompt0.layer{l}.hidden"]
    print(l, f"{float((x - y).abs().max()):.2e}", f"{float(x.abs().max()):.2e}")
