"""Drop rates of per-expert and per-rank capacity, from OLMoE recordings (TASK-17).

    python capacity_curve.py NOMINAL.jsonl SPIKE.jsonl

NOMINAL is a recording of 64-token steps and SPIKE of 192-token steps, for example
`olmoe local MODEL --gpu --context 6144 --chunk 64 --generate 2 --tokens T --record F`.
Limits are sized from 64-token steps; "per rank" splits the experts in half as the split does.
"""
import json, math, sys
nominal, k, E, L = 64, 8, 64, 16
def load(path):
    steps = []
    for line in open(path):
        o = json.loads(line)
        if o.get("t") == "step" and o["n"] > 1:
            c = [[0] * E for _ in range(L)]
            x = o["x"]
            for j in range(0, len(x), 3):
                c[x[j]][x[j + 1]] += x[j + 2]
            steps.append((o["n"], c))
    return steps
nom = load(sys.argv[1])
spk = load(sys.argv[2])
def rate(steps, per_expert, C):
    calls = drop = 0
    for n, c in steps:
        for layer in c:
            calls += sum(layer)
            if per_expert:
                lim = math.ceil(C * nominal * k / E)
                drop += sum(max(0, v - lim) for v in layer)
            else:
                lim = math.ceil(C * nominal * k / 2)
                drop += max(0, sum(layer[:32]) - lim) + max(0, sum(layer[32:]) - lim)
    return 100 * drop / calls
print("steps", len(nom), len(spk))
for C in (1.0, 1.1, 1.25, 1.5, 2, 3, 4, 5, 6, 7, 8):
    print(f"C={C}: per expert nominal {rate(nom, True, C):.3f}% spike {rate(spk, True, C):.3f}% | per rank nominal {rate(nom, False, C):.3f}% spike {rate(spk, False, C):.3f}%")
# Per-rank share of calls per layer at nominal
shares = [sum(layer[32:]) / max(sum(layer), 1) for n, c in nom for layer in c]
shares.sort()
print("rank 1 share per layer pass: min %.3f p1 %.3f median %.3f p99 %.3f max %.3f" % (shares[0], shares[len(shares)//100], shares[len(shares)//2], shares[len(shares)*99//100], shares[-1]))
