"""Plot test accuracy vs. epoch for 3 seeds with a bold mean line."""
import json, pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
METRICS = HERE.parent / "experiment_logs" / "metrics.jsonl"
OUT = HERE / "learning-curves.png"

per_seed = defaultdict(list)
for line in METRICS.read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    per_seed[rec["seed"]].append((rec["epoch"], rec["test_acc"]))

fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

# per-seed faint lines
for seed, pairs in sorted(per_seed.items()):
    pairs.sort()
    xs = [e for e, _ in pairs]
    ys = [a for _, a in pairs]
    ax.plot(xs, ys, color="#4C72B0", alpha=0.35, linewidth=1.3,
            marker="o", markersize=3.5, label=f"seed {seed}")

# mean line (per epoch across seeds)
epochs = sorted({e for pairs in per_seed.values() for e, _ in pairs})
means = []
for e in epochs:
    vals = [pa for pairs in per_seed.values() for pe, pa in pairs if pe == e]
    means.append(sum(vals) / len(vals))
ax.plot(epochs, means, color="#4C72B0", linewidth=2.6, marker="o",
        markersize=5.5, label="mean", zorder=5)

ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1.0, label="random baseline")

ax.set_xlabel("Epoch")
ax.set_ylabel("Pair-preference accuracy")
ax.set_xticks(epochs)
ax.set_ylim(0.48, 0.66)
ax.set_title("Test accuracy vs. epoch (3 seeds)")
ax.legend(loc="lower right", fontsize=8, frameon=True)
ax.grid(True, linestyle=":", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"OK: {OUT}")
