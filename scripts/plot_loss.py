"""Plot the training loss curve from outputs/<run>/log.jsonl.

    python scripts/plot_loss.py outputs/mnist/log.jsonl
"""
import json
import sys

import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1])]
plt.plot([r["step"] for r in rows], [r["loss"] for r in rows])
plt.xlabel("step"); plt.ylabel("ε-prediction MSE"); plt.yscale("log"); plt.grid(alpha=0.3)
out = sys.argv[1].rsplit("/", 1)[0] + "/loss.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
