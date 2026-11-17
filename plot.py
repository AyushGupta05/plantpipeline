import pandas as pd
import matplotlib.pyplot as plt

# GrowFields on its own 8 test plants (TrackPlant3D).
# "Day 0"   = original GrowFields: roll out from each leaf's first scan.
# "Prev scan" = same trained models, roll out from the most recent observed scan.
df = pd.DataFrame({
    "test":   ["Fit", "Fit", "Fill missing days", "Fill missing days", "Predict future days", "Predict future days"],
    "start":  ["Day 0", "Prev scan"] * 3,
    "shape":      [0.87, 0.44, 10.32, 10.93, 98.82, 133.88],   # Chamfer distance, mm^2
    "shape_std":  [0.02, 0.01,  0.08,  0.07, 12.72,  22.72],
    "tip":        [1.53, 1.02,  5.38,  3.65, 21.21,  11.70],   # leaf-tip error, mm
    "tip_std":    [0.06, 0.04,  0.14,  0.04,  0.52,   0.68],
})

tests = ["Fit", "Fill missing days", "Predict future days"]
metrics = [("shape", "Shape error (Chamfer, mm²)"), ("tip", "Leaf-tip error (mm)")]
colors = {"Day 0": "#9aa0a6", "Prev scan": "#1a73e8"}

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for r, (col, label) in enumerate(metrics):
    for c, test in enumerate(tests):
        ax = axes[r, c]
        d = df[df["test"] == test].set_index("start")
        bars = ax.bar(d.index, d[col], yerr=d[f"{col}_std"], capsize=4,
                      color=[colors[s] for s in d.index])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylim(0, (d[col] + d[f"{col}_std"]).max() * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.set_title(test)
        if c == 0:
            ax.set_ylabel(label)

fig.suptitle("GrowFields: rollout from day 0 vs from the previous scan (lower is better)")
fig.tight_layout()
plt.show()

# Percentage change table
pivot = df.pivot(index="test", columns="start", values=["shape", "tip"]).reindex(tests)
change = pd.DataFrame({
    m: (pivot[(m, "Prev scan")] / pivot[(m, "Day 0")] - 1) * 100 for m in ["shape", "tip"]
}).round(0)
print(change.rename(columns={"shape": "shape error % change", "tip": "tip error % change"}))