# ab_graph.py
# Generates a publication-quality A–B single-case design graph as SVG.

import matplotlib as mpl
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Calibri", "DejaVu Sans", "Arial"]

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- Editable inputs ----
y_label = "% correct responses"      # Dependent variable label
x_unit_label = "Sessions"            # X-axis unit label

# Y-axis scale
y_min = 0
y_max = 100
y_tick = 10

# X tick density, set to 1 for every point, 2 for every other, etc.
x_tick_step = 1

# Data, baseline first, then intervention
baseline = [40, 42, 39, 41, 43]
intervention = [55, 60, 65, 68, 70, 74]

# Appearance
baseline_color = "#1f77b4"           # Blue
intervention_color = "#ff7f0e"       # Orange
line_width = 2.0
marker_size = 6

# Output file
out_file = Path("ab_graph.svg")
# -------------------------

# Build x values
nA = len(baseline)
nB = len(intervention)
x_A = np.arange(1, nA + 1)
x_B = np.arange(nA + 1, nA + nB + 1)

# Figure
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Plot A and B separately, no connecting line between phases
ax.plot(x_A, baseline,
        color=baseline_color, linewidth=line_width, marker="o",
        markersize=marker_size, label="Baseline (A)")
ax.plot(x_B, intervention,
        color=intervention_color, linewidth=line_width, marker="s",
        markersize=marker_size, label="Intervention (B)")

# Phase change line, dashed vertical between A and B
phase_boundary = nA + 0.5
ax.axvline(x=phase_boundary, color="black", linestyle=(0, (5, 5)), linewidth=1.5)

# Axes and ticks
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.tick_params(axis="both", which="both", direction="out", length=6, pad=6)

# Y scale and ticks
ax.set_ylim(y_min, y_max)
ax.set_yticks(np.arange(y_min, y_max + 0.0001, y_tick))

# X ticks
total_points = nA + nB
ax.set_xlim(0.5, total_points + 0.5)
ax.set_xticks(np.arange(1, total_points + 1, x_tick_step))

# Labels
ax.set_ylabel(y_label, rotation=90, labelpad=20)
ax.set_xlabel(x_unit_label, labelpad=12)

# Legend
leg = ax.legend(frameon=False, loc="upper left")

# No grid lines inside the plot
ax.grid(False)

# Save as SVG for crisp scaling in Word
fig.savefig(out_file, format="svg")
print(f"Saved: {out_file.resolve()}")
