"""
Enhanced conceptual schematic for multiscale backbone filtering (Serrano-Boguñá-Vespignani)
Focus: one node i, incident edges, local significance test, and edge retention visualization.

Saves:
- plots/multiscale_backbone_schematic.png
- plots/multiscale_backbone_schematic.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example local neighborhood around node i
neighbors = ["A", "B", "C", "D", "E", "F", "G", "H"]
weights = np.array([20.0, 15.0, 3.0, 2.0, 1.0, 0.8, 0.5, 0.3], dtype=float)  # w_ij
k = len(neighbors)
strength = weights.sum()  # s_i
p = weights / strength    # normalized weights
alpha = (1.0 - p) ** (k - 1)  # significance test

ALPHA_THR = 0.05
significant = alpha < ALPHA_THR

# Radial layout for neighbors
R = 3.0
angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
coords = [(R * np.cos(a), R * np.sin(a)) for a in angles]

# Edge linewidths scaled by weights
lw = 1.0 + 4.0 * (weights - weights.min()) / (weights.ptp() if weights.ptp() else 1.0)

fig = plt.figure(figsize=(12, 6))

# ----------------------
# Left panel: local network
# ----------------------
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_aspect('equal')
ax1.axis('off')

center = (0, 0)
ax1.add_patch(Circle(center, radius=0.35, facecolor="#34495E", edgecolor='black', lw=1.5))
ax1.text(*center, "i", color="white", ha="center", va="center", fontsize=12, fontweight='bold')

for idx, (xy, name) in enumerate(zip(coords, neighbors)):
    color = "#2ECC71" if significant[idx] else "#B0B0B0"
    alpha_edge = 0.95 if significant[idx] else 0.4
    ax1.plot([center[0], xy[0]], [center[1], xy[1]], color=color, lw=lw[idx], alpha=alpha_edge, solid_capstyle='round')
    ax1.add_patch(Circle(xy, radius=0.25, facecolor="#ECF0F1", edgecolor='black'))
    ax1.text(xy[0], xy[1], name, ha='center', va='center', fontsize=10)

    # Edge annotation
    midx = (center[0] + xy[0]) / 2
    midy = (center[1] + xy[1]) / 2
    # Highlight weight for significant edges
    lbl_color = "#2ECC71" if significant[idx] else "#555555"
    lbl = f"w={weights[idx]:.1f}\np={p[idx]:.2f}\nα={alpha[idx]:.3f}"
    ax1.text(midx, midy, lbl, fontsize=9, ha='center', va='center', color=lbl_color,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='gray'))

# ax1.set_title("Local view: retain edges with α_ij < α", fontsize=12, fontweight='bold')
ax1.text(0.02, 0.03,
         "w_ij: raw weight\np_ij = w_ij / s_i\nα_ij = (1 - p_ij)^(k_i - 1)",
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

# ----------------------
# Right panel: alpha vs threshold
# ----------------------
ax2 = fig.add_subplot(1, 2, 2)
order = np.argsort(alpha)
nn = [neighbors[i] for i in order]
vals = alpha[order]
colors = ["#2ECC71" if v < ALPHA_THR else "#B0B0B0" for v in vals]

ax2.bar(nn, vals, color=colors, edgecolor='black')
ax2.axhline(ALPHA_THR, color="#E74C3C", lw=2, linestyle='--', label=f"Threshold α = {ALPHA_THR}")
ax2.set_ylabel("α_ij (significance)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Edges from node i", fontsize=11, fontweight='bold')
ax2.set_title("Edge significance test", fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(0.25, vals.max() + 0.03))
ax2.legend(frameon=True)

# Legend-like markers for retained/pruned edges (visual guide)
fig.legend(
    handles=[
        plt.Line2D([0],[0], color="#2ECC71", lw=3, label="Retained (significant)"),
        plt.Line2D([0],[0], color="#B0B0B0", lw=3, label="Pruned (non-significant)")
    ],
    loc='upper right', ncol=2, bbox_to_anchor=(0.5, 0.995), frameon=True
)

# ----------------------
# Caption
# ----------------------
caption = (
    "Multiscale backbone: edges from node i are tested against a uniform null over k_i edges. "
    "Edges with α_ij below threshold (green) are retained; others (gray) are pruned."
)
fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=10)

fig.tight_layout(rect=[0, 0.05, 1, 1])

# Save
png = OUTPUT_DIR / "multiscale_backbone_schematic.png"
pdf = OUTPUT_DIR / "multiscale_backbone_schematic.pdf"
fig.savefig(png, dpi=300, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print(f"Saved: {png}")
print(f"Saved: {pdf}")
