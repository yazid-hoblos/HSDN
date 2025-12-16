"""
Slide-ready schematic: Construction of the Symptom–Gene–PPI Disease Network (SGPDN)

Layers:
- Top (HSDN): diseases linked by symptom similarity (filtered > 0.1)
- Middle (Molecular): shared genes (green) and PPI (blue) between disease-associated proteins
- Bottom (SGPDN): integrated edges with thickness ~ evidence count

Outputs:
- presentation/plots/sgpdn_schematic.png
- presentation/plots/sgpdn_schematic.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed small example set for clarity
nodes = ["D1","D2","D3","D4","D5","D6"]
N = len(nodes)

# Common coordinates (rows for layers, same x positions)
x = np.linspace(-4.5, 4.5, N)
row_y = {"top": 3.5, "mid": 0.0, "bot": -3.5}
coords = {layer: {n: (x[i], y) for i, n in enumerate(nodes)} for layer, y in row_y.items()}

# Edges per layer (pairs by index), small network for readability
# Symptom similarity (HSDN), thresholded > 0.1 (orange)
hsdn_edges = {
    ("D1","D2"): 0.35,
    ("D1","D3"): 0.22,
    ("D2","D4"): 0.18,
    ("D3","D5"): 0.41,
    ("D4","D6"): 0.29,
    ("D5","D6"): 0.14,
}

# Molecular: shared genes (green, solid) and PPI (blue, dashed)
shared_gene_edges = {
    ("D1","D3"), ("D2","D4"), ("D3","D6")
}
ppi_edges = {
    ("D2","D3"), ("D3","D5"), ("D4","D6")
}

# SGPDN: integrate if present in either HSDN or Molecular
# Weight = count of evidences across layers: HSDN(+1), shared_gene(+1), PPI(+1)
from collections import defaultdict
w_integrated = defaultdict(int)
for (a,b) in hsdn_edges.keys():
    w_integrated[tuple(sorted((a,b)))] += 1
for (a,b) in shared_gene_edges:
    w_integrated[tuple(sorted((a,b)))] += 1
for (a,b) in ppi_edges:
    w_integrated[tuple(sorted((a,b)))] += 1

# Visual helpers
def draw_nodes(ax, layer, title):
    for n in nodes:
        cx, cy = coords[layer][n]
        ax.add_patch(Circle((cx, cy), 0.28, facecolor="#F4F6F7", edgecolor='black', lw=1.5))
        ax.text(cx, cy, n, ha='center', va='center', fontsize=10)
    ax.text(0, row_y[layer]+0.8, title, ha='center', va='bottom', fontsize=12, fontweight='bold')


def draw_edges(ax, pairs, layer, color, lw=2.5, style='-'):
    for (a,b) in pairs:
        (x1,y1) = coords[layer][a]
        (x2,y2) = coords[layer][b]
        ax.plot([x1,x2],[y1,y2], color=color, lw=lw, ls=style, alpha=0.9, solid_capstyle='round', zorder=0)


def draw_hsdn(ax):
    # Orange edges, annotate similarity near midpoints
    for (a,b), sim in hsdn_edges.items():
        (x1,y1) = coords['top'][a]
        (x2,y2) = coords['top'][b]
        ax.plot([x1,x2],[y1,y2], color="#E67E22", lw=3, alpha=0.9, solid_capstyle='round')
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, f"sim={sim:.2f}", fontsize=8, ha='center', va='bottom', color="#E67E22")


def draw_molecular(ax):
    # Shared genes (green solid)
    draw_edges(ax, shared_gene_edges, 'mid', color="#27AE60", lw=3.0, style='-')
    # PPIs (blue dashed)
    draw_edges(ax, ppi_edges, 'mid', color="#2980B9", lw=3.0, style='--')


def draw_sgpdn(ax):
    # Edge thickness reflects combined evidence count (1,2,3)
    for (a,b), w in sorted(w_integrated.items()):
        (x1,y1) = coords['bot'][a]
        (x2,y2) = coords['bot'][b]
        lw = 2.0 + 1.5*(w-1)
        ax.plot([x1,x2],[y1,y2], color="#7D3C98", lw=lw, alpha=0.95, solid_capstyle='round')
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my-0.18, f"evidence={w}", fontsize=8, ha='center', va='top', color="#7D3C98")

    # Highlight node diversity (illustrative): mark D3 as diverse
    (cx, cy) = coords['bot']["D3"]
    ax.add_patch(Circle((cx, cy), 0.42, facecolor='none', edgecolor="#F1C40F", lw=2.5))
    ax.text(cx+0.65, cy+0.05, "high diversity", fontsize=9, color="#F1C40F")


# Build figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
ax.set_xlim(-6, 6)
ax.set_ylim(-5.2, 5.2)

# Titles
fig.suptitle("Construction of the Symptom–Gene–PPI Disease Network (SGPDN)", fontsize=15, fontweight='bold', y=0.98)

# Draw each layer
draw_nodes(ax, 'top', 'HSDN: symptom similarity (> 0.1)')
draw_hsdn(ax)

draw_nodes(ax, 'mid', 'Molecular layer: shared genes (green), PPIs (blue)')
draw_molecular(ax)

draw_nodes(ax, 'bot', 'SGPDN: integrated disease network (edge thickness = evidence count)')
draw_sgpdn(ax)

# Flow arrows
arrow_kwargs = dict(arrowstyle='-|>', mutation_scale=16, lw=2, color='gray', alpha=0.8)
ax.add_patch(FancyArrowPatch((0, row_y['top']-0.6), (0, row_y['mid']+0.6), **arrow_kwargs))
ax.add_patch(FancyArrowPatch((0, row_y['mid']-0.6), (0, row_y['bot']+0.6), **arrow_kwargs))
ax.text(0.1, row_y['top']-0.1, "filter by similarity", fontsize=9, color='gray')
ax.text(0.1, row_y['mid']-0.1, "add shared genes / PPIs", fontsize=9, color='gray')

# Legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], color="#E67E22", lw=3, label='Symptom similarity (HSDN)'),
    Line2D([0],[0], color="#27AE60", lw=3, label='Shared genes'),
    Line2D([0],[0], color="#2980B9", lw=3, ls='--', label='PPIs between associated proteins'),
    Line2D([0],[0], color="#7D3C98", lw=4, label='SGPDN integrated edge')
]
ax.legend(handles=legend_elems, loc='lower right', frameon=True, title='Edge types')

# Optional annotations (example counts; change as needed for your talk)
ax.text(-5.6, 4.6, "Example scale:\n1,596 diseases\n133,106 interactions", fontsize=9, ha='left', va='top', color='dimgray')

# Save
out_png = OUTPUT_DIR / 'sgpdn_schematic.png'
out_pdf = OUTPUT_DIR / 'sgpdn_schematic.pdf'
fig.tight_layout(rect=[0,0.03,1,0.96])
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
