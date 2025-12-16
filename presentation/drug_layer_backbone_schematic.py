from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

OUTPUT_DIR = Path("presentation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Disease nodes (SGPDN)
# ------------------------
Diseases = ["Aneurysm", "Behcet", "Pulmonary Fibrosis", "Metabolic Syndrome", "Proteinuria"]
coords = {
    "Aneurysm": (-3.4, -1.0),
    "Behcet": (-1.5, -1.6),
    "Pulmonary Fibrosis": (1.0, -0.8),
    "Metabolic Syndrome": (3.2, -1.5),
    "Proteinuria": (-0.6, -2.2),
}

# Backbone edges (illustrative)
backbone_edges = {
    ("Aneurysm", "Behcet"): 0.68,
    ("Aneurysm", "Pulmonary Fibrosis"): 0.75,
    ("Behcet", "Pulmonary Fibrosis"): 0.61,
    ("Proteinuria", "Metabolic Syndrome"): 0.70,
    ("Proteinuria", "Pulmonary Fibrosis"): 0.66,
}

# Molecular overlays
shared_genes = {("Proteinuria", "Pulmonary Fibrosis")}
ppi_support = {("Aneurysm", "Behcet"), ("Pulmonary Fibrosis", "Metabolic Syndrome")}

# Drugs & targets
Drugs = {
    "DrugX": {"target_genes": ["COL3A1"], "targets_diseases": ["Aneurysm"]},
    "DrugY": {"target_genes": ["TGFB1","MMP7"], "targets_diseases": ["Pulmonary Fibrosis"]},
    "DrugZ": {"target_genes": ["INSR"], "targets_diseases": ["Metabolic Syndrome"]},
    "DrugP": {"target_genes": ["NPHS1","APOL1"], "targets_diseases": ["Proteinuria"]},
}

# Indirect repurposing paths
indirect_paths = [("DrugY", "Proteinuria")]

# ------------------------
# Figure setup
# ------------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-3.5, 3.5)

# ------------------------
# Draw backbone edges
# ------------------------
for (a, b), w in backbone_edges.items():
    x1, y1 = coords[a]; x2, y2 = coords[b]
    lw = 1.0 + 2.5 * w
    ax.plot([x1, x2], [y1, y2], color="#7D3C98", lw=lw, alpha=0.25, solid_capstyle='round', zorder=1)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my+0.12, f"{w:.2f}", fontsize=7, color="#7D3C98", ha='center', zorder=6)

# ------------------------
# Draw molecular overlays
# ------------------------
for (a, b) in shared_genes:
    x1, y1 = coords[a]; x2, y2 = coords[b]
    ax.plot([x1, x2], [y1, y2], color="#27AE60", lw=3, alpha=0.9, linestyle='-', zorder=3)
    ax.text((x1+x2)/2, (y1+y2)/2 - 0.14, "shared genes", fontsize=7, color="#27AE60", ha='center', zorder=6)

for (a, b) in ppi_support:
    x1, y1 = coords[a]; x2, y2 = coords[b]
    ax.plot([x1, x2], [y1, y2], color="#2980B9", lw=2, alpha=0.9, linestyle='--', zorder=2)
    ax.text((x1+x2)/2, (y1+y2)/2 + 0.14, "PPI support", fontsize=7, color="#2980B9", ha='center', zorder=6)

# ------------------------
# Draw disease nodes
# ------------------------
for d, (x, y) in coords.items():
    ax.add_patch(Circle((x, y), 0.35, facecolor="#F4F6F7", edgecolor='black', lw=1.5, zorder=4))
    ax.text(x, y, d, ha='center', va='center', fontsize=9, fontweight='bold',
            zorder=5, bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.8))

# ------------------------
# Drug layer
# ------------------------
drug_y = 2.5
drug_x_positions = np.linspace(-3.6, 3.6, len(Drugs))
for (drug, info), dx in zip(Drugs.items(), drug_x_positions):
    ax.add_patch(Circle((dx, drug_y), 0.3, facecolor="#FADBD8", edgecolor='#E74C3C', lw=1.8, zorder=5))
    ax.text(dx, drug_y, drug, ha='center', va='center', fontsize=8, fontweight='bold',
            color='#E74C3C', zorder=6, bbox=dict(facecolor='white', edgecolor='none', pad=0.15, alpha=0.8))
    # Direct targets
    for tgt in info['targets_diseases']:
        if tgt not in coords: continue
        tx, ty = coords[tgt]
        arrow = FancyArrowPatch((dx, drug_y-0.25), (tx, ty+0.65),
                                arrowstyle='-|>', mutation_scale=15, lw=2.3, color='#27AE60', alpha=0.9,
                                connectionstyle="arc3,rad=0.1", zorder=5)
        ax.add_patch(arrow)
        ax.text((dx+tx)/2, (drug_y+ty)/2 + 0.38, "direct", fontsize=7, color='#27AE60', ha='center', zorder=6)

# Indirect repurposing arrows
for drug, dz in indirect_paths:
    if drug not in Drugs or dz not in coords: continue
    dx = list(Drugs.keys()).index(drug)
    dx = drug_x_positions[dx]
    zx, zy = coords[dz]
    arrow = FancyArrowPatch((dx, drug_y-0.3), (zx, zy+0.9),
                            arrowstyle='-|>', mutation_scale=14, lw=1.8, color='#2980B9', alpha=0.8,
                            linestyle='--', connectionstyle="arc3,rad=-0.15", zorder=5)
    ax.add_patch(arrow)
    ax.text((dx+zx)/2, (drug_y+zy)/2 + 0.48, "indirect", fontsize=7, color='#2980B9', ha='center', zorder=6)

# ------------------------
# Layer labels
# ------------------------
ax.text(-4.3, 3.1, "Drug layer", fontsize=10, fontweight='bold', zorder=7)
ax.text(-4.3, 0.8, "Molecular evidence", fontsize=10, fontweight='bold', zorder=7)
ax.text(-4.3, -2.4, "SGPDN backbone", fontsize=10, fontweight='bold', zorder=7)

# ------------------------
# Legend
# ------------------------
legend_elems = [
    Line2D([0],[0], color="#7D3C98", lw=3, label="Backbone edge (weight)"),
    Line2D([0],[0], color="#27AE60", lw=3, label="Shared genes"),
    Line2D([0],[0], color="#2980B9", lw=2.5, ls='--', label="PPI support"),
    Line2D([0],[0], color="#27AE60", lw=2.3, label="Drug → direct"),
    Line2D([0],[0], color="#2980B9", lw=1.8, ls='--', label="Drug → indirect")
]
ax.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=8, title='Edge types')

# ------------------------
# Title & save
# ------------------------
fig.suptitle("Drug layer integration over SGPDN backbone", fontsize=14, fontweight='bold', y=0.97)
fig.tight_layout(rect=[0,0,1,0.95])

png = OUTPUT_DIR / 'drug_layer_backbone_refined_nodes_above.png'
pdf = OUTPUT_DIR / 'drug_layer_backbone_refined_nodes_above.pdf'
fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf, bbox_inches='tight', facecolor='white')
print('Saved:', png)
print('Saved:', pdf)
