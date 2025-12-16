"""
Option A: Drug Repurposing Pipeline & Top Candidates
- Left: approach pipeline diagram (SGPDN → similarity → genes → repurposing)
- Right: top 10 candidates as ranked table with heatmap
Outputs: presentation/plots/drug_repurposing_pipeline.png|pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import seaborn as sns

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Example top 10 candidates (from analysis/repurposing_summary.txt)
candidates = [
    ("Proteinuria", "Nephrotic Syndrome", 0.994, 2, 0.810),
    ("Idiopathic Pulmonary Fibrosis", "Pulmonary Fibrosis", 0.999, 1, 0.808),
    ("Pseudohypoparathyroidism", "Pseudopseudohypoparathyroidism", 0.995, 1, 0.803),
    ("Dystonia", "Dystonic Disorders", 0.999, 0, 0.800),
    ("Proteinuria", "Glomerulosclerosis, Focal Segmental", 0.998, 0, 0.799),
    ("Nephrocalcinosis", "Nephrolithiasis", 0.956, 6, 0.798),
    ("Coronary Artery Disease", "Coronary Disease", 0.995, 0, 0.795),
    ("Insulin Resistance", "Metabolic Syndrome X", 0.991, 0, 0.790),
    ("Syncope", "Long QT Syndrome", 0.990, 0, 0.790),
    ("Cholelithiasis", "Gallstones", 0.989, 0, 0.788),
]

df = pd.DataFrame(candidates, columns=["Disease A", "Disease B", "Similarity", "Shared Genes", "Repurposing Score"])

# Create figure with two panels
fig = plt.figure(figsize=(14, 7))

# ========== LEFT PANEL: APPROACH PIPELINE ==========
ax_left = fig.add_subplot(1, 2, 1)
ax_left.set_xlim(-0.5, 4.5)
ax_left.set_ylim(-0.5, 5)
ax_left.axis('off')

# Pipeline steps (boxes)
steps = [
    ("SGPDN", "Integrated disease\nnetwork"),
    ("Symptom\nSimilarity", "Filter disease pairs\nwith sim > 0.3"),
    ("Shared\nGenes", "Identify molecular\nsupport"),
    ("Drug Targets", "Map known drugs\nto diseases"),
    ("Repurposing\nCandidates", "Ranked by evidence"),
]

colors = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C", "#9B59B6"]
x_pos = np.linspace(0, 4, len(steps))

for i, (title, desc) in enumerate(steps):
    # Box
    box = FancyBboxPatch((x_pos[i]-0.35, 2), 0.7, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=colors[i], alpha=0.85, lw=2)
    ax_left.add_patch(box)
    ax_left.text(x_pos[i], 2.75, title, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax_left.text(x_pos[i], 0.7, desc, ha='center', va='top', fontsize=8, color='black')

# Arrows between steps
for i in range(len(steps)-1):
    arrow = FancyArrowPatch((x_pos[i]+0.4, 2.75), (x_pos[i+1]-0.4, 2.75),
                           arrowstyle='->', mutation_scale=20, lw=2.5, color='gray')
    ax_left.add_patch(arrow)

# Title
ax_left.text(2, 4.5, "Drug Repurposing Pipeline", fontsize=13, fontweight='bold', ha='center')

# ========== RIGHT PANEL: TOP 10 CANDIDATES TABLE + HEATMAP ==========
ax_right = fig.add_subplot(1, 2, 2)

# Create a simplified table view with heatmap coloring
table_data = []
labels = []
for idx, row in df.iterrows():
    if idx >= 10:
        break
    label = f"{row['Disease A'][:20]}\n↔ {row['Disease B'][:20]}"
    labels.append(label)
    table_data.append([row['Similarity'], row['Shared Genes'], row['Repurposing Score']])

table_array = np.array(table_data)

# Normalize for heatmap coloring
table_norm = (table_array - table_array.min(axis=0)) / (table_array.max(axis=0) - table_array.min(axis=0))

# Plot as heatmap
im = ax_right.imshow(table_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Overlay text
for i in range(table_norm.shape[0]):
    for j in range(table_norm.shape[1]):
        val = table_array[i, j]
        text = ax_right.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="black" if table_norm[i,j] > 0.5 else "white", fontweight='bold', fontsize=9)

# Set ticks and labels
ax_right.set_xticks([0, 1, 2])
ax_right.set_xticklabels(['Symptom\nSimilarity', 'Shared\nGenes', 'Repurposing\nScore'], fontsize=10)
ax_right.set_yticks(range(10))
ax_right.set_yticklabels([f"#{i+1}" for i in range(10)], fontsize=9)

ax_right.set_title("Top 10 Repurposing Candidates", fontsize=12, fontweight='bold', pad=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax_right, orientation='vertical', pad=0.02)
cbar.set_label('Normalized Score', fontsize=9)

# Add legend with disease pairs on the side
legend_text = "Candidate Pairs:\n" + "\n".join([f"{i+1}. {r['Disease A'][:15]} ↔ {r['Disease B'][:15]}" 
                                                 for i, (_, r) in enumerate(df.head(10).iterrows())])
fig.text(0.02, 0.98, legend_text, fontsize=8, ha='left', va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle("Drug Repurposing: Approach & Top Candidates", fontsize=15, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0.15, 0, 1, 0.96])

# Save
png_path = OUTPUT_DIR / "drug_repurposing_pipeline.png"
pdf_path = OUTPUT_DIR / "drug_repurposing_pipeline.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
