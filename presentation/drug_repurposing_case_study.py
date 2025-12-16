"""
Option B: Drug Repurposing Case Study
Detailed 3-panel visualization for one high-confidence pair:
- Panel 1: Disease pair with shared symptoms (heatmap)
- Panel 2: Molecular mechanism (shared genes, PPI)
- Panel 3: Known drugs on source → repurposable to target

Focus: Insulin Resistance ↔ Metabolic Syndrome X (sim=0.991, shared genes=0, score=0.790)
Alternative: Proteinuria ↔ Nephrotic Syndrome (sim=0.994, shared genes=2, score=0.810)

Outputs: presentation/plots/drug_repurposing_case_study.png|pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Case study: Proteinuria ↔ Nephrotic Syndrome (high sim + shared genes)
disease_a = "Proteinuria"
disease_b = "Nephrotic Syndrome"
similarity = 0.994
shared_genes = ["NPHS1", "APOL1"]  # Example shared genes
shared_symptoms = ["Edema", "Albuminuria", "Hyperlipidemia", "Hypoproteinemia", "Nephrotic Syndrome"]

# Known drugs for Proteinuria (examples)
drugs_for_a = [
    ("ACE Inhibitor", "Reduces protein loss"),
    ("Diuretic", "Manages edema"),
    ("Statin", "Lowers cholesterol"),
]

fig = plt.figure(figsize=(14, 6))

# ========== PANEL 1: SHARED SYMPTOMS HEATMAP ==========
ax1 = fig.add_subplot(1, 3, 1)

symptoms_full = ["Edema", "Albuminuria", "Hyperlipidemia", "Hypoproteinemia", "Nephrotic Syndrome",
                 "Hypertension", "Proteinuria", "Fever", "Weight Loss", "Fatigue"]

# Simulated presence matrix (1 = symptom present, 0 = absent)
presence = np.array([
    [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  # Proteinuria row
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Nephrotic Syndrome row
])

im = ax1.imshow(presence, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(len(symptoms_full)))
ax1.set_xticklabels(symptoms_full, rotation=45, ha='right', fontsize=9)
ax1.set_yticks([0, 1])
ax1.set_yticklabels([disease_a, disease_b], fontsize=10)
ax1.set_title("Shared Symptoms", fontsize=12, fontweight='bold')

# Add text: shared count
shared_count = np.sum(presence[0] * presence[1])
ax1.text(len(symptoms_full)/2, -0.6, f"Shared: {int(shared_count)}/{len(symptoms_full)}", 
        ha='center', fontsize=10, fontweight='bold', transform=ax1.get_xaxis_transform())

# ========== PANEL 2: MOLECULAR MECHANISM ==========
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-2.5, 2.5)
ax2.axis('off')

# Left: disease A
ax2.add_patch(Circle((-1.5, 0), 0.4, facecolor="#3498DB", edgecolor='black', lw=2))
ax2.text(-1.5, 0, disease_a.split()[0], ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Right: disease B
ax2.add_patch(Circle((1.5, 0), 0.4, facecolor="#E74C3C", edgecolor='black', lw=2))
ax2.text(1.5, 0, disease_b.split()[0], ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Shared genes in middle (nodes)
gene_y = np.linspace(0.5, -0.5, len(shared_genes))
for i, gene in enumerate(shared_genes):
    y = gene_y[i]
    ax2.add_patch(Circle((0, y), 0.2, facecolor="#2ECC71", edgecolor='black', lw=1.5))
    ax2.text(0, y, gene, ha='center', va='center', fontsize=8, fontweight='bold')
    # Edges from diseases to genes
    ax2.plot([-1.1, -0.2], [0, y], 'k-', lw=1.5, alpha=0.6)
    ax2.plot([1.1, 0.2], [0, y], 'k-', lw=1.5, alpha=0.6)

ax2.set_title("Molecular Bridge\n(Shared Genes)", fontsize=12, fontweight='bold')
ax2.text(0, -1.8, f"Similarity: {similarity:.3f}\nShared genes: {len(shared_genes)}", 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ========== PANEL 3: DRUG REPURPOSING ==========
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_xlim(-0.5, 3)
ax3.set_ylim(-0.5, 4.5)
ax3.axis('off')

# Known drugs for disease A
y_pos = 4
ax3.text(0.5, y_pos, f"Drugs for {disease_a}:", fontsize=11, fontweight='bold')
y_pos -= 0.5

for drug, mechanism in drugs_for_a:
    # Drug box
    box = FancyBboxPatch((0, y_pos-0.35), 1.2, 0.4, boxstyle="round,pad=0.05",
                         edgecolor='#3498DB', facecolor='#ECF0F1', lw=2)
    ax3.add_patch(box)
    ax3.text(0.6, y_pos-0.15, drug, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow to repurpose
    arrow = FancyArrowPatch((1.3, y_pos-0.15), (1.7, y_pos-0.15),
                           arrowstyle='->', mutation_scale=15, lw=2.5, color='#27AE60')
    ax3.add_patch(arrow)
    
    # Target box
    box2 = FancyBboxPatch((1.8, y_pos-0.35), 1.2, 0.4, boxstyle="round,pad=0.05",
                          edgecolor='#E74C3C', facecolor='#FADBD8', lw=2)
    ax3.add_patch(box2)
    ax3.text(2.4, y_pos-0.15, disease_b.split()[0], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Mechanism text
    ax3.text(2.8, y_pos, mechanism, fontsize=8, style='italic', color='gray')
    
    y_pos -= 0.7

ax3.set_title(f"Repurposing:\n{disease_a} → {disease_b}", fontsize=12, fontweight='bold')

# Overall title
fig.suptitle(f"Case Study: Drug Repurposing for {disease_a} to {disease_b}",
            fontsize=14, fontweight='bold', y=0.98)

fig.tight_layout(rect=[0, 0, 1, 0.96])

# Save
png_path = OUTPUT_DIR / "drug_repurposing_case_study.png"
pdf_path = OUTPUT_DIR / "drug_repurposing_case_study.pdf"
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
