"""
Analyze weight correlation for overlapping edges between networks.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

OUTPUT_DIR = Path("presentation/plots/backbone")


def analyze_weight_correlation():
    """Compare edge weights for overlapping edges"""
    
    df_orig = pd.read_csv(OUTPUT_DIR / "overlapping_edges_original.tsv", sep='\t')
    df_cons = pd.read_csv(OUTPUT_DIR / "overlapping_edges_consensus.tsv", sep='\t')
    
    # Build edge mapping
    orig_weights = {}
    for _, row in df_orig.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        orig_weights[edge] = row['weight']
    
    cons_weights = {}
    for _, row in df_cons.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        cons_weights[edge] = row['weight']
    
    # Get paired weights
    orig_w = []
    cons_w = []
    edges_data = []
    
    for edge in orig_weights.keys():
        if edge in cons_weights:
            orig_w.append(orig_weights[edge])
            cons_w.append(cons_weights[edge])
            nodes = list(edge)
            edges_data.append({
                'Source': nodes[0],
                'Target': nodes[1],
                'orig_weight': orig_weights[edge],
                'cons_weight': cons_weights[edge],
                'weight_diff': abs(orig_weights[edge] - cons_weights[edge])
            })
    
    # Convert to arrays
    orig_w = np.array(orig_w)
    cons_w = np.array(cons_w)
    
    # Compute correlations
    pcc, pcc_pval = pearsonr(orig_w, cons_w)
    scc, scc_pval = spearmanr(orig_w, cons_w)
    slope, intercept, r2, p_val, std_err = linregress(orig_w, cons_w)
    
    print(f"Weight Correlation Analysis for {len(orig_w)} Overlapping Edges")
    print("="*60)
    print(f"Pearson correlation:  r = {pcc:.3f} (p = {pcc_pval:.2e})")
    print(f"Spearman correlation: ρ = {scc:.3f} (p = {scc_pval:.2e})")
    print(f"Linear regression:    y = {slope:.3f}x + {intercept:.3f}")
    print(f"R² = {r2:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot with regression line
    ax = axes[0, 0]
    ax.scatter(orig_w, cons_w, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
    
    x_line = np.array([orig_w.min(), orig_w.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'y = {slope:.3f}x + {intercept:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect agreement')
    
    ax.set_xlabel("Original Backbone Weight", fontsize=11)
    ax.set_ylabel("Consensus GenePPI Weight", fontsize=11)
    ax.set_title("Edge Weight Correlation", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add correlation stats
    stats_text = f"Pearson r = {pcc:.3f}\nSpearman ρ = {scc:.3f}\nR² = {r2:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # 2. Histograms
    ax = axes[0, 1]
    ax.hist(orig_w, bins=25, alpha=0.6, label='Original', color='steelblue', edgecolor='black')
    ax.hist(cons_w, bins=25, alpha=0.6, label='Consensus', color='coral', edgecolor='black')
    ax.set_xlabel("Edge Weight", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Weight Distribution (Overlapping Edges)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Weight difference
    ax = axes[1, 0]
    diff = np.abs(orig_w - cons_w)
    ax.hist(diff, bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Absolute Weight Difference", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Weight Differences (n={len(diff)}, mean={diff.mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    stats_table = [
        ['Metric', 'Original', 'Consensus'],
        ['Mean', f'{orig_w.mean():.4f}', f'{cons_w.mean():.4f}'],
        ['Median', f'{np.median(orig_w):.4f}', f'{np.median(cons_w):.4f}'],
        ['Std Dev', f'{orig_w.std():.4f}', f'{cons_w.std():.4f}'],
        ['Min', f'{orig_w.min():.4f}', f'{cons_w.min():.4f}'],
        ['Max', f'{orig_w.max():.4f}', f'{cons_w.max():.4f}'],
        ['', '', ''],
        ['Abs Diff Mean', f'{diff.mean():.4f}', ''],
        ['Abs Diff Max', f'{diff.max():.4f}', ''],
    ]
    
    table = ax.table(cellText=stats_table, cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color separator
    for i in range(3):
        table[(6, i)].set_facecolor('#ecf0f1')
    
    ax.set_title("Weight Statistics", fontsize=12, fontweight='bold', pad=15)
    
    fig.suptitle("Overlapping Edge Weights: Original vs Consensus GenePPI", 
                fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "overlapping_weights_correlation.png"
    pdf = OUTPUT_DIR / "overlapping_weights_correlation.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png}")
    print(f"Saved: {pdf}")
    
    # Save detailed edges with both weights
    edges_df = pd.DataFrame(edges_data)
    edges_df = edges_df.sort_values('weight_diff', ascending=False)
    edges_file = OUTPUT_DIR / "overlapping_edges_with_weights.tsv"
    edges_df.to_csv(edges_file, sep='\t', index=False)
    print(f"Saved: {edges_file}")
    
    # Show top differences
    print(f"\nTop 10 largest weight differences:")
    print("="*80)
    for idx, row in edges_df.head(10).iterrows():
        print(f"{row['Source']} <-> {row['Target']}")
        print(f"  Original: {row['orig_weight']:.4f}, Consensus: {row['cons_weight']:.4f}, Diff: {row['weight_diff']:.4f}")


if __name__ == "__main__":
    analyze_weight_correlation()
