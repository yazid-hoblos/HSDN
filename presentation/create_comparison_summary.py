"""
Create a comprehensive 1-page summary figure comparing both networks.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

OUTPUT_DIR = Path("presentation/plots/backbone")

def load_data():
    """Load all necessary data"""
    orig_comm = pd.read_csv("presentation/plots/original_backbone_communities.tsv", sep='\t')
    cons_comm = pd.read_csv("presentation/plots/consensus_geneppi_communities.tsv", sep='\t')
    
    orig_stats = pd.read_csv("presentation/plots/backbone/backbone_comparison_network_stats.tsv", sep='\t')
    comm_stats = pd.read_csv("presentation/plots/community_comparison_detailed_stats.tsv", sep='\t')
    
    return orig_comm, cons_comm, orig_stats, comm_stats


def create_summary_figure():
    """Create comprehensive comparison summary"""
    orig_comm, cons_comm, orig_stats, comm_stats = load_data()
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle("Backbone Network Comparison: Original vs Replication", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Network Statistics Table (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('tight')
    ax1.axis('off')
    
    stats_data = [
        ['Metric', 'Original Backbone', 'Consensus GenePPI'],
        ['Nodes', f"{orig_stats.loc[0, 'Nodes']:.0f}", f"{orig_stats.loc[1, 'Nodes']:.0f}"],
        ['Edges', f"{orig_stats.loc[0, 'Edges']:.0f}", f"{orig_stats.loc[1, 'Edges']:.0f}"],
        ['Density', f"{orig_stats.loc[0, 'Density']:.4f}", f"{orig_stats.loc[1, 'Density']:.4f}"],
        ['Avg Degree', f"{orig_stats.loc[0, 'Avg Degree']:.2f}", f"{orig_stats.loc[1, 'Avg Degree']:.2f}"],
        ['Avg Clustering', f"{orig_stats.loc[0, 'Avg Clustering']:.3f}", f"{orig_stats.loc[1, 'Avg Clustering']:.3f}"],
        ['Communities', 
         f"{comm_stats[comm_stats['Network']=='Original']['Community'].nunique()}", 
         f"{comm_stats[comm_stats['Network']=='Consensus GenePPI']['Community'].nunique()}"],
    ]
    
    table = ax1.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    # ax1.set_title("Network Statistics", fontsize=12, fontweight='bold', pad=10)
    
    # 2. Community Distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    orig_sizes = comm_stats[comm_stats['Network']=='Original']['Size'].values
    cons_sizes = comm_stats[comm_stats['Network']=='Consensus GenePPI']['Size'].values
    
    bp1 = ax2.boxplot([orig_sizes, cons_sizes], positions=[1, 2], widths=0.6,
                      patch_artist=True, showmeans=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    
    ax2.set_xticklabels(['Original', 'Replication'])
    ax2.set_ylabel("Community Size (nodes)", fontsize=10)
    ax2.set_title("Community Size Distribution", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Degree Distribution Comparison (middle row, left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    orig_degrees = orig_comm['degree'].values
    cons_degrees = cons_comm['degree'].values
    
    ax3.hist(orig_degrees, bins=30, alpha=0.6, label='Original', color='steelblue', edgecolor='black')
    ax3.hist(cons_degrees, bins=30, alpha=0.6, label='RE', color='coral', edgecolor='black')
    ax3.set_xlabel("Degree", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title("Degree Distribution", fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Strength Distribution (middle row, middle)
    ax4 = fig.add_subplot(gs[1, 1])
    
    orig_strength = orig_comm['strength'].values
    cons_strength = cons_comm['strength'].values
    
    ax4.hist(np.log10(orig_strength + 1e-9), bins=30, alpha=0.6, label='Original', 
             color='steelblue', edgecolor='black')
    ax4.hist(np.log10(cons_strength + 1e-9), bins=30, alpha=0.6, label='Replication', 
             color='coral', edgecolor='black')
    ax4.set_xlabel("log₁₀(Strength)", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_title("Node Strength Distribution", fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 5. Community Sizes Ranked (middle row, right)
    ax5 = fig.add_subplot(gs[1, 2])
    
    orig_sorted = sorted(orig_sizes, reverse=True)[:20]
    cons_sorted = sorted(cons_sizes, reverse=True)[:20]
    
    x_orig = np.arange(len(orig_sorted))
    x_cons = np.arange(len(cons_sorted))
    
    ax5.plot(x_orig, orig_sorted, 'o-', label='Original', color='steelblue', linewidth=2)
    ax5.plot(x_cons, cons_sorted, 's-', label='Replication', color='coral', linewidth=2)
    ax5.set_xlabel("Community Rank", fontsize=10)
    ax5.set_ylabel("Size (nodes)", fontsize=10)
    ax5.set_title("Top 20 Communities by Size", fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(alpha=0.3, linestyle='--')
    
    # 6. Top 5 nodes from each network (bottom row, spans all)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Get top 5 nodes by strength from each
    orig_top = orig_comm.nlargest(5, 'strength')[['label', 'community', 'degree', 'strength']]
    cons_top = cons_comm.nlargest(5, 'strength')[['label', 'community', 'degree', 'strength']]
    
    # Format data
    orig_data = []
    for _, row in orig_top.iterrows():
        orig_data.append([row['label'], row['community'], row['degree'], f"{row['strength']:.3f}"])
    
    cons_data = []
    for _, row in cons_top.iterrows():
        cons_data.append([row['label'], row['community'], row['degree'], f"{row['strength']:.3f}"])
    
    # Combine into one table
    combined_data = [['Rank', 'Original: Node', 'Comm', 'Deg', 'Strength', '', 
                     'Replication: Node', 'Comm', 'Deg', 'Strength']]
    
    for i in range(5):
        row = [str(i+1)] + orig_data[i] + [''] + cons_data[i]
        combined_data.append(row)
    
    table2 = ax6.table(cellText=combined_data, cellLoc='left', loc='center',
                      colWidths=[0.05, 0.22, 0.05, 0.05, 0.08, 0.02, 0.22, 0.05, 0.05, 0.08])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.5)
    
    # Color header
    for i in range(10):
        table2[(0, i)].set_facecolor('#2c3e50')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color separator column
    for i in range(6):
        table2[(i, 5)].set_facecolor('#95a5a6')
    
    # ax6.set_title("Top 5 Nodes by Strength (Network Hubs)", fontsize=12, fontweight='bold', pad=15)
    
    # Add footer note
    fig.text(0.5, 0.02, 
             "Note: Different node ID systems prevent direct node-level comparison. "
             "Structural metrics remain comparable.",
             ha='center', fontsize=9, style='italic', color='gray')
    
    # Save
    png = OUTPUT_DIR / "COMPREHENSIVE_BACKBONE_COMPARISON.png"
    pdf = OUTPUT_DIR / "COMPREHENSIVE_BACKBONE_COMPARISON.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    print("Generating comprehensive comparison figure...")
    create_summary_figure()
    print("Done!")
