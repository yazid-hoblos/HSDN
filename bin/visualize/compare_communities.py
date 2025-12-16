"""
Compare communities between two backbone networks.

Inputs:
- Original backbone communities: presentation/plots/original_backbone_communities.tsv
- Consensus genePPI communities: presentation/plots/consensus_geneppi_communities.tsv

Outputs:
- Community overlap analysis plots
- Top nodes per community comparison
- Modularity and community quality metrics
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import chi2_contingency
import networkx as nx

OUTPUT_DIR = Path("presentation/plots/backbone")


def load_communities(filepath):
    """Load community assignments"""
    return pd.read_csv(filepath, sep='\t')


def compute_community_stats(df, label):
    """Compute statistics for each community"""
    stats = []
    for comm_id in sorted(df['community'].unique()):
        if comm_id == -1:  # Skip unassigned nodes if any
            continue
        
        comm_nodes = df[df['community'] == comm_id]
        stats.append({
            'Network': label,
            'Community': comm_id,
            'Size': len(comm_nodes),
            'Avg_Degree': comm_nodes['degree'].mean(),
            'Avg_Strength': comm_nodes['strength'].mean(),
            'Total_Strength': comm_nodes['strength'].sum(),
            'Top_Node': comm_nodes.iloc[0]['label'] if len(comm_nodes) > 0 else 'N/A',
        })
    
    return pd.DataFrame(stats)


def plot_community_sizes(df_orig, df_cons):
    """Compare community size distributions"""
    orig_sizes = df_orig.groupby('community').size().values
    cons_sizes = df_cons.groupby('community').size().values
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot comparison
    ax = axes[0]
    x_orig = np.arange(len(orig_sizes))
    x_cons = np.arange(len(cons_sizes))
    
    ax.bar(x_orig, sorted(orig_sizes, reverse=True), alpha=0.7, label='Original', color='steelblue')
    ax.bar(x_cons, sorted(cons_sizes, reverse=True), alpha=0.7, label='Consensus GenePPI', color='coral')
    ax.set_xlabel("Community Rank (by size)", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_title("Community Size Distribution", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Histogram
    ax = axes[1]
    ax.hist(orig_sizes, bins=20, alpha=0.6, label='Original', color='steelblue', edgecolor='black')
    ax.hist(cons_sizes, bins=20, alpha=0.6, label='Consensus GenePPI', color='coral', edgecolor='black')
    ax.set_xlabel("Community Size", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Community Size Histogram", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "community_comparison_sizes.png"
    pdf = OUTPUT_DIR / "community_comparison_sizes.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def compute_node_overlap(df_orig, df_cons):
    """Compute overlap of nodes between networks using normalized labels"""
    # Normalize function
    def normalize_name(name):
        return str(name).strip().lower()
    
    # Use labels (disease names) instead of node_ids for comparison
    orig_nodes = set(normalize_name(label) for label in df_orig['label'])
    cons_nodes = set(normalize_name(label) for label in df_cons['label'])
    
    overlap = orig_nodes & cons_nodes
    orig_only = orig_nodes - cons_nodes
    cons_only = cons_nodes - orig_nodes
    
    return {
        'Total_Original_Nodes': len(orig_nodes),
        'Total_Consensus_Nodes': len(cons_nodes),
        'Overlapping_Nodes': len(overlap),
        'Original_Only': len(orig_only),
        'Consensus_Only': len(cons_only),
        'Jaccard_Index': len(overlap) / len(orig_nodes | cons_nodes) if orig_nodes | cons_nodes else 0,
    }, overlap


def analyze_community_overlap(df_orig, df_cons, overlap_nodes):
    """Analyze how overlapping nodes are distributed across communities"""
    # Normalize function
    def normalize_name(name):
        return str(name).strip().lower()
    
    # Filter to only overlapping nodes (using normalized labels)
    df_orig_overlap = df_orig[df_orig['label'].apply(normalize_name).isin(overlap_nodes)]
    df_cons_overlap = df_cons[df_cons['label'].apply(normalize_name).isin(overlap_nodes)]
    
    # Create mapping using normalized labels
    orig_comm_map = dict(zip(df_orig_overlap['label'].apply(normalize_name), df_orig_overlap['community']))
    cons_comm_map = dict(zip(df_cons_overlap['label'].apply(normalize_name), df_cons_overlap['community']))
    
    # Build contingency table
    orig_comms = sorted(df_orig_overlap['community'].unique())
    cons_comms = sorted(df_cons_overlap['community'].unique())
    
    contingency = np.zeros((len(orig_comms), len(cons_comms)))
    
    for node in overlap_nodes:
        if node in orig_comm_map and node in cons_comm_map:
            orig_idx = orig_comms.index(orig_comm_map[node])
            cons_idx = cons_comms.index(cons_comm_map[node])
            contingency[orig_idx, cons_idx] += 1
    
    return contingency, orig_comms, cons_comms


def plot_community_overlap_heatmap(contingency, orig_comms, cons_comms):
    """Plot heatmap of community overlap"""
    # Handle case where there's no overlap
    if contingency.size == 0 or contingency.sum() == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, 
                "No Node Overlap Between Networks\n\n"
                "The two networks use different node identifiers:\n"
                "• Original: Disease names\n"
                "• Consensus: MeSH IDs\n\n"
                "Cannot compute direct community overlap.\n"
                "See other plots for structural comparisons.",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title("Community Overlap Analysis", fontsize=12, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(contingency, annot=True, fmt='.0f', cmap='YlOrRd', 
                    xticklabels=cons_comms, yticklabels=orig_comms,
                    cbar_kws={'label': 'Number of Shared Nodes'}, ax=ax)
        ax.set_xlabel("Consensus GenePPI Community", fontsize=11)
        ax.set_ylabel("Original Backbone Community", fontsize=11)
        ax.set_title("Community Overlap: Shared Node Distribution", fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "community_comparison_overlap_heatmap.png"
    pdf = OUTPUT_DIR / "community_comparison_overlap_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_top_nodes_per_community(df_orig, df_cons, top_n=5):
    """Display top nodes per community for both networks"""
    orig_comms = sorted([c for c in df_orig['community'].unique() if c != -1])
    cons_comms = sorted([c for c in df_cons['community'].unique() if c != -1])
    
    # Get top nodes for each community
    orig_top = {}
    for comm in orig_comms[:10]:  # Limit to first 10 communities for readability
        top_nodes = df_orig[df_orig['community'] == comm].nlargest(top_n, 'strength')
        orig_top[comm] = list(top_nodes['label'])
    
    cons_top = {}
    for comm in cons_comms[:10]:
        top_nodes = df_cons[df_cons['community'] == comm].nlargest(top_n, 'strength')
        cons_top[comm] = list(top_nodes['label'])
    
    # Create comparison table
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    for ax, top_dict, label in zip(axes, [orig_top, cons_top], ['Original Backbone', 'Consensus GenePPI']):
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for comm_id, nodes in sorted(top_dict.items()):
            table_data.append([f"Community {comm_id}"] + nodes[:5])
        
        # Pad to ensure all rows have same length
        max_cols = max(len(row) for row in table_data) if table_data else 1
        for row in table_data:
            while len(row) < max_cols:
                row.append('')
        
        if table_data:
            table = ax.table(cellText=table_data, 
                           colLabels=['Community'] + [f'Top {i+1}' for i in range(max_cols-1)],
                           cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Color header
            for i in range(max_cols):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(f"{label}\nTop {top_n} Nodes per Community (by strength)", 
                    fontsize=11, fontweight='bold', pad=20)
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "community_comparison_top_nodes.png"
    pdf = OUTPUT_DIR / "community_comparison_top_nodes.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def save_community_summary(stats_orig, stats_cons, overlap_stats):
    """Save community statistics summary"""
    # Save detailed stats
    all_stats = pd.concat([stats_orig, stats_cons], ignore_index=True)
    stats_file = OUTPUT_DIR / "community_comparison_detailed_stats.tsv"
    all_stats.to_csv(stats_file, sep='\t', index=False)
    print(f"Saved: {stats_file}")
    
    # Save overlap stats
    overlap_df = pd.DataFrame([overlap_stats])
    overlap_file = OUTPUT_DIR / "community_comparison_node_overlap.tsv"
    overlap_df.to_csv(overlap_file, sep='\t', index=False)
    print(f"Saved: {overlap_file}")
    
    # Create visual summary
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Original Backbone', 'Consensus GenePPI'],
        ['Number of Communities', 
         stats_orig['Community'].nunique(), 
         stats_cons['Community'].nunique()],
        ['Total Nodes', 
         overlap_stats['Total_Original_Nodes'], 
         overlap_stats['Total_Consensus_Nodes']],
        ['Overlapping Nodes', 
         overlap_stats['Overlapping_Nodes'], 
         overlap_stats['Overlapping_Nodes']],
        ['Largest Community Size', 
         stats_orig['Size'].max(), 
         stats_cons['Size'].max()],
        ['Smallest Community Size', 
         stats_orig['Size'].min(), 
         stats_cons['Size'].min()],
        ['Avg Community Size', 
         f"{stats_orig['Size'].mean():.1f}", 
         f"{stats_cons['Size'].mean():.1f}"],
        ['Jaccard Index (Nodes)', 
         f"{overlap_stats['Jaccard_Index']:.4f}", 
         f"{overlap_stats['Jaccard_Index']:.4f}"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title("Community Detection Summary", fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    
    png = OUTPUT_DIR / "community_comparison_summary.png"
    pdf = OUTPUT_DIR / "community_comparison_summary.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    print("=" * 70)
    print("COMMUNITY COMPARISON: Original vs Consensus GenePPI")
    print("=" * 70)
    
    orig_file = OUTPUT_DIR / "original_backbone_communities.tsv"
    cons_file = OUTPUT_DIR / "consensus_geneppi_communities.tsv"
    
    if not orig_file.exists():
        print(f"\nERROR: Original communities file not found: {orig_file}")
        print("Run plot_backbone_pyvis.py first to generate community assignments.")
        return
    
    if not cons_file.exists():
        print(f"\nERROR: Consensus communities file not found: {cons_file}")
        print("Run plot_backbone_pyvis.py first to generate community assignments.")
        return
    
    print("\n1. Loading community assignments...")
    df_orig = load_communities(orig_file)
    df_cons = load_communities(cons_file)
    
    print(f"   Original: {len(df_orig)} nodes, {df_orig['community'].nunique()} communities")
    print(f"   Consensus: {len(df_cons)} nodes, {df_cons['community'].nunique()} communities")
    
    print("\n2. Computing community statistics...")
    stats_orig = compute_community_stats(df_orig, "Original")
    stats_cons = compute_community_stats(df_cons, "Consensus GenePPI")
    
    print("\n3. Analyzing node overlap...")
    overlap_stats, overlap_nodes = compute_node_overlap(df_orig, df_cons)
    print(f"   Overlapping nodes: {overlap_stats['Overlapping_Nodes']}")
    print(f"   Jaccard index: {overlap_stats['Jaccard_Index']:.4f}")
    
    print("\n4. Analyzing community overlap...")
    contingency, orig_comms, cons_comms = analyze_community_overlap(df_orig, df_cons, overlap_nodes)
    
    print("\n5. Generating comparison plots...")
    plot_community_sizes(df_orig, df_cons)
    plot_community_overlap_heatmap(contingency, orig_comms, cons_comms)
    plot_top_nodes_per_community(df_orig, df_cons, top_n=5)
    
    print("\n6. Saving summary statistics...")
    save_community_summary(stats_orig, stats_cons, overlap_stats)
    
    print("\n" + "=" * 70)
    print("DONE! Check presentation/plots/ for all outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
