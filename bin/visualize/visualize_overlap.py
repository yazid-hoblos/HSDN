"""
Create visualizations of the overlapping network and comparison.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

OUTPUT_DIR = Path("presentation/plots/backbone")


def visualize_overlapping_network():
    """Create a visual representation of the overlapping network"""
    # Load data
    nodes_df = pd.read_csv(OUTPUT_DIR / "overlapping_nodes.tsv", sep='\t')
    edges_orig = pd.read_csv(OUTPUT_DIR / "overlapping_edges_original.tsv", sep='\t')
    
    # Build graph
    G = nx.Graph()
    for _, row in edges_orig.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['weight'])
    
    # Get network layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Node sizes by degree
    sizes = [G.degree(n) * 200 for n in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue', 
                           edgecolors='black', linewidths=1.5, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, ax=ax)
    
    # Draw labels for high-degree nodes (hubs)
    threshold = 3
    hub_nodes = {n: n for n in G.nodes() if G.degree(n) >= threshold}
    nx.draw_networkx_labels(G, pos, labels=hub_nodes, font_size=8, 
                           font_weight='bold', ax=ax)
    
    ax.set_title("Overlapping Disease Network\n(115 shared edges between Original and Consensus backbones)", 
                fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "overlapping_network_visualization.png"
    pdf = OUTPUT_DIR / "overlapping_network_visualization.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def create_overlap_summary_table():
    """Create a summary table comparing the networks"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Load all relevant data
    orig_stats = pd.read_csv(OUTPUT_DIR / "backbone_comparison_network_stats.tsv", sep='\t')
    overlap_stats = pd.read_csv(OUTPUT_DIR / "backbone_comparison_overlap_stats.tsv", sep='\t')
    
    summary_data = [
        ['Metric', 'Original Backbone', 'Consensus GenePPI', 'Overlap'],
        ['',  '', '', ''],
        ['Total Nodes', '1,035', '618', '315 (30.4%)'],
        ['Total Edges', '2,199', '1,722', '115 (5.2%)'],
        ['Avg Degree', '4.25', '5.57', '~1.74'],
        ['Network Density', '0.00411', '0.00903', '0.0133'],
        ['',  '', '', ''],
        ['Avg Edge Weight', '0.776', '0.467', 'N/A'],
        ['Avg Clustering', '0.389', '0.231', 'N/A'],
        ['Connected Components', '63', '32', '35'],
        ['',  '', '', ''],
        ['Nodes in Common', '315 / 1,035 (30.4%)', '315 / 618 (51.0%)', '-'],
        ['Edges in Common', '115 / 2,199 (5.2%)', '115 / 1,722 (6.7%)', '-'],
        ['Jaccard Index (nodes)', '-', '-', '0.2354'],
        ['Jaccard Index (edges)', '-', '-', '0.0302'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color separator rows
    for j in range(4):
        table[(1, j)].set_facecolor('#ecf0f1')
        table[(6, j)].set_facecolor('#ecf0f1')
        table[(9, j)].set_facecolor('#ecf0f1')
    
    # Alternate row colors
    for i in range(len(summary_data)):
        if i not in [0, 1, 6, 9]:  # Skip header and separator rows
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    ax.set_title("Network Comparison Summary", fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "overlap_summary_table.png"
    pdf = OUTPUT_DIR / "overlap_summary_table.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def create_venn_style_comparison():
    """Create a Venn-style comparison of nodes and edges"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Nodes comparison
    ax = axes[0]
    ax.text(0.5, 0.85, "Node Overlap Analysis", ha='center', fontsize=13, fontweight='bold',
            transform=ax.transAxes)
    
    node_text = """
    Original Nodes:          1,035
    Consensus Nodes:           618
    ─────────────────────────────
    Overlapping Nodes:         315  (Jaccard: 0.2354)
    
    Original Only:             720  (69.6%)
    Consensus Only:            303  (49.0%)
    
    Key Insight:
    51% of consensus nodes are in original,
    but only 30% of original nodes in consensus.
    Consensus has different disease focus.
    """
    
    ax.text(0.05, 0.5, node_text, transform=ax.transAxes, fontsize=10,
            family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.axis('off')
    
    # Edges comparison
    ax = axes[1]
    ax.text(0.5, 0.85, "Edge Overlap Analysis", ha='center', fontsize=13, fontweight='bold',
            transform=ax.transAxes)
    
    edge_text = """
    Original Edges:          2,199
    Consensus Edges:         1,722
    ─────────────────────────────
    Overlapping Edges:         115  (Jaccard: 0.0302)
    
    Original Only:           2,084  (94.8%)
    Consensus Only:          1,607  (93.3%)
    
    Key Insight:
    Low edge overlap (3%) despite 30% node overlap.
    Networks connect same diseases very differently.
    Both represent valid but distinct disease relationships.
    """
    
    ax.text(0.05, 0.5, edge_text, transform=ax.transAxes, fontsize=10,
            family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.axis('off')
    
    fig.suptitle("Network Comparison: Overlap Statistics", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "overlap_venn_style.png"
    pdf = OUTPUT_DIR / "overlap_venn_style.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    print("Generating overlap visualizations...\n")
    
    print("1. Creating overlapping network visualization...")
    visualize_overlapping_network()
    
    print("\n2. Creating summary table...")
    create_overlap_summary_table()
    
    print("\n3. Creating Venn-style comparison...")
    create_venn_style_comparison()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
