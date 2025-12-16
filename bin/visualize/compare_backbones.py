"""
Compare two backbone networks: original (from paper replication) vs consensus genePPI.

Inputs:
- Original backbone: data/replication/filtering/disease_network_backbone.csv
- Consensus genePPI backbone: data/replication/consensus_symptom_geneppi_backbone.tsv
- Consensus edges (for name mapping): data/replication/consensus_symptom_geneppi_edges.tsv

Outputs:
- Comparison plots: presentation/plots/backbone_comparison_*.png/pdf
- Statistics table: presentation/plots/backbone_comparison_stats.tsv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr, spearmanr

OUTPUT_DIR = Path("presentation/plots/backbone")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_original_backbone():
    """Load the original backbone from paper replication"""
    df = pd.read_csv("data/replication/filtering/disease_network_backbone.csv")
    # Standardize columns
    df = df.rename(columns={"Weight": "weight", "Alpha": "alpha"})
    return df


def load_consensus_backbone():
    """Load the consensus genePPI backbone and map MeSH IDs to disease names"""
    df = pd.read_csv("consensus_symptom_geneppi_backbone.tsv", sep="\t")
    
    # Load name mapping
    name_map = load_name_map()
    
    # Map Source and Target from MeSH IDs to disease names
    df['Source'] = df['Source'].map(lambda x: name_map.get(x, x))
    df['Target'] = df['Target'].map(lambda x: name_map.get(x, x))
    
    print(f"   Mapped consensus network: {df['Source'].nunique()} unique sources, {df['Target'].nunique()} unique targets")
    print(f"   Sample mapped edge: {df.iloc[0]['Source']} <-> {df.iloc[0]['Target']}")
    
    return df


def load_name_map():
    """Load disease name mapping from consensus edges"""
    df = pd.read_csv("consensus_symptom_geneppi_edges.tsv", sep="\t")
    name_map = {}
    if 'source_name' in df.columns and 'target_name' in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row['source_name']):
                name_map[row['Source']] = str(row['source_name']).strip()
            if pd.notna(row['target_name']):
                name_map[row['Target']] = str(row['target_name']).strip()
    
    print(f"   Loaded name mapping: {len(name_map)} MeSH ID -> disease name mappings")
    return name_map


def compute_network_stats(df, label):
    """Compute basic network statistics"""
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', 
                                 edge_attr='weight', create_using=nx.Graph())
    
    stats = {
        "Network": label,
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Avg Degree": np.mean([d for n, d in G.degree()]),
        "Avg Weight": df['weight'].mean(),
        "Median Weight": df['weight'].median(),
        "Weight Std": df['weight'].std(),
        "Connected Components": nx.number_connected_components(G),
        "Largest CC Size": len(max(nx.connected_components(G), key=len)),
    }
    
    # Clustering coefficient (can be slow for large networks)
    if G.number_of_nodes() < 5000:
        stats["Avg Clustering"] = nx.average_clustering(G)
    else:
        stats["Avg Clustering"] = "N/A (too large)"
    
    return stats, G


def compare_edge_overlap(df_orig, df_cons, name_map=None):
    """Compare edge overlap between the two networks"""
    # Normalize disease names (case-insensitive, strip whitespace)
    def normalize_name(name):
        return str(name).strip().lower()
    
    # Create edge sets with normalized names
    orig_edges = set()
    for _, row in df_orig.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        orig_edges.add(edge)
    
    cons_edges = set()
    for _, row in df_cons.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        cons_edges.add(edge)
    
    # Overlap analysis
    overlap = orig_edges & cons_edges
    orig_only = orig_edges - cons_edges
    cons_only = cons_edges - orig_edges
    
    results = {
        "Total Original Edges": len(orig_edges),
        "Total Consensus Edges": len(cons_edges),
        "Overlapping Edges": len(overlap),
        "Original Only": len(orig_only),
        "Consensus Only": len(cons_only),
        "Jaccard Index": len(overlap) / len(orig_edges | cons_edges) if orig_edges | cons_edges else 0,
    }
    
    return results, overlap, orig_only, cons_only


def plot_weight_comparison(df_orig, df_cons, overlap_edges):
    """Compare edge weights for overlapping edges"""
    # Normalize names function
    def normalize_name(name):
        return str(name).strip().lower()
    
    # Build weight dictionaries with normalized names
    orig_weights = {}
    for _, row in df_orig.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        orig_weights[edge] = row['weight']
    
    cons_weights = {}
    for _, row in df_cons.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        cons_weights[edge] = row['weight']
    
    # Get overlapping edge weights
    overlap_orig = [orig_weights[e] for e in overlap_edges if e in orig_weights]
    overlap_cons = [cons_weights[e] for e in overlap_edges if e in cons_weights]
    
    if len(overlap_orig) == 0:
        print("No overlapping edges found for weight comparison")
        return
    
    # Compute correlation
    pcc, pcc_pval = pearsonr(overlap_orig, overlap_cons)
    scc, scc_pval = spearmanr(overlap_orig, overlap_cons)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(overlap_orig, overlap_cons, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.set_xlabel("Original Backbone Weight", fontsize=11)
    ax.set_ylabel("Consensus GenePPI Backbone Weight", fontsize=11)
    ax.set_title("Edge Weight Comparison (Overlapping Edges)", fontsize=12, fontweight='bold')
    
    # Add diagonal reference line
    min_w = min(min(overlap_orig), min(overlap_cons))
    max_w = max(max(overlap_orig), max(overlap_cons))
    ax.plot([min_w, max_w], [min_w, max_w], 'r--', linewidth=2, alpha=0.7, label='y=x')
    
    # Add stats
    ax.text(0.05, 0.95, f"Pearson r = {pcc:.3f} (P={pcc_pval:.2e})\n"
                        f"Spearman ρ = {scc:.3f} (P={scc_pval:.2e})\n"
                        f"n = {len(overlap_orig)} edges",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Distribution comparison
    ax = axes[1]
    ax.hist(overlap_orig, bins=30, alpha=0.6, label='Original', color='steelblue', edgecolor='black')
    ax.hist(overlap_cons, bins=30, alpha=0.6, label='Consensus GenePPI', color='coral', edgecolor='black')
    ax.set_xlabel("Edge Weight", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Weight Distribution (Overlapping Edges)", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "backbone_comparison_weights.png"
    pdf = OUTPUT_DIR / "backbone_comparison_weights.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_degree_distribution(G_orig, G_cons):
    """Compare degree distributions"""
    orig_degrees = [d for n, d in G_orig.degree()]
    cons_degrees = [d for n, d in G_cons.degree()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(orig_degrees, bins=50, alpha=0.6, label='Original', color='steelblue', edgecolor='black')
    ax.hist(cons_degrees, bins=50, alpha=0.6, label='Consensus GenePPI', color='coral', edgecolor='black')
    ax.set_xlabel("Degree", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Degree Distribution Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Log-log plot
    ax = axes[1]
    orig_hist, orig_bins = np.histogram(orig_degrees, bins=50)
    cons_hist, cons_bins = np.histogram(cons_degrees, bins=50)
    
    orig_bin_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
    cons_bin_centers = (cons_bins[:-1] + cons_bins[1:]) / 2
    
    # Filter zeros for log
    orig_mask = orig_hist > 0
    cons_mask = cons_hist > 0
    
    ax.loglog(orig_bin_centers[orig_mask], orig_hist[orig_mask], 'o-', 
              label='Original', color='steelblue', alpha=0.7)
    ax.loglog(cons_bin_centers[cons_mask], cons_hist[cons_mask], 's-', 
              label='Consensus GenePPI', color='coral', alpha=0.7)
    ax.set_xlabel("Degree (log)", fontsize=11)
    ax.set_ylabel("Frequency (log)", fontsize=11)
    ax.set_title("Degree Distribution (Log-Log)", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "backbone_comparison_degree.png"
    pdf = OUTPUT_DIR / "backbone_comparison_degree.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_venn_diagram(overlap_stats):
    """Create a simple text-based Venn diagram representation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create text summary
    text_content = f"""
    Edge Overlap Analysis
    ══════════════════════════════════════════════════
    
    Original Backbone Edges:          {overlap_stats['Total Original Edges']:,}
    Consensus GenePPI Backbone Edges: {overlap_stats['Total Consensus Edges']:,}
    
    Overlapping Edges:                {overlap_stats['Overlapping Edges']:,}
    Original Only:                    {overlap_stats['Original Only']:,}
    Consensus Only:                   {overlap_stats['Consensus Only']:,}
    
    Jaccard Index:                    {overlap_stats['Jaccard Index']:.4f}
    
    Overlap Percentage (Original):   {100 * overlap_stats['Overlapping Edges'] / overlap_stats['Total Original Edges']:.2f}%
    Overlap Percentage (Consensus):  {100 * overlap_stats['Overlapping Edges'] / overlap_stats['Total Consensus Edges']:.2f}%
    """
    
    ax.text(0.5, 0.5, text_content, transform=ax.transAxes, fontsize=11,
            ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "backbone_comparison_overlap.png"
    pdf = OUTPUT_DIR / "backbone_comparison_overlap.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def save_statistics_table(stats_orig, stats_cons, overlap_stats):
    """Save comprehensive statistics as TSV"""
    # Combine network stats
    stats_df = pd.DataFrame([stats_orig, stats_cons])
    
    # Add overlap stats as separate section
    overlap_df = pd.DataFrame([{
        "Metric": k,
        "Value": v
    } for k, v in overlap_stats.items()])
    
    # Save both
    stats_file = OUTPUT_DIR / "backbone_comparison_network_stats.tsv"
    overlap_file = OUTPUT_DIR / "backbone_comparison_overlap_stats.tsv"
    
    stats_df.to_csv(stats_file, sep="\t", index=False)
    overlap_df.to_csv(overlap_file, sep="\t", index=False)
    
    print(f"Saved: {stats_file}")
    print(f"Saved: {overlap_file}")
    
    # Also create combined visual table
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Network stats table
    ax = axes[0]
    ax.axis('tight')
    ax.axis('off')
    table1 = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                      cellLoc='left', loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2)
    for i in range(len(stats_df.columns)):
        table1[(0, i)].set_facecolor('#40466e')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    ax.set_title("Network Statistics Comparison", fontsize=12, fontweight='bold', pad=20)
    
    # Overlap stats table
    ax = axes[1]
    ax.axis('tight')
    ax.axis('off')
    table2 = ax.table(cellText=overlap_df.values, colLabels=overlap_df.columns,
                      cellLoc='left', loc='center', colWidths=[0.7, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2)
    for i in range(len(overlap_df.columns)):
        table2[(0, i)].set_facecolor('#40466e')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    ax.set_title("Edge Overlap Statistics", fontsize=12, fontweight='bold', pad=20)
    
    fig.tight_layout()
    
    png = OUTPUT_DIR / "backbone_comparison_summary_tables.png"
    pdf = OUTPUT_DIR / "backbone_comparison_summary_tables.pdf"
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    print("=" * 70)
    print("BACKBONE COMPARISON: Original vs Consensus GenePPI")
    print("=" * 70)
    
    print("\n1. Loading data...")
    name_map = load_name_map()
    df_orig = load_original_backbone()
    df_cons = load_consensus_backbone()
    
    print(f"   Original backbone: {len(df_orig)} edges")
    print(f"   Consensus backbone: {len(df_cons)} edges")
    
    print("\n2. Computing network statistics...")
    stats_orig, G_orig = compute_network_stats(df_orig, "Original")
    stats_cons, G_cons = compute_network_stats(df_cons, "Consensus GenePPI")
    
    print("\n   Original Network:")
    for k, v in stats_orig.items():
        if k != "Network":
            print(f"      {k}: {v}")
    
    print("\n   Consensus GenePPI Network:")
    for k, v in stats_cons.items():
        if k != "Network":
            print(f"      {k}: {v}")
    
    print("\n3. Analyzing edge overlap...")
    overlap_stats, overlap_edges, orig_only, cons_only = compare_edge_overlap(df_orig, df_cons, name_map)
    
    print(f"   Overlapping edges: {overlap_stats['Overlapping Edges']}")
    print(f"   Jaccard index: {overlap_stats['Jaccard Index']:.4f}")
    print(f"   Original only: {overlap_stats['Original Only']}")
    print(f"   Consensus only: {overlap_stats['Consensus Only']}")
    
    print("\n4. Generating comparison plots...")
    plot_venn_diagram(overlap_stats)
    plot_weight_comparison(df_orig, df_cons, overlap_edges)
    plot_degree_distribution(G_orig, G_cons)
    
    print("\n5. Saving statistics tables...")
    save_statistics_table(stats_orig, stats_cons, overlap_stats)
    
    print("\n" + "=" * 70)
    print("DONE! Check presentation/plots/ for all outputs")
    print("=" * 70)


if __name__ == "__main__":
    main()
