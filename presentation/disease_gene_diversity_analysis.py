"""
Replicate the disease diversity vs gene diversity analysis from the article.
Compares disease node diversity/betweenness in SGPDN with corresponding 
gene diversity/betweenness in PPI network.

Data:
- SGPDN: data/replication/data4.txt (133,106 interactions, 1,596 diseases)
- Disease-gene associations: data/replication/gene_disease/gene_disease_associations.tsv
- PPI network: data/replication/ppi/ppi_interactions.tsv

Metrics:
- Node diversity: diversity of connections (Shannon entropy or similar)
- Betweenness centrality: number of shortest paths passing through node

Expected correlations:
- Node diversity: PCC=0.84, P=2.5e-10
- Betweenness: PCC=0.59, P=9.5e-7
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr, zscore
from collections import Counter

# Paths
SGPDN_PATH = Path("data/replication/data4.txt")
GENE_DISEASE_PATH = Path("data/replication/gene_disease/gene_disease_associations.tsv")
PPI_PATH = Path("data/replication/ppi/ppi_interactions.tsv")
OUTPUT_DIR = Path("presentation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sgpdn() -> nx.Graph:
    """Load the SGPDN network from data4.txt"""
    df = pd.read_csv(SGPDN_PATH, sep="\t")
    # Filter for similarity > 0.1 as mentioned (already done in data4)
    g = nx.Graph()
    for _, row in df.iterrows():
        d1, d2 = row.iloc[0], row.iloc[1]
        sim = float(row.iloc[2])
        g.add_edge(d1, d2, weight=sim)
    print(f"SGPDN: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def load_disease_genes() -> dict:
    """Load disease-gene associations, return dict of disease -> set of genes"""
    df = pd.read_csv(GENE_DISEASE_PATH, sep="\t")
    disease_genes = {}
    for _, row in df.iterrows():
        diseases = str(row["disease"]).split("|")
        gene = row["gene"]
        for d in diseases:
            d = d.strip()
            if d not in disease_genes:
                disease_genes[d] = set()
            disease_genes[d].add(gene)
    return disease_genes


def load_ppi(relevant_genes: set = None) -> nx.Graph:
    """Load the PPI network, optionally filtered to relevant genes"""
    df = pd.read_csv(PPI_PATH, sep="\t")
    
    # Filter to only disease-associated genes if provided
    if relevant_genes:
        print(f"Filtering PPI to {len(relevant_genes)} disease-associated genes...")
        df = df[
            df["protein_a"].isin(relevant_genes) & 
            df["protein_b"].isin(relevant_genes)
        ]
    
    edges = list(zip(df["protein_a"], df["protein_b"]))
    g = nx.Graph()
    g.add_edges_from(edges)
    print(f"PPI: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def compute_node_diversity_shannon(g: nx.Graph, node) -> float:
    """
    Compute node diversity as Shannon entropy of neighbor degree distribution.
    Alternative/legacy method for comparison.
    """
    if node not in g:
        return 0.0
    neighbors = list(g.neighbors(node))
    if not neighbors:
        return 0.0
    
    neighbor_degrees = [g.degree(n) for n in neighbors]
    counts = Counter(neighbor_degrees)
    total = sum(counts.values())
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)
    
    return entropy


def compute_node_diversity(g: nx.Graph, node) -> float:
    """
    Compute node diversity using the bridging coefficient formula from the paper.
    
    f(j) = Σ(i∈N(j)) d(i)/(k(i)-1)
    
    Where:
    - k(i) is the degree of node i
    - N(j) is the neighborhood of node j (set of direct neighbors)
    - d(i) is the total number of links leaving that neighborhood N(j)
    
    This diversity is large for nodes with many neighbors that themselves have 
    many out-going links.
    """
    if node not in g:
        return 0.0
    
    neighbors = list(g.neighbors(node))
    if not neighbors:
        return 0.0
    
    diversity = 0.0
    neighborhood = set(neighbors)  # N(j)
    
    for neighbor in neighbors:
        k_i = g.degree(neighbor)  # degree of neighbor
        
        # Skip if degree is 1 (would cause division by zero)
        if k_i <= 1:
            continue
        
        # Count links leaving the neighborhood N(j)
        # d(i) = number of neighbors of i that are NOT in the neighborhood N(j)
        d_i = 0
        for neighbor_of_neighbor in g.neighbors(neighbor):
            if neighbor_of_neighbor != node and neighbor_of_neighbor not in neighborhood:
                d_i += 1
        
        diversity += d_i / (k_i - 1)
    
    return diversity


def compute_max_gene_diversity(disease: str, disease_genes: dict, ppi: nx.Graph, metric="diversity") -> float:
    """
    Compute maximum diversity/betweenness of disease-related genes in PPI network
    """
    genes = disease_genes.get(disease, set())
    genes = [g for g in genes if g in ppi]
    
    if not genes:
        return np.nan
    
    if metric == "diversity":
        values = [compute_node_diversity(ppi, g) for g in genes]
    elif metric == "betweenness":
        # For efficiency: sample betweenness on subset if too many genes
        if len(genes) > 100:
            # Use approximate betweenness
            bc = nx.betweenness_centrality(ppi, k=min(len(genes), 50))
        else:
            # Full betweenness only for genes neighborhood
            subgraph_nodes = set(genes)
            for g in genes:
                if g in ppi:
                    subgraph_nodes.update(ppi.neighbors(g))
            if len(subgraph_nodes) > 5000:
                # Too large, use global approximation
                bc = nx.betweenness_centrality(ppi, k=100)
            else:
                subgraph = ppi.subgraph(subgraph_nodes)
                bc = nx.betweenness_centrality(subgraph)
        values = [bc.get(g, 0) for g in genes]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return max(values) if values else np.nan


def compute_disease_metrics(sgpdn: nx.Graph, disease_genes: dict, ppi: nx.Graph):
    """
    Compute diversity and betweenness for diseases and their corresponding genes
    Uses both Shannon entropy and bridging coefficient for node diversity
    """
    # Compute disease metrics
    print("Computing disease betweenness...")
    disease_bc = nx.betweenness_centrality(sgpdn)
    
    print("Computing disease node diversity (bridging coefficient)...")
    disease_diversity = {d: compute_node_diversity(sgpdn, d) for d in sgpdn.nodes()}
    
    print("Computing disease node diversity (Shannon entropy)...")
    disease_diversity_shannon = {d: compute_node_diversity_shannon(sgpdn, d) for d in sgpdn.nodes()}
    
    # Pre-compute PPI betweenness once for efficiency (approximate)
    print("Computing PPI betweenness (approximate for efficiency)...")
    ppi_bc = nx.betweenness_centrality(ppi, k=min(1000, ppi.number_of_nodes()//10))
    
    # For each disease, compute max gene metric
    diseases = list(sgpdn.nodes())
    
    print("Computing gene metrics for each disease...")
    results = []
    for i, disease in enumerate(diseases):
        if i % 100 == 0:
            print(f"  {i}/{len(diseases)}")
        
        # Get genes for this disease
        genes = disease_genes.get(disease, set())
        genes = [g for g in genes if g in ppi]
        
        if not genes:
            continue
        
        # Node diversity for genes (bridging coefficient)
        gene_div_vals = [compute_node_diversity(ppi, g) for g in genes]
        gene_div = max(gene_div_vals) if gene_div_vals else np.nan
        
        # Node diversity for genes (Shannon entropy)
        gene_div_shannon_vals = [compute_node_diversity_shannon(ppi, g) for g in genes]
        gene_div_shannon = max(gene_div_shannon_vals) if gene_div_shannon_vals else np.nan
        
        # Betweenness for genes (use pre-computed)
        gene_bc_vals = [ppi_bc.get(g, 0) for g in genes]
        gene_bc = max(gene_bc_vals) if gene_bc_vals else np.nan
        
        results.append({
            "disease": disease,
            "disease_diversity": disease_diversity.get(disease, np.nan),
            "disease_diversity_shannon": disease_diversity_shannon.get(disease, np.nan),
            "disease_betweenness": disease_bc.get(disease, np.nan),
            "gene_diversity": gene_div,
            "gene_diversity_shannon": gene_div_shannon,
            "gene_betweenness": gene_bc,
        })
    
    df = pd.DataFrame(results)
    # Remove rows with NaN
    df = df.dropna()
    return df


def plot_correlations(df: pd.DataFrame, n_bins=10):
    """
    Generate correlation plots with binned error bars
    Axes reversed: disease diversity on x-axis, gene diversity on y-axis
    Shows both bridging coefficient and Shannon entropy methods
    """
    # Z-score normalization
    df["disease_diversity_z"] = zscore(df["disease_diversity"])
    df["gene_diversity_z"] = zscore(df["gene_diversity"])
    df["disease_diversity_shannon_z"] = zscore(df["disease_diversity_shannon"])
    df["gene_diversity_shannon_z"] = zscore(df["gene_diversity_shannon"])
    df["disease_betweenness_z"] = zscore(df["disease_betweenness"])
    df["gene_betweenness_z"] = zscore(df["gene_betweenness"])
    
    # Debug: print summary statistics
    print("\nZ-score summary statistics:")
    print("Disease diversity z (bridging):", df["disease_diversity_z"].describe())
    print("Gene diversity z (bridging):", df["gene_diversity_z"].describe())
    print("Disease betweenness z:", df["disease_betweenness_z"].describe())
    print("Gene betweenness z:", df["gene_betweenness_z"].describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    
    # (a) Node diversity - Bridging coefficient (REVERSED AXES)
    ax = axes[0, 0]
    x = df["disease_diversity_z"].values
    y = df["gene_diversity_z"].values
    
    # Scatter
    ax.scatter(x, y, alpha=0.5, s=20, color="#5DADE2")
    
    # Binned error bars - use quantile-based bins to ensure balanced bins
    bins = pd.qcut(x, q=n_bins, duplicates='drop')
    binned = df.groupby(bins, observed=True).agg({
        "disease_diversity_z": "mean",
        "gene_diversity_z": ["mean", "std"]
    })
    bin_x = binned["disease_diversity_z"]["mean"].values
    bin_y = binned["gene_diversity_z"]["mean"].values
    bin_err = binned["gene_diversity_z"]["std"].values
    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt='o', color='red', 
                markersize=6, capsize=4, linewidth=2, label="Binned mean ± s.d.")
    
    # Correlation
    pcc, pval = pearsonr(x, y)
    ax.text(0.05, 0.95, f"PCC = {pcc:.2f}\nP = {pval:.2e}", 
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_xlabel("Disease node diversity (z-score, bridging coeff.)", fontsize=10)
    ax.set_ylabel("Gene node diversity (z-score, bridging coeff.)", fontsize=10)
    ax.set_title("(a) Node diversity correlation (bridging coefficient)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # (b) Node diversity - Shannon entropy (REVERSED AXES)
    ax = axes[0, 1]
    x = df["disease_diversity_shannon_z"].values
    y = df["gene_diversity_shannon_z"].values
    
    # Scatter
    ax.scatter(x, y, alpha=0.5, s=20, color="#F4A460")
    
    # Binned error bars
    bins = pd.qcut(x, q=n_bins, duplicates='drop')
    binned = df.groupby(bins, observed=True).agg({
        "disease_diversity_shannon_z": "mean",
        "gene_diversity_shannon_z": ["mean", "std"]
    })
    bin_x = binned["disease_diversity_shannon_z"]["mean"].values
    bin_y = binned["gene_diversity_shannon_z"]["mean"].values
    bin_err = binned["gene_diversity_shannon_z"]["std"].values
    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt='o', color='red', 
                markersize=6, capsize=4, linewidth=2, label="Binned mean ± s.d.")
    
    # Correlation
    pcc, pval = pearsonr(x, y)
    ax.text(0.05, 0.95, f"PCC = {pcc:.2f}\nP = {pval:.2e}", 
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_xlabel("Disease node diversity (z-score, Shannon entropy)", fontsize=10)
    ax.set_ylabel("Gene node diversity (z-score, Shannon entropy)", fontsize=10)
    ax.set_title("(b) Node diversity correlation (Shannon entropy)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # (c) Betweenness (REVERSED AXES)
    ax = axes[1, 0]
    x = df["disease_betweenness_z"].values
    y = df["gene_betweenness_z"].values
    
    # Scatter
    ax.scatter(x, y, alpha=0.5, s=20, color="#58D68D")
    
    # Binned error bars - use quantile-based bins
    bins = pd.qcut(x, q=n_bins, duplicates='drop')
    binned = df.groupby(bins, observed=True).agg({
        "disease_betweenness_z": "mean",
        "gene_betweenness_z": ["mean", "std"]
    })
    bin_x = binned["disease_betweenness_z"]["mean"].values
    bin_y = binned["gene_betweenness_z"]["mean"].values
    bin_err = binned["gene_betweenness_z"]["std"].values
    ax.errorbar(bin_x, bin_y, yerr=bin_err, fmt='o', color='red', 
                markersize=6, capsize=4, linewidth=2, label="Binned mean ± s.d.")
    
    # Correlation
    pcc, pval = pearsonr(x, y)
    ax.text(0.05, 0.95, f"PCC = {pcc:.2f}\nP = {pval:.2e}", 
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax.set_xlabel("Disease betweenness (z-score)", fontsize=10)
    ax.set_ylabel("Gene betweenness (z-score)", fontsize=10)
    ax.set_title("(c) Betweenness correlation", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # (d) Summary text / info panel
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
METHODOLOGY

Node Diversity Measures:
• Bridging coefficient (new): f(j) = Σ(i∈N(j)) d(i)/(k(i)-1)
  Quantifies connectedness and "bridge-like" property
  
• Shannon entropy (previous): -Σ p(k) log(p(k))
  Measures disorder in neighbor degree distribution

Betweenness Centrality:
• Number of shortest paths passing through node
• Indicates influence/control over information flow

PPI Network:
• Max value across disease-associated genes
• ~4,600 genes, ~118k interactions

SGPDN:
• 1,596 diseases, 133k interactions
• Combined symptom + gene/PPI evidence
    
N = {len(df)} diseases with complete metrics
    """
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='lightyellow', alpha=0.7))
    
    fig.suptitle("Disease diversity vs. gene diversity in PPI network (SGPDN analysis)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "disease_gene_diversity_correlation.png"
    pdf = OUTPUT_DIR / "disease_gene_diversity_correlation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    print("Loading networks...")
    sgpdn = load_sgpdn()
    disease_genes = load_disease_genes()
    
    print(f"\nDisease-gene mappings: {len(disease_genes)} diseases")
    
    # Extract all unique genes from disease associations
    all_genes = set()
    for genes in disease_genes.values():
        all_genes.update(genes)
    print(f"Total unique disease-associated genes: {len(all_genes)}")
    
    # Load PPI filtered to disease-associated genes only
    ppi = load_ppi(relevant_genes=all_genes)
    
    print("\nComputing diversity metrics...")
    df = compute_disease_metrics(sgpdn, disease_genes, ppi)
    
    print(f"\nAnalysis ready: {len(df)} diseases with complete metrics")
    
    # Save results
    out_tsv = OUTPUT_DIR / "disease_gene_diversity_metrics.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved metrics: {out_tsv}")
    
    print("\nGenerating plots...")
    plot_correlations(df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
