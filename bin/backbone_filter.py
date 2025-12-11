import pandas as pd
import networkx as nx
from pathlib import Path
import numpy as np
from scipy import stats

# ---------------------------
# Config
# ---------------------------
INPUT_FILE = Path("data/article_data/data4.txt")
OUTPUT_EDGELIST = Path("data/disease_network_edgelist.csv")
OUTPUT_BACKBONE = Path("data/disease_network_backbone.csv")
ALPHA = 0.05  # significance level for backbone filtering

# ---------------------------
# 1. Load and standardize edge list
# ---------------------------
def load_and_convert_edgelist(input_path, output_path):
    """
    Load edge list and convert to standard Gephi format.
    Standard format: Source,Target,Weight
    """
    print(f"Loading edge list from {input_path}...")
    df = pd.read_csv(input_path, sep='\t')
    print(f"Loaded {len(df)} edges")
    
    # Rename columns to Gephi standard
    df_gephi = df.rename(columns={
        df.columns[0]: 'Source',
        df.columns[1]: 'Target',
        df.columns[2]: 'Weight'
    })
    
    # Save as CSV
    df_gephi.to_csv(output_path, index=False)
    print(f"Saved standardized edge list to {output_path}")
    print(f"  Format: {output_path.suffix} with columns: Source, Target, Weight")
    
    return df_gephi

# ---------------------------
# 2. Multiscale Backbone Filtering
# ---------------------------
def disparity_filter(G, alpha=0.05):
    """
    Apply disparity filter (Serrano et al. 2009) for weighted network backbone.
    
    For each node i, for each edge (i,j) with weight w_ij:
    - Compute normalized weight: p_ij = w_ij / sum(w_ik for all k)
    - Compute significance using binomial distribution
    - Keep edge if significant at level alpha
    
    Reference: 
    Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009). 
    Extracting the multiscale backbone of complex weighted networks. 
    PNAS, 106(16), 6483-6488.
    """
    print("\nApplying multiscale backbone filtering (disparity filter)...")
    print(f"  Alpha = {alpha}")
    
    # Create backbone graph
    backbone = nx.Graph()
    edges_kept = 0
    edges_tested = 0
    
    for node in G.nodes():
        # Get all edges incident to this node
        neighbors = list(G.neighbors(node))
        if len(neighbors) <= 1:
            # Keep all edges if degree <= 1
            for neighbor in neighbors:
                if not backbone.has_edge(node, neighbor):
                    backbone.add_edge(node, neighbor, weight=G[node][neighbor]['weight'])
                    edges_kept += 1
            continue
        
        # Calculate normalized weights
        k = len(neighbors)  # degree
        total_weight = sum(G[node][neighbor]['weight'] for neighbor in neighbors)
        
        for neighbor in neighbors:
            edges_tested += 1
            w = G[node][neighbor]['weight']
            p = w / total_weight  # normalized weight
            
            # Disparity filter: alpha_ij = (1 - p)^(k-1)
            # Keep edge if alpha_ij < alpha
            alpha_ij = (1 - p) ** (k - 1)
            
            if alpha_ij < alpha:
                # Edge is significant, add to backbone
                if not backbone.has_edge(node, neighbor):
                    backbone.add_edge(node, neighbor, weight=w, alpha=alpha_ij)
                    edges_kept += 1
    
    print(f"  Original edges: {G.number_of_edges()}")
    print(f"  Backbone edges: {backbone.number_of_edges()}")
    print(f"  Reduction: {(1 - backbone.number_of_edges()/G.number_of_edges())*100:.1f}%")
    
    return backbone

# ---------------------------
# 3. Export backbone
# ---------------------------
def export_backbone(G, output_path):
    """Export backbone network to CSV edge list."""
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            'Source': u,
            'Target': v,
            'Weight': data['weight'],
            'Alpha': data.get('alpha', np.nan)
        })
    
    df = pd.DataFrame(edges)
    df.to_csv(output_path, index=False)
    print(f"\nSaved backbone to {output_path}")
    print(f"  Columns: Source, Target, Weight, Alpha")

# ---------------------------
# Main
# ---------------------------
def main():
    print("="*60)
    print("Disease Network Standardization & Backbone Filtering")
    print("="*60)
    
    # Step 1: Load and standardize
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        return
    
    df_edges = load_and_convert_edgelist(INPUT_FILE, OUTPUT_EDGELIST)
    
    # Step 2: Build NetworkX graph
    print("\nBuilding network graph...")
    G = nx.from_pandas_edgelist(
        df_edges,
        source='Source',
        target='Target',
        edge_attr=['Weight'],
        create_using=nx.Graph()
    )
    
    # Ensure weight attribute is lowercase 'weight' for NetworkX conventions
    for u, v, data in G.edges(data=True):
        if 'Weight' in data and 'weight' not in data:
            data['weight'] = data['Weight']
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Step 3: Apply backbone filtering
    backbone = disparity_filter(G, alpha=ALPHA)
    
    # Step 4: Export backbone
    export_backbone(backbone, OUTPUT_BACKBONE)
    
    # Optional: Export backbone as GEXF too
    nx.write_gexf(backbone, OUTPUT_BACKBONE.with_suffix('.gexf'))
    print(f"Also saved as {OUTPUT_BACKBONE.with_suffix('.gexf')}")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Standardized edge list: {OUTPUT_EDGELIST}")
    print(f"  Backbone (α={ALPHA}): {OUTPUT_BACKBONE}")
    print(f"  Both can be opened in Gephi")
    print("="*60)

if __name__ == "__main__":
    main()
