"""
Detailed diagnosis of overlap matching between two backbones.
Extracts and examines overlapping edges and nodes.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

OUTPUT_DIR = Path("presentation/plots/backbone")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all data"""
    # Load original backbone
    df_orig = pd.read_csv("data/replication/filtering/disease_network_backbone.csv")
    df_orig = df_orig.rename(columns={"Weight": "weight", "Alpha": "alpha"})
    
    # Load name mapping
    df_edges = pd.read_csv("data/replication/consensus_symptom_geneppi_edges.tsv", sep="\t")
    name_map = {}
    for _, row in df_edges.iterrows():
        if pd.notna(row['source_name']):
            name_map[row['Source']] = str(row['source_name']).strip()
        if pd.notna(row['target_name']):
            name_map[row['Target']] = str(row['target_name']).strip()
    
    # Load and map consensus backbone
    df_cons = pd.read_csv("data/replication/consensus_symptom_geneppi_backbone.tsv", sep="\t")
    df_cons['Source'] = df_cons['Source'].map(lambda x: name_map.get(x, x))
    df_cons['Target'] = df_cons['Target'].map(lambda x: name_map.get(x, x))
    
    return df_orig, df_cons, name_map


def normalize_name(name):
    """Normalize disease name for matching"""
    return str(name).strip().lower()


def analyze_matching():
    """Detailed analysis of matching logic"""
    df_orig, df_cons, name_map = load_data()
    
    print("="*80)
    print("DETAILED MATCHING DIAGNOSIS")
    print("="*80)
    
    # Check original nodes
    orig_sources = set(df_orig['Source'].unique())
    orig_targets = set(df_orig['Target'].unique())
    orig_nodes = orig_sources | orig_targets
    
    # Check consensus nodes
    cons_sources = set(df_cons['Source'].unique())
    cons_targets = set(df_cons['Target'].unique())
    cons_nodes = cons_sources | cons_targets
    
    print(f"\n1. NODES:")
    print(f"   Original: {len(orig_nodes)} unique nodes")
    print(f"   Consensus: {len(cons_nodes)} unique nodes")
    
    # Sample nodes
    print(f"\n   Sample Original nodes:")
    for node in sorted(list(orig_nodes))[:5]:
        print(f"      '{node}'")
    
    print(f"\n   Sample Consensus nodes (after mapping):")
    for node in sorted(list(cons_nodes))[:5]:
        print(f"      '{node}'")
    
    # Check for exact string matches
    exact_match = orig_nodes & cons_nodes
    print(f"\n   Exact string matches: {len(exact_match)}")
    
    # Check for normalized matches
    orig_norm = {normalize_name(n): n for n in orig_nodes}
    cons_norm = {normalize_name(n): n for n in cons_nodes}
    
    norm_match = set(orig_norm.keys()) & set(cons_norm.keys())
    print(f"   Normalized matches: {len(norm_match)}")
    
    # Check edges
    print(f"\n2. EDGES:")
    
    # Exact edge matches
    orig_edges_exact = set()
    for _, row in df_orig.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        orig_edges_exact.add(edge)
    
    cons_edges_exact = set()
    for _, row in df_cons.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        cons_edges_exact.add(edge)
    
    exact_edge_overlap = orig_edges_exact & cons_edges_exact
    
    print(f"   Original edges: {len(orig_edges_exact)}")
    print(f"   Consensus edges: {len(cons_edges_exact)}")
    print(f"   Exact edge matches: {len(exact_edge_overlap)}")
    
    # Normalized edge matches
    orig_edges_norm = set()
    for _, row in df_orig.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        orig_edges_norm.add(edge)
    
    cons_edges_norm = set()
    for _, row in df_cons.iterrows():
        edge = frozenset([normalize_name(row['Source']), normalize_name(row['Target'])])
        cons_edges_norm.add(edge)
    
    norm_edge_overlap = orig_edges_norm & cons_edges_norm
    
    print(f"   Normalized edge matches: {len(norm_edge_overlap)}")
    
    # Show sample overlaps
    print(f"\n   Sample overlapping edges:")
    for i, edge in enumerate(list(exact_edge_overlap)[:5]):
        nodes = list(edge)
        print(f"      {i+1}. {nodes[0]} <-> {nodes[1]}")
    
    # Check for unmapped consensus nodes
    unmapped = sum(1 for node in cons_nodes if node in name_map.values() and node not in orig_nodes)
    print(f"\n3. MAPPING QUALITY:")
    print(f"   Consensus nodes found in original: {len(cons_nodes & orig_nodes)} / {len(cons_nodes)}")
    
    # Analyze why nodes don't match
    print(f"\n4. SAMPLING NON-MATCHING NODES:")
    non_match_cons = list(cons_nodes - orig_nodes)[:10]
    print(f"   Consensus nodes NOT in original ({len(non_match_cons)} samples):")
    for node in non_match_cons:
        norm_node = normalize_name(node)
        matches_in_orig = [n for n in orig_nodes if normalize_name(n) == norm_node]
        if matches_in_orig:
            print(f"      '{node}' -> normalized match: '{matches_in_orig[0]}'")
        else:
            print(f"      '{node}' -> NO normalized match found")
    
    non_match_orig = list(orig_nodes - cons_nodes)[:10]
    print(f"\n   Original nodes NOT in consensus ({len(non_match_orig)} samples):")
    for node in non_match_orig:
        norm_node = normalize_name(node)
        matches_in_cons = [n for n in cons_nodes if normalize_name(n) == norm_node]
        if matches_in_cons:
            print(f"      '{node}' -> normalized match: '{matches_in_cons[0]}'")
        else:
            print(f"      '{node}' -> NO normalized match found")
    
    return df_orig, df_cons, exact_edge_overlap, norm_edge_overlap, orig_nodes, cons_nodes


def extract_overlapping_network(df_orig, df_cons, exact_overlap_edges):
    """Extract and save overlapping network"""
    print("\n5. EXTRACTING OVERLAPPING NETWORK:")
    
    # Get overlapping edges with both metadata
    overlap_orig_data = []
    overlap_cons_data = []
    
    for _, row in df_orig.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        if edge in exact_overlap_edges:
            overlap_orig_data.append(row)
    
    for _, row in df_cons.iterrows():
        edge = frozenset([row['Source'], row['Target']])
        if edge in exact_overlap_edges:
            overlap_cons_data.append(row)
    
    df_overlap_orig = pd.DataFrame(overlap_orig_data)
    df_overlap_cons = pd.DataFrame(overlap_cons_data)
    
    # Save overlapping edges
    orig_overlap_file = OUTPUT_DIR / "overlapping_edges_original.tsv"
    cons_overlap_file = OUTPUT_DIR / "overlapping_edges_consensus.tsv"
    
    df_overlap_orig.to_csv(orig_overlap_file, sep='\t', index=False)
    df_overlap_cons.to_csv(cons_overlap_file, sep='\t', index=False)
    
    print(f"   Saved {len(df_overlap_orig)} overlapping edges from original to {orig_overlap_file}")
    print(f"   Saved {len(df_overlap_cons)} overlapping edges from consensus to {cons_overlap_file}")
    
    # Extract overlapping nodes
    overlap_nodes = set()
    for _, row in df_overlap_orig.iterrows():
        overlap_nodes.add(row['Source'])
        overlap_nodes.add(row['Target'])
    
    # Build overlapping network
    G_overlap = nx.Graph()
    for _, row in df_overlap_orig.iterrows():
        G_overlap.add_edge(row['Source'], row['Target'], 
                          weight=row['weight'], 
                          orig_alpha=row.get('alpha', 0))
    
    # Add metadata from consensus
    for _, row in df_overlap_cons.iterrows():
        if G_overlap.has_edge(row['Source'], row['Target']):
            G_overlap[row['Source']][row['Target']]['cons_weight'] = row['weight']
    
    # Network stats
    print(f"\n   Overlapping network stats:")
    print(f"      Nodes: {G_overlap.number_of_nodes()}")
    print(f"      Edges: {G_overlap.number_of_edges()}")
    print(f"      Density: {nx.density(G_overlap):.4f}")
    print(f"      Connected components: {nx.number_connected_components(G_overlap)}")
    
    # Find hubs (high degree nodes)
    degrees = dict(G_overlap.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n   Top 10 hubs (by degree):")
    for node, degree in top_hubs:
        print(f"      {node}: degree {degree}")
    
    # Save network info
    network_info_file = OUTPUT_DIR / "overlapping_network_info.txt"
    with open(network_info_file, 'w') as f:
        f.write(f"Overlapping Network Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Nodes: {G_overlap.number_of_nodes()}\n")
        f.write(f"Edges: {G_overlap.number_of_edges()}\n")
        f.write(f"Density: {nx.density(G_overlap):.4f}\n")
        f.write(f"Connected Components: {nx.number_connected_components(G_overlap)}\n")
        f.write(f"\nTop Hub Nodes (by degree):\n")
        for node, degree in top_hubs:
            f.write(f"  {node}: {degree}\n")
    
    print(f"   Saved network info to {network_info_file}")
    
    # Save nodes list
    nodes_file = OUTPUT_DIR / "overlapping_nodes.tsv"
    nodes_df = pd.DataFrame({
        'node': sorted(list(overlap_nodes)),
        'degree': [degrees.get(n, 0) for n in sorted(list(overlap_nodes))]
    })
    nodes_df.to_csv(nodes_file, sep='\t', index=False)
    print(f"   Saved {len(nodes_df)} overlapping nodes to {nodes_file}")
    
    return G_overlap, overlap_nodes


def main():
    df_orig, df_cons, exact_overlap, norm_overlap, orig_nodes, cons_nodes = analyze_matching()
    
    print("\n" + "="*80)
    print("JACCARD INDICES:")
    print("="*80)
    orig_edges = set(frozenset([r['Source'], r['Target']]) for _, r in df_orig.iterrows())
    cons_edges = set(frozenset([r['Source'], r['Target']]) for _, r in df_cons.iterrows())
    
    print(f"Edge Jaccard (exact):      {len(exact_overlap) / len(orig_edges | cons_edges):.4f}")
    print(f"Edge Jaccard (normalized): {len(norm_overlap) / len(orig_edges | cons_edges):.4f}")
    print(f"Node Jaccard (exact):      {len(orig_nodes & cons_nodes) / len(orig_nodes | cons_nodes):.4f}")
    
    # Extract overlapping network
    print("\n" + "="*80)
    G_overlap, overlap_nodes = extract_overlapping_network(df_orig, df_cons, exact_overlap)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
