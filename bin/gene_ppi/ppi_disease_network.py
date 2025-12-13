"""
PPI-Based Disease Network Construction

This script constructs disease networks based on Protein-Protein Interaction (PPI) data.
Two diseases are connected if:
1. They share 1st order PPI neighbors (proteins that directly interact)
2. They share 2nd order PPI neighbors (proteins connected by 1 intermediate protein)

Reference:
- 1st order PPI: Direct protein interactions
- 2nd order PPI: Interactions at distance 2 in the PPI network

Requires:
- Gene-disease associations
- PPI network data
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
import time


def load_ppi_network(ppi_file):
    """
    Load PPI data and create a NetworkX graph.
    
    Parameters:
    -----------
    ppi_file : str or Path
        Path to PPI interaction file
        
    Returns:
    --------
    G : networkx.Graph
        PPI network
    """
    print("="*60)
    print("LOADING PPI NETWORK")
    print("="*60)
    
    df = pd.read_csv(ppi_file, sep='\t')
    print(f"\nLoaded {len(df)} interactions from {ppi_file}")
    
    # Create graph
    G = nx.Graph()
    
    for _, row in df.iterrows():
        protein_a = str(row['protein_a']).strip()
        protein_b = str(row['protein_b']).strip()
        
        if protein_a and protein_b:
            G.add_edge(protein_a, protein_b)
    
    print(f"PPI Network:")
    print(f"  Nodes (proteins): {G.number_of_nodes()}")
    print(f"  Edges (interactions): {G.number_of_edges()}")
    
    # Degree statistics
    degrees = dict(G.degree())
    print(f"  Mean degree: {np.mean(list(degrees.values())):.2f}")
    print(f"  Max degree: {max(degrees.values())}")
    
    return G


def load_gene_disease_associations(gene_disease_file):
    """
    Load gene-disease associations.
    
    Parameters:
    -----------
    gene_disease_file : str or Path
        Path to gene-disease association file
        
    Returns:
    --------
    associations : dict
        Mapping from disease to list of genes
    """
    print("\n" + "="*60)
    print("LOADING GENE-DISEASE ASSOCIATIONS")
    print("="*60)
    
    df = pd.read_csv(gene_disease_file, sep='\t')
    print(f"\nLoaded {len(df)} associations from {gene_disease_file}")
    
    # Create disease-to-genes mapping
    associations = defaultdict(set)
    
    for _, row in df.iterrows():
        disease = str(row['disease']).strip()
        gene = str(row['gene']).strip()
        
        if disease and gene:
            associations[disease].add(gene)
    
    print(f"  Unique diseases: {len(associations)}")
    print(f"  Unique genes: {len(set().union(*associations.values()))}")
    
    # Average genes per disease
    avg_genes = np.mean([len(genes) for genes in associations.values()])
    print(f"  Mean genes per disease: {avg_genes:.2f}")
    
    return associations


def get_1st_order_ppi_neighbors(genes, ppi_graph):
    """
    Get 1st order PPI neighbors of a set of genes/proteins.
    
    1st order = direct neighbors in PPI network
    
    Parameters:
    -----------
    genes : set or list
        Set of gene/protein names
    ppi_graph : networkx.Graph
        PPI network
        
    Returns:
    --------
    neighbors : set
        All 1st order neighbors
    """
    neighbors = set()
    
    for gene in genes:
        if gene in ppi_graph:
            neighbors.update(ppi_graph.neighbors(gene))
    
    return neighbors


def get_2nd_order_ppi_neighbors(genes, ppi_graph):
    """
    Get 2nd order PPI neighbors of a set of genes/proteins.
    
    2nd order = neighbors at distance 2 in PPI network
    
    Parameters:
    -----------
    genes : set or list
        Set of gene/protein names
    ppi_graph : networkx.Graph
        PPI network
        
    Returns:
    --------
    neighbors : set
        All 2nd order neighbors
    """
    neighbors = set()
    
    for gene in genes:
        if gene in ppi_graph:
            # Get 1st order neighbors
            first_order = set(ppi_graph.neighbors(gene))
            # Get neighbors of 1st order neighbors (2nd order)
            for neighbor in first_order:
                neighbors.update(ppi_graph.neighbors(neighbor))
    
    # Remove the original genes themselves
    neighbors.difference_update(genes)
    
    return neighbors


# Global variable for worker processes (avoids pickling large data)
_global_disease_ppi_data = None
_global_diseases = None

def _init_worker(disease_ppi_data, diseases):
    """Initialize worker process with shared data."""
    global _global_disease_ppi_data, _global_diseases
    _global_disease_ppi_data = disease_ppi_data
    _global_diseases = diseases

def _process_disease_pair_batch(batch_indices):
    """
    Process a batch of disease pairs (for parallel execution).
    Uses global variables to avoid memory duplication.
    
    Parameters:
    -----------
    batch_indices : list of tuples
        List of (i, j) disease index pairs
    
    Returns:
    --------
    edges_1st, edges_2nd : list of tuples
        Edge lists for 1st and 2nd order networks
    """
    edges_1st = []
    edges_2nd = []
    
    for i, j in batch_indices:
        disease_i = _global_diseases[i]
        disease_j = _global_diseases[j]
        
        # 1st order shared neighbors
        shared_1st = (_global_disease_ppi_data[disease_i]['ppi_1st_neighbors'] & 
                     _global_disease_ppi_data[disease_j]['ppi_1st_neighbors'])
        
        if shared_1st:
            edges_1st.append((disease_i, disease_j, len(shared_1st)))
        
        # 2nd order shared neighbors
        shared_2nd = (_global_disease_ppi_data[disease_i]['ppi_2nd_neighbors'] & 
                     _global_disease_ppi_data[disease_j]['ppi_2nd_neighbors'])
        
        if shared_2nd:
            edges_2nd.append((disease_i, disease_j, len(shared_2nd)))
    
    return edges_1st, edges_2nd


def build_ppi_disease_networks(associations, ppi_graph, n_processes=None):
    """
    Build disease networks based on 1st and 2nd order PPI interactions.
    
    Parameters:
    -----------
    associations : dict
        Disease to genes mapping
    ppi_graph : networkx.Graph
        PPI network
    n_processes : int, optional
        Number of parallel processes (default: CPU count)
        
    Returns:
    --------
    G_1st : networkx.Graph
        Disease network based on 1st order PPI
    G_2nd : networkx.Graph
        Disease network based on 2nd order PPI
    disease_ppi_data : dict
        Additional data about PPI neighbors
    """
    print("\n" + "="*60)
    print("BUILDING PPI-BASED DISEASE NETWORKS")
    print("="*60)
    
    diseases = list(associations.keys())
    n_diseases = len(diseases)
    
    print(f"\nProcessing {n_diseases} diseases...")
    
    # Create networks
    G_1st = nx.Graph()
    G_2nd = nx.Graph()
    
    # Add all diseases as nodes
    G_1st.add_nodes_from(diseases)
    G_2nd.add_nodes_from(diseases)
    
    # Store PPI neighbor data
    disease_ppi_data = {}
    
    # Calculate PPI neighbors for each disease
    for i, disease in enumerate(diseases):
        genes = associations[disease]
        
        # Get PPI neighbors
        ppi_1st = get_1st_order_ppi_neighbors(genes, ppi_graph)
        ppi_2nd = get_2nd_order_ppi_neighbors(genes, ppi_graph)
        
        disease_ppi_data[disease] = {
            'genes': genes,
            'ppi_1st_neighbors': ppi_1st,
            'ppi_2nd_neighbors': ppi_2nd,
        }
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_diseases} diseases")
    
    total_pairs = n_diseases * (n_diseases - 1) // 2
    
    print("\nComputing disease-disease connections...")
    print(f"Total pairs to compare: {total_pairs:,}")
    
    start_time = time.time()
    
    # Use parallelization if requested and viable
    if n_processes and n_processes > 1:
        print(f"Using {n_processes} processes (controlled parallelization)")
        
        # Generate all pair indices
        all_pairs = [(i, j) for i in range(n_diseases) for j in range(i + 1, n_diseases)]
        
        # Larger batches to reduce overhead
        batch_size = max(50000, total_pairs // (n_processes * 10))
        batches = [all_pairs[i:i + batch_size] for i in range(0, len(all_pairs), batch_size)]
        
        print(f"Processing {len(batches)} batches of ~{batch_size:,} pairs each")
        
        # Process with limited workers
        with Pool(processes=n_processes, initializer=_init_worker, 
                  initargs=(disease_ppi_data, diseases)) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(_process_disease_pair_batch, batches)):
                results.append(result)
                
                pairs_so_far = min((i + 1) * batch_size, total_pairs)
                if (i + 1) % max(1, len(batches) // 20) == 0 or i == len(batches) - 1:
                    elapsed = time.time() - start_time
                    rate = pairs_so_far / elapsed if elapsed > 0 else 0
                    remaining = (total_pairs - pairs_so_far) / rate if rate > 0 else 0
                    print(f"  Progress: {pairs_so_far:,}/{total_pairs:,} ({100*pairs_so_far/total_pairs:.1f}%) - "
                          f"Rate: {rate:,.0f} pairs/s - ETA: {remaining/60:.1f} min")
        
        # Merge results
        print("Merging results...")
        for edges_1st, edges_2nd in results:
            for disease_i, disease_j, shared_count in edges_1st:
                G_1st.add_edge(disease_i, disease_j, shared_proteins=shared_count)
            for disease_i, disease_j, shared_count in edges_2nd:
                G_2nd.add_edge(disease_i, disease_j, shared_proteins=shared_count)
    
    else:
        # Sequential processing (memory-safe)
        print("Using sequential processing (memory-safe)")
        pairs_processed = 0
        
        for i in range(n_diseases):
            for j in range(i + 1, n_diseases):
                disease_i = diseases[i]
                disease_j = diseases[j]
                
                # 1st order shared neighbors
                shared_1st = (disease_ppi_data[disease_i]['ppi_1st_neighbors'] & 
                             disease_ppi_data[disease_j]['ppi_1st_neighbors'])
                
                if shared_1st:
                    G_1st.add_edge(disease_i, disease_j, shared_proteins=len(shared_1st))
                
                # 2nd order shared neighbors
                shared_2nd = (disease_ppi_data[disease_i]['ppi_2nd_neighbors'] & 
                             disease_ppi_data[disease_j]['ppi_2nd_neighbors'])
                
                if shared_2nd:
                    G_2nd.add_edge(disease_i, disease_j, shared_proteins=len(shared_2nd))
                
                pairs_processed += 1
                
                # Progress every 1M pairs
                if pairs_processed % 1_000_000 == 0:
                    elapsed = time.time() - start_time
                    rate = pairs_processed / elapsed
                    remaining = (total_pairs - pairs_processed) / rate
                    print(f"  Progress: {pairs_processed:,}/{total_pairs:,} ({100*pairs_processed/total_pairs:.1f}%) - "
                          f"Rate: {rate:,.0f} pairs/s - ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed/60:.1f} minutes ({total_pairs/elapsed:,.0f} pairs/sec)")
    
    print(f"\n1st order PPI disease network:")
    print(f"  Nodes: {G_1st.number_of_nodes()}")
    print(f"  Edges: {G_1st.number_of_edges()}")
    if G_1st.number_of_edges() > 0:
        print(f"  Density: {nx.density(G_1st):.6f}")
    
    print(f"\n2nd order PPI disease network:")
    print(f"  Nodes: {G_2nd.number_of_nodes()}")
    print(f"  Edges: {G_2nd.number_of_edges()}")
    if G_2nd.number_of_edges() > 0:
        print(f"  Density: {nx.density(G_2nd):.6f}")
    
    return G_1st, G_2nd, disease_ppi_data


def export_ppi_disease_network(G, disease_ppi_data, output_path, order=1):
    """
    Export PPI-based disease network as edgelist.
    
    Parameters:
    -----------
    G : networkx.Graph
        Disease network
    disease_ppi_data : dict
        PPI neighbor data for diseases
    output_path : str or Path
        Output file path
    order : int
        1 or 2 (for naming purposes)
    """
    print(f"\nExporting {order}-order PPI disease network...")
    
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            'disease1': u,
            'disease2': v,
            'shared_proteins': data.get('shared_proteins', 0)
        })
    
    edgelist = pd.DataFrame(edges)
    edgelist = edgelist.sort_values('shared_proteins', ascending=False)
    
    edgelist.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved to: {output_path}")
    print(f"  Edges: {len(edgelist)}")
    
    if len(edgelist) > 0:
        print(f"  Shared proteins - Mean: {edgelist['shared_proteins'].mean():.2f}, "
              f"Max: {edgelist['shared_proteins'].max()}")
    
    return edgelist


def export_disease_ppi_nodes(disease_ppi_data, output_path, order=1):
    """
    Export disease node data including PPI information.
    
    Parameters:
    -----------
    disease_ppi_data : dict
        PPI neighbor data for diseases
    output_path : str or Path
        Output file path
    order : int
        1 or 2 (for naming purposes)
    """
    print(f"\nExporting {order}-order PPI disease nodes...")
    
    node_data = []
    for disease, data in disease_ppi_data.items():
        if order == 1:
            num_ppi = len(data['ppi_1st_neighbors'])
        else:
            num_ppi = len(data['ppi_2nd_neighbors'])
        
        node_data.append({
            'disease': disease,
            'num_genes': len(data['genes']),
            f'num_ppi_{order}st_order': num_ppi if order == 1 else num_ppi,
        })
    
    df = pd.DataFrame(node_data).sort_values(f'num_ppi_{order}st_order', ascending=False)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"  Saved to: {output_path}")


def calculate_ppi_network_statistics(G_1st, G_2nd):
    """
    Calculate network statistics.
    
    Parameters:
    -----------
    G_1st : networkx.Graph
        1st order disease network
    G_2nd : networkx.Graph
        2nd order disease network
    """
    print("\n" + "="*60)
    print("PPI-BASED DISEASE NETWORK STATISTICS")
    print("="*60)
    
    for G, order in [(G_1st, "1st"), (G_2nd, "2nd")]:
        print(f"\n{order} Order PPI Network:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        
        if G.number_of_edges() == 0:
            print("  No edges in this network")
            continue
        
        print(f"  Density: {nx.density(G):.6f}")
        
        # Degree statistics
        degrees = dict(G.degree())
        print(f"  Mean degree: {np.mean(list(degrees.values())):.2f}")
        print(f"  Max degree: {max(degrees.values())}")
        
        # Connected components
        components = list(nx.connected_components(G))
        print(f"  Connected components: {len(components)}")
        if len(components) > 0:
            largest_cc = max(components, key=len)
            print(f"  Largest component size: {len(largest_cc)} "
                  f"({100*len(largest_cc)/G.number_of_nodes():.1f}%)")


def main():
    """Main execution."""
    
    print("="*60)
    print("PPI-BASED DISEASE NETWORK GENERATOR")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    
    gene_disease_file = data_dir / 'gene_disease_associations_cleaned.tsv'
    ppi_file = data_dir / 'ppi_interactions.tsv'
    
    # Check input files
    if not gene_disease_file.exists():
        print(f"\nError: Gene-disease file not found: {gene_disease_file}")
        print("Run: python bin/clean_gene_disease_data.py")
        return
    
    if not ppi_file.exists():
        print(f"\nError: PPI file not found: {ppi_file}")
        print("Run: python bin/download_ppi_data.py")
        return
    
    # Load data
    ppi_graph = load_ppi_network(ppi_file)
    associations = load_gene_disease_associations(gene_disease_file)
    
    # Optional: Filter diseases with too few genes (speeds up computation)
    print("\n" + "="*60)
    print("FILTERING OPTIONS")
    print("="*60)
    print(f"Current: {len(associations)} diseases = {len(associations)*(len(associations)-1)//2:,} pairs")
    print("\nTo reduce computation time, you can filter diseases by minimum gene count:")
    print("  min_genes=2: Keep diseases with ≥2 genes")
    print("  min_genes=3: Keep diseases with ≥3 genes (recommended)")
    print("  min_genes=5: Keep diseases with ≥5 genes (faster)")
    
    try:
        min_genes_input = input("\nEnter minimum genes per disease (press Enter to skip): ").strip()
        if min_genes_input:
            min_genes = int(min_genes_input)
            filtered = {d: g for d, g in associations.items() if len(g) >= min_genes}
            print(f"Filtered: {len(associations)} → {len(filtered)} diseases")
            print(f"Pairs: {len(associations)*(len(associations)-1)//2:,} → {len(filtered)*(len(filtered)-1)//2:,}")
            associations = filtered
    except (ValueError, EOFError, KeyboardInterrupt):
        print("Skipping filter, using all diseases")
    
    # Ask about parallelization
    print("\n" + "="*60)
    print("PARALLELIZATION OPTIONS")
    print("="*60)
    print("Parallel processing speeds up computation but uses more memory.")
    print("Recommendations:")
    print("  2-4 processes: Moderate speedup, safe memory usage")
    print("  0 or 1: Sequential (slowest, safest)")
    print("  >4: Faster but may cause out-of-memory")
    
    n_procs = None
    try:
        proc_input = input("\nEnter number of processes (0/1 for sequential, or press Enter for sequential): ").strip()
        if proc_input and int(proc_input) > 1:
            n_procs = int(proc_input)
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    
    # Build networks
    G_1st, G_2nd, disease_ppi_data = build_ppi_disease_networks(associations, ppi_graph, n_processes=n_procs)
    
    # Export networks
    output_1st = data_dir / 'ppi_1st_order_disease_network.tsv'
    output_2nd = data_dir / 'ppi_2nd_order_disease_network.tsv'
    
    export_ppi_disease_network(G_1st, disease_ppi_data, output_1st, order=1)
    export_ppi_disease_network(G_2nd, disease_ppi_data, output_2nd, order=2)
    
    # Export node data
    nodes_1st = data_dir / 'ppi_1st_order_disease_nodes.tsv'
    nodes_2nd = data_dir / 'ppi_2nd_order_disease_nodes.tsv'
    
    export_disease_ppi_nodes(disease_ppi_data, nodes_1st, order=1)
    export_disease_ppi_nodes(disease_ppi_data, nodes_2nd, order=2)
    
    # Statistics
    calculate_ppi_network_statistics(G_1st, G_2nd)
    
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"\n1st Order PPI Disease Networks:")
    print(f"  Edgelist: {output_1st}")
    print(f"  Nodes: {nodes_1st}")
    print(f"\n2nd Order PPI Disease Networks:")
    print(f"  Edgelist: {output_2nd}")
    print(f"  Nodes: {nodes_2nd}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
