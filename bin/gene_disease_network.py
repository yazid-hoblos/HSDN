"""
Gene-Based Disease Network Generator

This script constructs a Human Disease Network (HDN) where two diseases are connected
if they share at least one associated gene. This follows the methodology from Goh et al. (2007).

The network is built from disease-gene associations obtained from genotype-phenotype databases
such as OMIM Morbid Map, PharmGKB, and GAD.

Reference:
Goh et al. (2007) "The human disease network"
Two diseases are connected if they share an associated gene.
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from pathlib import Path
import requests
import json


def load_gene_disease_associations(filepath):
    """
    Load gene-disease associations from a file.
    
    Expected format:
    - Tab or comma separated
    - Columns: disease, gene (or similar names)
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the gene-disease association file
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with disease-gene associations
    """
    filepath = Path(filepath)
    
    # Try different separators
    try:
        df = pd.read_csv(filepath, sep='\t')
    except:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Could not read file {filepath}: {e}")
    
    print(f"Loaded {len(df)} disease-gene associations")
    print(f"Columns: {list(df.columns)}")
    
    return df


def prepare_gene_disease_data(df, disease_col=None, gene_col=None):
    """
    Prepare and standardize gene-disease association data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw gene-disease association data
    disease_col : str, optional
        Name of the disease column
    gene_col : str, optional
        Name of the gene column
        
    Returns:
    --------
    associations : pandas.DataFrame
        Standardized DataFrame with columns: disease, gene
    """
    # Auto-detect column names if not provided
    if disease_col is None:
        possible_disease_cols = ['disease', 'Disease', 'DISEASE', 'disease_name', 
                                 'disorder', 'Disorder', 'phenotype', 'Phenotype']
        for col in possible_disease_cols:
            if col in df.columns:
                disease_col = col
                break
        if disease_col is None:
            disease_col = df.columns[0]
            print(f"Using '{disease_col}' as disease column")
    
    if gene_col is None:
        possible_gene_cols = ['gene', 'Gene', 'GENE', 'gene_symbol', 'gene_name',
                             'GeneName', 'GeneSymbol']
        for col in possible_gene_cols:
            if col in df.columns:
                gene_col = col
                break
        if gene_col is None:
            gene_col = df.columns[1]
            print(f"Using '{gene_col}' as gene column")
    
    # Create standardized DataFrame
    associations = df[[disease_col, gene_col]].copy()
    associations.columns = ['disease', 'gene']
    
    # Remove missing values
    associations = associations.dropna()
    
    # Remove duplicates
    associations = associations.drop_duplicates()
    
    print(f"\nStandardized data:")
    print(f"  Unique diseases: {associations['disease'].nunique()}")
    print(f"  Unique genes: {associations['gene'].nunique()}")
    print(f"  Total associations: {len(associations)}")
    
    return associations


def build_gene_based_disease_network(associations):
    """
    Build a disease-disease network based on shared genes.
    
    Two diseases are connected if they share at least one associated gene.
    
    Parameters:
    -----------
    associations : pandas.DataFrame
        DataFrame with columns: disease, gene
        
    Returns:
    --------
    G : networkx.Graph
        Disease network where edges represent shared genes
    gene_disease_map : dict
        Mapping from genes to lists of associated diseases
    """
    print("\nBuilding gene-based disease network...")
    
    # Create gene-to-diseases mapping
    gene_disease_map = defaultdict(list)
    for _, row in associations.iterrows():
        gene_disease_map[row['gene']].append(row['disease'])
    
    # Create disease network
    G = nx.Graph()
    
    # Add all diseases as nodes
    all_diseases = associations['disease'].unique()
    G.add_nodes_from(all_diseases)
    
    # Connect diseases that share genes
    edges_with_genes = []
    
    for gene, diseases in gene_disease_map.items():
        # For each gene, connect all pairs of diseases that share it
        for i in range(len(diseases)):
            for j in range(i + 1, len(diseases)):
                disease1 = diseases[i]
                disease2 = diseases[j]
                
                # Add edge or increment weight if edge already exists
                if G.has_edge(disease1, disease2):
                    G[disease1][disease2]['weight'] += 1
                    G[disease1][disease2]['shared_genes'].append(gene)
                else:
                    G.add_edge(disease1, disease2, weight=1, shared_genes=[gene])
    
    print(f"Network created:")
    print(f"  Nodes (diseases): {G.number_of_nodes()}")
    print(f"  Edges (disease pairs): {G.number_of_edges()}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    return G, gene_disease_map


def export_edgelist(G, output_path):
    """
    Export disease network as an edgelist.
    
    Parameters:
    -----------
    G : networkx.Graph
        Disease network
    output_path : str or Path
        Path to save the edgelist
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            'disease1': u,
            'disease2': v,
            'shared_genes_count': data['weight'],
            'shared_genes': ';'.join(data['shared_genes'])
        })
    
    edgelist = pd.DataFrame(edges)
    edgelist = edgelist.sort_values('shared_genes_count', ascending=False)
    
    edgelist.to_csv(output_path, sep='\t', index=False)
    print(f"\nEdgelist saved to: {output_path}")
    print(f"  Total edges: {len(edgelist)}")
    print(f"  Max shared genes: {edgelist['shared_genes_count'].max()}")
    print(f"  Mean shared genes: {edgelist['shared_genes_count'].mean():.2f}")
    
    return edgelist


def calculate_network_statistics(G):
    """
    Calculate comprehensive network statistics.
    
    Parameters:
    -----------
    G : networkx.Graph
        Disease network
    """
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.6f}")
    
    # Degree statistics
    degrees = dict(G.degree())
    print(f"\nDegree statistics:")
    print(f"  Mean: {np.mean(list(degrees.values())):.2f}")
    print(f"  Median: {np.median(list(degrees.values())):.2f}")
    print(f"  Max: {max(degrees.values())}")
    print(f"  Min: {min(degrees.values())}")
    
    # Connected components
    components = list(nx.connected_components(G))
    print(f"\nConnected components: {len(components)}")
    if len(components) > 0:
        largest_cc = max(components, key=len)
        print(f"Largest component size: {len(largest_cc)} ({100*len(largest_cc)/G.number_of_nodes():.1f}%)")
    
    # Weight statistics
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    print(f"\nShared genes statistics:")
    print(f"  Mean: {np.mean(weights):.2f}")
    print(f"  Median: {np.median(weights):.2f}")
    print(f"  Max: {max(weights)}")
    
    # Top disease pairs by shared genes
    print(f"\nTop 10 disease pairs by shared genes:")
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    for i, (u, v, data) in enumerate(edges_sorted[:10], 1):
        print(f"  {i}. {u} -- {v}: {data['weight']} genes")


def download_omim_data(output_path):
    """
    Placeholder function to download OMIM Morbid Map data.
    
    NOTE: OMIM requires registration and API key.
    This is a template - you need to implement actual download logic.
    """
    print("\n" + "="*60)
    print("OMIM DATA DOWNLOAD")
    print("="*60)
    print("OMIM requires registration and API key from:")
    print("https://www.omim.org/api")
    print("\nPlease download 'morbidmap.txt' manually and place it in:")
    print(f"  {output_path}")
    print("="*60)


def download_pharmgkb_data(output_path):
    """
    Placeholder function to download PharmGKB data.
    
    NOTE: PharmGKB data requires registration.
    This is a template - you need to implement actual download logic.
    """
    print("\n" + "="*60)
    print("PHARMGKB DATA DOWNLOAD")
    print("="*60)
    print("PharmGKB data can be downloaded from:")
    print("https://www.pharmgkb.org/downloads")
    print("\nLook for 'diseases.zip' and 'genes.zip' files.")
    print("="*60)


def create_example_data():
    """
    Create example gene-disease association data for testing.
    
    Returns:
    --------
    df : pandas.DataFrame
        Example gene-disease associations
    """
    # Example data (small sample)
    data = {
        'disease': [
            'Alzheimer Disease', 'Alzheimer Disease', 'Alzheimer Disease',
            'Parkinson Disease', 'Parkinson Disease',
            'Breast Cancer', 'Breast Cancer', 'Ovarian Cancer',
            'Diabetes Type 2', 'Diabetes Type 2', 'Obesity',
            'Hypertension', 'Coronary Artery Disease',
        ],
        'gene': [
            'APP', 'APOE', 'PSEN1',
            'SNCA', 'LRRK2',
            'BRCA1', 'BRCA2', 'BRCA1',
            'TCF7L2', 'PPARG', 'PPARG',
            'AGT', 'APOE',
        ]
    }
    return pd.DataFrame(data)


def main():
    """Main execution function."""
    
    print("="*60)
    print("GENE-BASED DISEASE NETWORK GENERATOR")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    
    # Check if user has gene-disease data
    print("\nLooking for gene-disease association data...")
    
    # Option 1: Try cleaned data first, then raw data
    user_file = data_dir / 'gene_disease_associations_cleaned.tsv'
    raw_file = data_dir / 'gene_disease_associations.tsv'
    
    if user_file.exists():
        print(f"Found cleaned gene-disease data: {user_file}")
        df = load_gene_disease_associations(user_file)
        associations = prepare_gene_disease_data(df)
    elif raw_file.exists():
        print(f"Found raw gene-disease data: {raw_file}")
        print("Note: This data needs cleaning first (pipe-separated diseases)")
        print("Running: python bin/clean_gene_disease_data.py")
        import subprocess
        subprocess.run([
            "python", 
            str(Path(__file__).parent / "clean_gene_disease_data.py")
        ])
        # Try again after cleaning
        if user_file.exists():
            df = load_gene_disease_associations(user_file)
            associations = prepare_gene_disease_data(df)
        else:
            print("Error: Could not clean data")
            return
    else:
        print(f"\nNo gene-disease data found at:")
        print(f"  {user_file}")
        print(f"  {raw_file}")
        print("\nOptions:")
        print("  1. Place your gene-disease association file at the path above")
        print("     (Format: tab-separated with 'disease' and 'gene' columns)")
        print("  2. Download from OMIM, PharmGKB, or GAD manually")
        print("  3. Use example data for testing")
        
        use_example = input("\nUse example data for testing? (y/n): ").lower().strip()
        
        if use_example == 'y':
            print("\nCreating example gene-disease associations...")
            associations = create_example_data()
            print(f"Created {len(associations)} example associations")
        else:
            print("\nPlease provide gene-disease association data and run again.")
            return
    
    # Build network
    G, gene_disease_map = build_gene_based_disease_network(associations)
    
    # Export edgelist
    output_path = data_dir / 'gene_based_disease_network.tsv'
    edgelist = export_edgelist(G, output_path)
    
    # Calculate statistics
    calculate_network_statistics(G)
    
    # Save node list with gene counts
    node_data = []
    for disease in G.nodes():
        disease_genes = associations[associations['disease'] == disease]['gene'].tolist()
        node_data.append({
            'disease': disease,
            'num_genes': len(disease_genes),
            'degree': G.degree(disease),
            'genes': ';'.join(disease_genes)
        })
    
    nodes_df = pd.DataFrame(node_data).sort_values('degree', ascending=False)
    nodes_path = data_dir / 'gene_based_disease_nodes.tsv'
    nodes_df.to_csv(nodes_path, sep='\t', index=False)
    print(f"\nNode data saved to: {nodes_path}")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR REAL DATA")
    print("="*60)
    print("\n1. OMIM Morbid Map:")
    print("   - Register at https://www.omim.org/api")
    print("   - Download morbidmap.txt")
    print("   - Parse format: Phenotype | Gene Symbol | MIM Number")
    print("\n2. PharmGKB:")
    print("   - Download from https://www.pharmgkb.org/downloads")
    print("   - Look for disease-gene relationships")
    print("\n3. GAD (Genetic Association Database):")
    print("   - Download from https://geneticassociationdb.nih.gov/")
    print("\n4. Format your data as TSV with columns: disease, gene")
    print(f"   Save to: {user_file}")
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
