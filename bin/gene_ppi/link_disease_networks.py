"""
Link Disease Networks Using MeSH Terminology

This script links three disease networks (symptom-based, gene-based, PPI-based)
by using MeSH IDs as a common identifier bridge.

Data flow:
1. Symptom-based network: data3.txt uses MeSH disease terms
2. Gene-based network: ClinVar uses clinical disease names
3. PPI-based network: Derived from gene associations

Strategy:
- Map ClinVar disease names to MeSH terms using fuzzy matching and manual lookup
- Create unified disease mapping file: disease_name -> mesh_id -> network_id
- Link networks using this mapping
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np


def load_mesh_data(mesh_diseases_file):
    """
    Load MeSH disease terminology.
    
    Parameters:
    -----------
    mesh_diseases_file : str or Path
        Path to mesh_diseases.tsv
        
    Returns:
    --------
    mesh_dict : dict
        Mapping from MeSH term name to MeSH ID
    mesh_by_id : dict
        Mapping from MeSH ID to name
    """
    print("="*70)
    print("LOADING MESH TERMINOLOGY")
    print("="*70)
    
    df = pd.read_csv(mesh_diseases_file, sep='\t')
    
    mesh_dict = {}
    mesh_by_id = {}
    
    for _, row in df.iterrows():
        mesh_id = str(row['mesh_ui']).strip()
        name = str(row['name']).strip()
        
        mesh_dict[name] = mesh_id
        mesh_by_id[mesh_id] = name
    
    print(f"Loaded {len(mesh_dict)} MeSH disease terms\n")
    
    return mesh_dict, mesh_by_id


def load_symptom_network_diseases(data3_file):
    """
    Load disease names from symptom-based network (data3.txt).
    These are already in MeSH terminology.
    
    Parameters:
    -----------
    data3_file : str or Path
        Path to data3.txt
        
    Returns:
    --------
    symptom_diseases : set
        Set of MeSH disease names from symptom network
    """
    print("="*70)
    print("LOADING SYMPTOM-BASED NETWORK DISEASES")
    print("="*70)
    
    df = pd.read_csv(data3_file, sep='\t')
    diseases = set(df['MeSH Disease Term'].unique())
    
    print(f"Found {len(diseases)} unique diseases in symptom network")
    print(f"Sample: {list(diseases)[:5]}\n")
    
    return diseases


def load_gene_network_diseases(gene_disease_file):
    """
    Load disease names from gene-based network (ClinVar).
    These need to be mapped to MeSH terms.
    
    Parameters:
    -----------
    gene_disease_file : str or Path
        Path to gene_disease_associations_cleaned.tsv
        
    Returns:
    --------
    gene_diseases : set
        Set of disease names from gene network
    """
    print("="*70)
    print("LOADING GENE-BASED NETWORK DISEASES")
    print("="*70)
    
    df = pd.read_csv(gene_disease_file, sep='\t')
    diseases = set(df['disease'].unique())
    
    print(f"Found {len(diseases)} unique diseases in gene network")
    print(f"Sample: {list(diseases)[:5]}\n")
    
    return diseases


def fuzzy_match_disease_name(clinvar_name, mesh_names, threshold=0.6):
    """
    Find best MeSH match for a ClinVar disease name using fuzzy string matching.
    
    Parameters:
    -----------
    clinvar_name : str
        ClinVar disease name
    mesh_names : set
        Set of available MeSH disease names
    threshold : float
        Minimum similarity score (0-1)
        
    Returns:
    --------
    best_match : str or None
        Best matching MeSH disease name, or None if no good match
    score : float
        Similarity score
    """
    best_match = None
    best_score = 0
    
    clinvar_lower = clinvar_name.lower()
    
    for mesh_name in mesh_names:
        mesh_lower = mesh_name.lower()
        
        # Exact match
        if clinvar_lower == mesh_lower:
            return mesh_name, 1.0
        
        # Check if one contains the other (partial match)
        if clinvar_lower in mesh_lower or mesh_lower in clinvar_lower:
            score = 0.85
        else:
            # Fuzzy string matching
            score = SequenceMatcher(None, clinvar_lower, mesh_lower).ratio()
        
        if score > best_score:
            best_score = score
            best_match = mesh_name
    
    if best_score >= threshold:
        return best_match, best_score
    
    return None, 0


def create_disease_mapping(mesh_dict, symptom_diseases, gene_diseases):
    """
    Create a mapping from all disease names to MeSH IDs.
    
    Parameters:
    -----------
    mesh_dict : dict
        MeSH name -> ID mapping
    symptom_diseases : set
        Disease names from symptom network (already in MeSH)
    gene_diseases : set
        Disease names from gene network (ClinVar format)
        
    Returns:
    --------
    mapping : dict
        disease_name -> {'mesh_id': str, 'source': str, 'match_score': float}
    """
    print("="*70)
    print("CREATING DISEASE NAME MAPPING")
    print("="*70)
    
    mapping = {}
    
    # Step 1: Map symptom-based diseases (already in MeSH)
    print(f"\nMapping {len(symptom_diseases)} symptom-based diseases...")
    
    symptom_matches = 0
    symptom_no_match = []
    
    for disease in symptom_diseases:
        if disease in mesh_dict:
            mapping[disease] = {
                'mesh_id': mesh_dict[disease],
                'source': 'symptom',
                'match_score': 1.0,
                'match_type': 'exact'
            }
            symptom_matches += 1
        else:
            symptom_no_match.append(disease)
    
    print(f"  Matched: {symptom_matches}/{len(symptom_diseases)}")
    if symptom_no_match:
        print(f"  Unmatched ({len(symptom_no_match)}): {symptom_no_match[:5]}")
    
    # Step 2: Map gene-based diseases (ClinVar format)
    print(f"\nMapping {len(gene_diseases)} gene-based diseases...")
    
    gene_matches = 0
    gene_partial = 0
    gene_no_match = []
    
    mesh_names = set(mesh_dict.keys())
    done=0
    start_time = pd.Timestamp.now()
    for disease in gene_diseases:
        if done % 100 ==0 and done>0:
            print(f"  Processed {done}/{len(gene_diseases)} gene diseases...")
            print(f"estimated time remaining: {((len(gene_diseases)-done)/(done))*((pd.Timestamp.now()-start_time).total_seconds()/60):.2f} minutes")
        done+=1
        if disease in mesh_dict:
            # Exact match
            mapping[disease] = {
                'mesh_id': mesh_dict[disease],
                'source': 'gene',
                'match_score': 1.0,
                'match_type': 'exact'
            }
            gene_matches += 1
        else:
            # Try fuzzy matching
            best_match, score = fuzzy_match_disease_name(disease, mesh_names, threshold=0.75)
            
            if best_match:
                mapping[disease] = {
                    'mesh_id': mesh_dict[best_match],
                    'source': 'gene',
                    'match_score': score,
                    'match_type': 'fuzzy',
                    'fuzzy_match': best_match
                }
                gene_partial += 1
            else:
                gene_no_match.append(disease)
    
    print(f"  Exact matches: {gene_matches}/{len(gene_diseases)}")
    print(f"  Fuzzy matches: {gene_partial}/{len(gene_diseases)}")
    print(f"  Unmatched: {len(gene_no_match)}/{len(gene_diseases)}")
    
    if gene_no_match:
        print(f"\n  Sample unmatched genes (first 10):")
        for disease in gene_no_match[:10]:
            print(f"    - {disease}")
    
    print(f"\n  Total mapped: {len(mapping)}")
    
    return mapping, gene_no_match


def save_disease_mapping(mapping, output_file):
    """
    Save disease mapping to TSV file for manual review.
    
    Parameters:
    -----------
    mapping : dict
        Disease name mapping
    output_file : str or Path
        Output file path
    """
    print(f"\nSaving disease mapping to {output_file}...")
    
    rows = []
    for disease_name, info in mapping.items():
        rows.append({
            'disease_name': disease_name,
            'mesh_id': info['mesh_id'],
            'source': info['source'],
            'match_score': info['match_score'],
            'match_type': info['match_type'],
            'fuzzy_match': info.get('fuzzy_match', '')
        })
    
    df = pd.DataFrame(rows).sort_values('match_score', ascending=False)
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"Saved {len(df)} mappings")


def load_disease_mapping(mapping_file):
    """
    Load existing disease mapping from TSV.
    Returns None if file missing.
    """
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        return None

    print(f"\nLoading existing mapping from {mapping_path}...")
    df = pd.read_csv(mapping_path, sep='\t')

    mapping = {}
    for _, row in df.iterrows():
        disease_name = str(row['disease_name']).strip()
        mapping[disease_name] = {
            'mesh_id': str(row['mesh_id']).strip(),
            'source': row.get('source', ''),
            'match_score': float(row.get('match_score', 1.0)),
            'match_type': row.get('match_type', 'exact'),
            'fuzzy_match': row.get('fuzzy_match', '')
        }

    print(f"Loaded {len(mapping)} mappings\n")
    return mapping


def create_unified_disease_network(symptom_network_file, gene_network_file, 
                                   ppi_1st_network_file, mapping, mesh_by_id,
                                   output_file):
    """
    Create a unified network linking all three disease networks via MeSH IDs.
    
    Parameters:
    -----------
    symptom_network_file : str or Path
        Symptom-based network edgelist
    gene_network_file : str or Path
        Gene-based network edgelist
    ppi_1st_network_file : str or Path
        PPI 1st order network edgelist
    mapping : dict
        Disease name to MeSH ID mapping
    mesh_by_id : dict
        MeSH ID to name mapping
    output_file : str or Path
        Output unified network file
    """
    print("\n" + "="*70)
    print("CREATING UNIFIED NETWORK")
    print("="*70)
    
    unified_edges = []
    edge_sources = defaultdict(set)
    
    # Load and remap each network
    networks_loaded = {
        'symptom': False,
        'gene': False,
        'ppi': False
    }
    
    # 1. Symptom-based network
    if Path(symptom_network_file).exists():
        print(f"\nLoading symptom network from {symptom_network_file}...")
        df_symptom = pd.read_csv(symptom_network_file, sep='\t')
        
        symptom_edges = 0
        for _, row in df_symptom.iterrows():
            disease1 = str(row['disease1']).strip()
            disease2 = str(row['disease2']).strip()
            
            if disease1 in mapping and disease2 in mapping:
                mesh_id1 = mapping[disease1]['mesh_id']
                mesh_id2 = mapping[disease2]['mesh_id']
                
                if mesh_id1 != mesh_id2:  # Avoid self-loops
                    unified_edges.append({
                        'disease1_mesh_id': mesh_id1,
                        'disease1_name': mesh_by_id.get(mesh_id1, disease1),
                        'disease2_mesh_id': mesh_id2,
                        'disease2_name': mesh_by_id.get(mesh_id2, disease2),
                        'network_source': 'symptom',
                        'similarity_score': row.get('cosine_similarity', row.get('weight', 1.0))
                    })
                    
                    # Track which networks connect this pair
                    edge_key = tuple(sorted([mesh_id1, mesh_id2]))
                    edge_sources[edge_key].add('symptom')
                    symptom_edges += 1
        
        print(f"  Mapped {symptom_edges} edges to MeSH IDs")
        networks_loaded['symptom'] = True
    
    # 2. Gene-based network (if exists)
    if gene_network_file and Path(gene_network_file).exists():
        print(f"\nLoading gene network from {gene_network_file}...")
        df_gene = pd.read_csv(gene_network_file, sep='\t')
        
        gene_edges = 0
        for _, row in df_gene.iterrows():
            disease1 = str(row['disease1']).strip()
            disease2 = str(row['disease2']).strip()
            
            if disease1 in mapping and disease2 in mapping:
                mesh_id1 = mapping[disease1]['mesh_id']
                mesh_id2 = mapping[disease2]['mesh_id']
                
                if mesh_id1 != mesh_id2:
                    unified_edges.append({
                        'disease1_mesh_id': mesh_id1,
                        'disease1_name': mesh_by_id.get(mesh_id1, disease1),
                        'disease2_mesh_id': mesh_id2,
                        'disease2_name': mesh_by_id.get(mesh_id2, disease2),
                        'network_source': 'gene',
                        'shared_genes': row.get('shared_genes', 1)
                    })
                    
                    edge_key = tuple(sorted([mesh_id1, mesh_id2]))
                    edge_sources[edge_key].add('gene')
                    gene_edges += 1
        
        print(f"  Mapped {gene_edges} edges to MeSH IDs")
        networks_loaded['gene'] = True
    
    # 3. PPI-based network
    if ppi_1st_network_file and Path(ppi_1st_network_file).exists():
        print(f"\nLoading PPI network from {ppi_1st_network_file}...")
        df_ppi = pd.read_csv(ppi_1st_network_file, sep='\t')
        
        ppi_edges = 0
        for _, row in df_ppi.iterrows():
            disease1 = str(row['disease1']).strip()
            disease2 = str(row['disease2']).strip()
            
            if disease1 in mapping and disease2 in mapping:
                mesh_id1 = mapping[disease1]['mesh_id']
                mesh_id2 = mapping[disease2]['mesh_id']
                
                if mesh_id1 != mesh_id2:
                    unified_edges.append({
                        'disease1_mesh_id': mesh_id1,
                        'disease1_name': mesh_by_id.get(mesh_id1, disease1),
                        'disease2_mesh_id': mesh_id2,
                        'disease2_name': mesh_by_id.get(mesh_id2, disease2),
                        'network_source': 'ppi',
                        'shared_proteins': row.get('shared_proteins', 1)
                    })
                    
                    edge_key = tuple(sorted([mesh_id1, mesh_id2]))
                    edge_sources[edge_key].add('ppi')
                    ppi_edges += 1
        
        print(f"  Mapped {ppi_edges} edges to MeSH IDs")
        networks_loaded['ppi'] = True
    
    # Create DataFrame
    df_unified = pd.DataFrame(unified_edges)
    
    if len(df_unified) > 0:
        # Merge duplicate edges (same pair appearing in multiple networks)
        df_unified = df_unified.sort_values('disease1_mesh_id')
        
        # Create a field showing which networks support each edge
        df_unified['networks'] = df_unified.apply(
            lambda row: ';'.join(sorted(edge_sources[tuple(sorted([row['disease1_mesh_id'], row['disease2_mesh_id']]))]))
            if tuple(sorted([row['disease1_mesh_id'], row['disease2_mesh_id']])) in edge_sources else 'unknown',
            axis=1
        )
        
        # Count edges by source
        print(f"\nEdge statistics by source:")
        for source in ['symptom', 'gene', 'ppi']:
            if source in df_unified['network_source'].values:
                count = (df_unified['network_source'] == source).sum()
                print(f"  {source}: {count} edges")
        
        # Count edges in multiple networks
        edge_counts = df_unified.groupby(['disease1_mesh_id', 'disease2_mesh_id']).size()
        print(f"\nEdges appearing in multiple networks:")
        print(f"  In exactly 2 networks: {(edge_counts == 2).sum()}")
        print(f"  In all 3 networks: {(edge_counts == 3).sum()}")
        
        # Save
        df_unified.to_csv(output_file, sep='\t', index=False)
        print(f"\nSaved unified network to {output_file}")
        print(f"Total edges: {len(df_unified)}")
    else:
        print("\nWarning: No edges could be mapped to MeSH IDs!")
    
    return df_unified


def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-remap', action='store_true',
                        help='Recompute disease name mapping even if mapping file exists.')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("DISEASE NETWORK LINKING SCRIPT")
    print("="*70)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'replication'
    mesh_dir = base_dir / 'data' / 'MeSH'
    
    # Input files
    mesh_diseases_file = mesh_dir / 'mesh_diseases.tsv'
    data3_file = data_dir / 'data3.txt'
    gene_disease_file = data_dir / 'gene_disease/gene_disease_associations_cleaned.tsv'
    symptom_network_file = data_dir / 'disease_symptoms_projection.tsv'
    gene_network_file = data_dir / 'gene_disease/gene_based_disease_network.tsv'  # If exists
    ppi_1st_network_file = data_dir / 'ppi/ppi_1st_order_disease_network.tsv'
    
    # Output files
    mapping_file = data_dir / 'disease_name_mapping.tsv'
    unified_network_file = data_dir / 'unified_disease_network.tsv'
    
    # Check required files
    if not mesh_diseases_file.exists():
        print(f"Error: MeSH diseases file not found: {mesh_diseases_file}")
        return
    
    if not data3_file.exists():
        print(f"Error: data3.txt not found: {data3_file}")
        return
    
    if not gene_disease_file.exists():
        print(f"Error: Gene-disease file not found: {gene_disease_file}")
        return
    
    # Load MeSH terminology (always)
    mesh_dict, mesh_by_id = load_mesh_data(mesh_diseases_file)

    # Load or create disease mapping
    mapping = None
    if mapping_file.exists() and not args.force_remap:
        mapping = load_disease_mapping(mapping_file)

    if mapping is None:
        # Build mapping from scratch
        symptom_diseases = load_symptom_network_diseases(data3_file)
        gene_diseases = load_gene_network_diseases(gene_disease_file)
        mapping, unmatched_genes = create_disease_mapping(mesh_dict, symptom_diseases, gene_diseases)
        save_disease_mapping(mapping, mapping_file)
    else:
        print("Reusing existing disease_name_mapping.tsv (use --force-remap to rebuild)\n")
    
    # Create unified network
    if symptom_network_file.exists():
        create_unified_disease_network(
            symptom_network_file,
            gene_network_file if gene_network_file.exists() else None,
            ppi_1st_network_file if ppi_1st_network_file.exists() else None,
            mapping,
            mesh_by_id,
            unified_network_file
        )
    else:
        print(f"\nWarning: Symptom network not found at {symptom_network_file}")
        print("Run: python bin/disease_similarity_network.py")
    
    # Summary
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print(f"\n1. Disease Name Mapping: {mapping_file}")
    print(f"   - Maps all disease names to MeSH IDs")
    print(f"   - Shows match confidence and type")
    print(f"   - Use for manual review/correction")
    
    print(f"\n2. Unified Network: {unified_network_file}")
    print(f"   - Links diseases via MeSH IDs")
    print(f"   - Shows which networks support each connection")
    print(f"   - Combines symptom + gene + PPI evidence")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Review disease_name_mapping.tsv for accuracy
   - Check unmatched genes (match_type = None)
   - Manually fix any incorrect fuzzy matches (match_score < 0.95)

2. Create a manual corrections file if needed:
   - Add rows with your corrections
   - Use same columns as mapping file

3. Use unified_disease_network.tsv for:
   - Network analysis combining all three approaches
   - Identify consensus links (edges in multiple networks)
   - Validate gene-disease associations with symptoms
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
