#!/usr/bin/env python3
"""
Compare HPO disease-phenotype network with replication symptom-disease network.
This analyzes the overlap and differences between the two networks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import networkx as nx


def load_data():
    """Load all necessary data files"""
    base_path = Path(__file__).parent.parent
    data = {}
    
    # HPO data
    print("Loading HPO data...")
    data['hpo_network'] = pd.read_csv(
        base_path / 'data' / 'HPO' / 'disease_phenotype_network.tsv',
        sep='\t'
    )
    data['omim_names'] = pd.read_csv(
        base_path / 'data' / 'HPO' / 'omim_name_map.tsv',
        sep='\t'
    )
    
    # Replication data
    print("Loading replication data...")
    data['mesh_diseases'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'data1.txt',
        sep='\t'
    )
    data['mesh_symptoms'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'data2.txt',
        sep='\t'
    )
    data['symptom_disease'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'data3.txt',
        sep='\t'
    )
    data['disease_disease'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'data4.txt',
        sep='\t'
    )
    
    # HPO-MeSH mapping
    print("Loading HPO-MeSH mapping...")
    data['hpo_mesh_mapping'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'hpo_mesh_symptom_mapping.tsv',
        sep='\t'
    )
    data['hpo_enhanced'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'hpo_phenotypes_enhanced.tsv',
        sep='\t'
    )
    
    # Load data6 for full HPO info
    data['hpo_umls'] = pd.read_csv(
        base_path / 'data' / 'replication' / 'data6.csv',
        header=None,
        names=['row_id', 'hpo_term_name', 'hpo_id', 'umls_cui', 
               'umls_concept', 'semantic_type_code', 'semantic_type']
    )
    
    return data, base_path


def analyze_symptom_overlap(data):
    """Analyze overlap between HPO phenotypes and MeSH symptoms"""
    print("\n" + "="*80)
    print("SYMPTOM/PHENOTYPE OVERLAP ANALYSIS")
    print("="*80)
    
    # Get unique symptoms from each source
    hpo_symptoms = set(data['hpo_mesh_mapping']['hpo_id'].unique())
    mesh_symptoms = set(data['mesh_symptoms']['MeSH Symptom Term'].unique())
    mapped_mesh_symptoms = set(data['hpo_mesh_mapping']['mesh_term'].unique())
    
    print(f"\nUnique HPO phenotypes (that matched to MeSH): {len(hpo_symptoms)}")
    print(f"Unique MeSH symptoms in replication data: {len(mesh_symptoms)}")
    print(f"MeSH symptoms matched from HPO: {len(mapped_mesh_symptoms)}")
    print(f"Coverage: {len(mapped_mesh_symptoms)/len(mesh_symptoms)*100:.1f}% of MeSH symptoms have HPO matches")
    
    # Get high-quality matches
    high_quality = data['hpo_mesh_mapping'][data['hpo_mesh_mapping']['match_quality'] == 'high']
    print(f"\nHigh-quality matches (similarity ≥ 0.9): {len(high_quality)}")
    print(f"Unique HPO terms in high-quality matches: {high_quality['hpo_id'].nunique()}")
    print(f"Unique MeSH terms in high-quality matches: {high_quality['mesh_term'].nunique()}")
    
    return hpo_symptoms, mapped_mesh_symptoms


def analyze_network_structure(data):
    """Compare network structures between HPO and replication data"""
    print("\n" + "="*80)
    print("NETWORK STRUCTURE COMPARISON")
    print("="*80)
    
    # HPO network structure
    print("\nHPO Disease-Phenotype Network:")
    hpo_net = data['hpo_network']
    hpo_diseases = set([s for s in hpo_net['Source'].unique() if s.startswith('OMIM:')])
    hpo_phenotypes = set([t for t in hpo_net['Target'].unique() if t.startswith('HP:')])
    
    print(f"  Nodes: {len(set(hpo_net['Source']) | set(hpo_net['Target']))}")
    print(f"  Edges: {len(hpo_net)}")
    print(f"  OMIM Diseases: {len(hpo_diseases)}")
    print(f"  HPO Phenotypes: {len(hpo_phenotypes)}")
    
    # Build bipartite graph
    G_hpo = nx.Graph()
    for _, row in hpo_net.iterrows():
        if row['Source'].startswith('OMIM:') and row['Target'].startswith('HP:'):
            G_hpo.add_edge(row['Source'], row['Target'])
    
    print(f"  Avg phenotypes per disease: {len(hpo_net)/len(hpo_diseases):.2f}")
    
    # Replication network structure
    print("\nReplication Symptom-Disease Network (data3):")
    symptom_disease = data['symptom_disease']
    
    print(f"  Edges: {len(symptom_disease)}")
    print(f"  Unique diseases: {symptom_disease['MeSH Disease Term'].nunique()}")
    print(f"  Unique symptoms: {symptom_disease['MeSH Symptom Term'].nunique()}")
    
    # Build bipartite graph
    G_repl = nx.Graph()
    for _, row in symptom_disease.iterrows():
        G_repl.add_edge(row['MeSH Disease Term'], row['MeSH Symptom Term'])
    
    avg_symptoms = len(symptom_disease) / symptom_disease['MeSH Disease Term'].nunique()
    print(f"  Avg symptoms per disease: {avg_symptoms:.2f}")
    
    return G_hpo, G_repl


def find_shared_disease_symptom_pairs(data):
    """Find disease-symptom pairs that appear in both networks"""
    print("\n" + "="*80)
    print("SHARED DISEASE-SYMPTOM PAIRS")
    print("="*80)
    
    # Create a mapping of MeSH symptoms to HPO IDs
    mesh_to_hpo = defaultdict(set)
    for _, row in data['hpo_mesh_mapping'].iterrows():
        if row['match_quality'] == 'high':  # Use only high-quality matches
            mesh_to_hpo[row['mesh_term']].add(row['hpo_id'])
    
    # Get HPO disease-phenotype associations
    hpo_associations = set()
    for _, row in data['hpo_network'].iterrows():
        if row['Source'].startswith('OMIM:') and row['Target'].startswith('HP:'):
            hpo_associations.add((row['Source'], row['Target']))
    
    # Try to find matching pairs
    matches = []
    
    for _, row in data['symptom_disease'].iterrows():
        mesh_symptom = row['MeSH Symptom Term']
        mesh_disease = row['MeSH Disease Term']
        
        if mesh_symptom in mesh_to_hpo:
            # Get corresponding HPO phenotypes
            hpo_phenotypes = mesh_to_hpo[mesh_symptom]
            
            # Check if any OMIM disease name contains the MeSH disease term
            for omim_id, omim_name in zip(data['omim_names']['omim_id'], 
                                          data['omim_names']['name']):
                # Simple name matching (could be improved)
                if mesh_disease.lower() in omim_name.lower():
                    # Check if this disease-phenotype pair exists
                    for hpo_id in hpo_phenotypes:
                        if (omim_id, hpo_id) in hpo_associations:
                            matches.append({
                                'omim_id': omim_id,
                                'omim_name': omim_name,
                                'hpo_id': hpo_id,
                                'mesh_disease': mesh_disease,
                                'mesh_symptom': mesh_symptom,
                                'tfidf_score': row['TFIDF score']
                            })
    
    if matches:
        matches_df = pd.DataFrame(matches)
        print(f"\nFound {len(matches)} matching disease-symptom pairs!")
        print(f"Unique diseases: {matches_df['omim_id'].nunique()}")
        print(f"Unique phenotypes/symptoms: {matches_df['hpo_id'].nunique()}")
        
        print("\nTop 10 matches by TFIDF score:")
        print(matches_df.nlargest(10, 'tfidf_score')[
            ['omim_name', 'mesh_symptom', 'tfidf_score']
        ].to_string(index=False))
        
        return matches_df
    else:
        print("\nNo exact disease name matches found.")
        print("Note: This is expected as OMIM uses specific syndrome names")
        print("while MeSH uses broader disease categories.")
        return None


def analyze_phenotype_profiles(data, mapped_mesh_symptoms):
    """Analyze diseases by their phenotype profiles"""
    print("\n" + "="*80)
    print("PHENOTYPE PROFILE ANALYSIS")
    print("="*80)
    
    # Get HPO phenotypes that have MeSH matches
    matched_hpo_ids = set(data['hpo_mesh_mapping'][
        data['hpo_mesh_mapping']['match_quality'] == 'high'
    ]['hpo_id'])
    
    # Count how many diseases have these phenotypes
    hpo_net = data['hpo_network']
    disease_phenotype_counts = defaultdict(set)
    
    for _, row in hpo_net.iterrows():
        if row['Source'].startswith('OMIM:') and row['Target'] in matched_hpo_ids:
            disease_phenotype_counts[row['Source']].add(row['Target'])
    
    # Get diseases with most matched phenotypes
    disease_counts = [(d, len(p)) for d, p in disease_phenotype_counts.items()]
    disease_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nDiseases with HPO phenotypes that map to MeSH symptoms:")
    print(f"  Total diseases: {len(disease_counts)}")
    print(f"  Avg matched phenotypes per disease: {np.mean([c for _, c in disease_counts]):.2f}")
    
    print("\nTop 10 diseases with most MeSH-matched phenotypes:")
    for omim_id, count in disease_counts[:10]:
        omim_name = data['omim_names'][data['omim_names']['omim_id'] == omim_id]['name'].values
        name = omim_name[0] if len(omim_name) > 0 else "Unknown"
        print(f"  {omim_id}: {name[:60]:<60} ({count} phenotypes)")
    
    # Analyze semantic types of matched phenotypes
    print("\n\nSemantic type distribution of matched phenotypes:")
    matched_enhanced = data['hpo_enhanced'][
        data['hpo_enhanced']['hpo_id'].isin(matched_hpo_ids)
    ]
    semantic_counts = matched_enhanced['semantic_type'].value_counts()
    print(semantic_counts.to_string())
    
    return disease_phenotype_counts


def create_integrated_network_sample(data, disease_phenotype_counts):
    """Create sample of integrated network showing both HPO and MeSH connections"""
    print("\n" + "="*80)
    print("INTEGRATED NETWORK SAMPLE")
    print("="*80)
    
    # Get top diseases with matched phenotypes
    top_diseases = sorted(disease_phenotype_counts.items(), 
                         key=lambda x: len(x[1]), reverse=True)[:5]
    
    integrated_edges = []
    
    for omim_id, hpo_phenotypes in top_diseases:
        omim_name = data['omim_names'][
            data['omim_names']['omim_id'] == omim_id
        ]['name'].values
        disease_name = omim_name[0] if len(omim_name) > 0 else "Unknown"
        
        for hpo_id in hpo_phenotypes:
            # Get MeSH symptom mappings
            mesh_mappings = data['hpo_mesh_mapping'][
                (data['hpo_mesh_mapping']['hpo_id'] == hpo_id) &
                (data['hpo_mesh_mapping']['match_quality'] == 'high')
            ]
            
            for _, mapping in mesh_mappings.iterrows():
                # Check if this symptom appears in replication data
                symptom_occurrences = data['symptom_disease'][
                    data['symptom_disease']['MeSH Symptom Term'] == mapping['mesh_term']
                ]['PubMed occurrence'].sum()
                
                integrated_edges.append({
                    'omim_id': omim_id,
                    'omim_name': disease_name,
                    'hpo_id': hpo_id,
                    'hpo_term': mapping['hpo_term'],
                    'mesh_symptom': mapping['mesh_term'],
                    'semantic_type': mapping['semantic_type'],
                    'pubmed_occurrences': symptom_occurrences
                })
    
    integrated_df = pd.DataFrame(integrated_edges)
    
    print(f"\nCreated integrated network with {len(integrated_df)} edges")
    print(f"Diseases: {integrated_df['omim_id'].nunique()}")
    print(f"HPO phenotypes: {integrated_df['hpo_id'].nunique()}")
    print(f"MeSH symptoms: {integrated_df['mesh_symptom'].nunique()}")
    
    print("\nSample integrated connections:")
    print(integrated_df.head(20)[
        ['omim_name', 'mesh_symptom', 'pubmed_occurrences', 'semantic_type']
    ].to_string(index=False))
    
    return integrated_df


def main():
    # Load data
    data, base_path = load_data()
    
    # Run analyses
    hpo_symptoms, mapped_mesh = analyze_symptom_overlap(data)
    G_hpo, G_repl = analyze_network_structure(data)
    matches_df = find_shared_disease_symptom_pairs(data)
    disease_pheno_counts = analyze_phenotype_profiles(data, mapped_mesh)
    integrated_df = create_integrated_network_sample(data, disease_pheno_counts)
    
    # Save integrated network
    output_file = base_path / 'data' / 'replication' / 'integrated_hpo_mesh_network.tsv'
    integrated_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Saved integrated network to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"""
    HPO Network:
      - {len(set([s for s in data['hpo_network']['Source'].unique() if s.startswith('OMIM:')]))} OMIM diseases
      - {len(set([t for t in data['hpo_network']['Target'].unique() if t.startswith('HP:')]))} HPO phenotypes
      - {len(data['hpo_network'])} disease-phenotype associations
    
    Replication Network:
      - {data['symptom_disease']['MeSH Disease Term'].nunique()} MeSH diseases
      - {data['symptom_disease']['MeSH Symptom Term'].nunique()} MeSH symptoms
      - {len(data['symptom_disease'])} symptom-disease associations
    
    Integration via data6.csv:
      - {len(data['hpo_mesh_mapping'])} total HPO→MeSH mappings
      - {data['hpo_mesh_mapping']['hpo_id'].nunique()} unique HPO phenotypes mapped
      - {data['hpo_mesh_mapping']['mesh_term'].nunique()} unique MeSH symptoms matched
      - {len(data['hpo_mesh_mapping'][data['hpo_mesh_mapping']['match_quality']=='high'])} high-quality matches
    
    Integrated Network:
      - {integrated_df['omim_id'].nunique()} OMIM diseases with mapped phenotypes
      - {integrated_df['hpo_id'].nunique()} HPO phenotypes mapped to MeSH
      - {integrated_df['mesh_symptom'].nunique()} MeSH symptoms in common vocabulary
    """)
    
    print("\n✓ Network comparison complete!")


if __name__ == '__main__':
    main()
