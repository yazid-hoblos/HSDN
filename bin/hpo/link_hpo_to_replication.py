#!/usr/bin/env python3
"""
Link HPO phenotypes to replication data using UMLS semantic types.
This script uses data6.csv to bridge HPO terms with MeSH-based replication data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import re


def load_hpo_umls_mapping(data_path):
    """Load HPO to UMLS mapping from data6.csv"""
    df = pd.read_csv(data_path, header=None, names=[
        'row_id', 'hpo_term_name', 'hpo_id', 'umls_cui', 
        'umls_concept', 'semantic_type_code', 'semantic_type'
    ])
    return df


def load_mesh_symptoms(data_path):
    """Load MeSH symptom terms from replication data"""
    df = pd.read_csv(data_path, sep='\t')
    return df


def load_mesh_diseases(data_path):
    """Load MeSH disease terms from replication data"""
    df = pd.read_csv(data_path, sep='\t')
    return df


def load_hpo_disease_network(data_path):
    """Load HPO disease-phenotype network"""
    df = pd.read_csv(data_path, sep='\t')
    return df


def normalize_term(term):
    """Normalize medical terms for comparison"""
    if pd.isna(term):
        return ""
    term = str(term).lower()
    # Remove parenthetical content
    term = re.sub(r'\([^)]*\)', '', term)
    # Remove special characters but keep spaces and hyphens
    term = re.sub(r'[^a-z0-9\s-]', '', term)
    # Normalize whitespace
    term = ' '.join(term.split())
    return term


def string_similarity(s1, s2):
    """Calculate string similarity ratio"""
    return SequenceMatcher(None, normalize_term(s1), normalize_term(s2)).ratio()


def find_mesh_matches(hpo_term, mesh_terms, threshold=0.7):
    """Find matching MeSH terms for an HPO term"""
    matches = []
    hpo_normalized = normalize_term(hpo_term)
    
    for mesh_term in mesh_terms:
        similarity = string_similarity(hpo_term, mesh_term)
        if similarity >= threshold:
            matches.append({
                'mesh_term': mesh_term,
                'similarity': similarity
            })
    
    # Sort by similarity descending
    matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    return matches


def main():
    # Set up paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'replication'
    hpo_path = base_path / 'data' / 'HPO'
    mesh_path = base_path / 'data' / 'MeSH'
    output_path = data_path
    
    print("Loading data files...")
    
    # Load HPO-UMLS mapping
    hpo_umls = load_hpo_umls_mapping(data_path / 'data6.csv')
    print(f"Loaded {len(hpo_umls)} HPO-UMLS mappings")
    
    # Filter for symptoms and signs
    symptom_types = ['Sign or Symptom', 'Finding', 'Disease or Syndrome']
    hpo_symptoms = hpo_umls[hpo_umls['semantic_type'].isin(symptom_types)].copy()
    print(f"Found {len(hpo_symptoms)} HPO terms classified as symptoms/findings/diseases")
    
    # Load MeSH data
    mesh_symptoms = load_mesh_symptoms(data_path / 'data2.txt')
    mesh_diseases = load_mesh_diseases(data_path / 'data1.txt')
    print(f"Loaded {len(mesh_symptoms)} MeSH symptoms and {len(mesh_diseases)} MeSH diseases")
    
    # Load HPO disease-phenotype network
    hpo_network = load_hpo_disease_network(hpo_path / 'disease_phenotype_network.tsv')
    print(f"Loaded {len(hpo_network)} HPO disease-phenotype associations")
    
    # Get unique HPO phenotypes used in the network
    hpo_phenotypes_in_network = set(hpo_network['Target'].unique()) | set(hpo_network['Source'].unique())
    hpo_phenotypes_in_network = {p for p in hpo_phenotypes_in_network if p.startswith('HP:')}
    print(f"Found {len(hpo_phenotypes_in_network)} unique HPO phenotypes in network")
    
    # Filter to only HPO terms that are in the network
    hpo_symptoms_in_network = hpo_symptoms[hpo_symptoms['hpo_id'].isin(hpo_phenotypes_in_network)]
    print(f"Found {len(hpo_symptoms_in_network)} symptom/finding HPO terms used in your network")
    
    # Create mappings
    print("\nCreating HPO to MeSH symptom mappings...")
    hpo_to_mesh_symptom = []
    
    mesh_symptom_list = mesh_symptoms['MeSH Symptom Term'].tolist()
    
    for idx, row in hpo_symptoms_in_network.iterrows():
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(hpo_symptoms_in_network)}...")
        
        hpo_term = row['hpo_term_name']
        hpo_id = row['hpo_id']
        umls_cui = row['umls_cui']
        semantic_type = row['semantic_type']
        
        # Find MeSH matches
        matches = find_mesh_matches(hpo_term, mesh_symptom_list, threshold=0.7)
        
        if matches:
            for match in matches[:3]:  # Top 3 matches
                hpo_to_mesh_symptom.append({
                    'hpo_id': hpo_id,
                    'hpo_term': hpo_term,
                    'umls_cui': umls_cui,
                    'semantic_type': semantic_type,
                    'mesh_term': match['mesh_term'],
                    'similarity_score': match['similarity'],
                    'match_quality': 'high' if match['similarity'] >= 0.9 else 'medium'
                })
    
    # Convert to DataFrame and save
    hpo_mesh_mapping = pd.DataFrame(hpo_to_mesh_symptom)
    
    if len(hpo_mesh_mapping) > 0:
        output_file = output_path / 'hpo_mesh_symptom_mapping.tsv'
        hpo_mesh_mapping.to_csv(output_file, sep='\t', index=False)
        print(f"\nCreated {len(hpo_mesh_mapping)} HPO-MeSH symptom mappings")
        print(f"Saved to: {output_file}")
        
        # Summary statistics
        print("\nMapping Summary:")
        print(f"  Unique HPO terms mapped: {hpo_mesh_mapping['hpo_id'].nunique()}")
        print(f"  Unique MeSH terms matched: {hpo_mesh_mapping['mesh_term'].nunique()}")
        print(f"  High quality matches (≥0.9): {len(hpo_mesh_mapping[hpo_mesh_mapping['match_quality']=='high'])}")
        print(f"  Medium quality matches (0.7-0.9): {len(hpo_mesh_mapping[hpo_mesh_mapping['match_quality']=='medium'])}")
        
        print("\nSemantic type distribution:")
        print(hpo_mesh_mapping['semantic_type'].value_counts())
        
        print("\nTop 10 matches by similarity:")
        print(hpo_mesh_mapping.nlargest(10, 'similarity_score')[
            ['hpo_term', 'mesh_term', 'similarity_score', 'semantic_type']
        ].to_string(index=False))
    else:
        print("\nNo mappings found. Consider lowering the similarity threshold.")
    
    # Create enhanced HPO phenotype annotations
    print("\n\nCreating enhanced HPO phenotype annotations...")
    hpo_enhanced = hpo_symptoms_in_network.copy()
    hpo_enhanced['has_mesh_match'] = hpo_enhanced['hpo_id'].isin(hpo_mesh_mapping['hpo_id'])
    
    output_file = output_path / 'hpo_phenotypes_enhanced.tsv'
    hpo_enhanced.to_csv(output_file, sep='\t', index=False)
    print(f"Saved enhanced annotations to: {output_file}")
    print(f"  Total HPO phenotypes: {len(hpo_enhanced)}")
    print(f"  With MeSH matches: {hpo_enhanced['has_mesh_match'].sum()}")
    print(f"  Without MeSH matches: {(~hpo_enhanced['has_mesh_match']).sum()}")
    
    # Create summary by semantic type
    print("\n\nHPO terms by semantic type in your network:")
    semantic_summary = hpo_symptoms_in_network.groupby('semantic_type').agg({
        'hpo_id': 'count',
        'umls_cui': 'nunique'
    }).rename(columns={'hpo_id': 'count', 'umls_cui': 'unique_umls_concepts'})
    print(semantic_summary.sort_values('count', ascending=False))
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
