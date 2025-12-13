"""
Disease-Disease Network Generator based on Symptom Similarity

This script generates a projected disease-disease network using symptom-based
similarity. Diseases are represented as vectors of symptoms with TF-IDF weights,
and similarity is calculated using cosine similarity.

Reference:
- wi,j = Wi,j * log(N/ni)
  where Wi,j is the co-occurrence count,
  N is the total number of diseases,
  ni is the number of diseases where symptom i appears.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from collections import defaultdict
import networkx as nx
from pathlib import Path


def load_symptom_disease_data(filepath):
    """
    Load symptom-disease associations from data file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file containing symptom-disease associations
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with symptom-disease associations
    """
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} symptom-disease associations")
    print(f"Unique symptoms: {df['MeSH Symptom Term'].nunique()}")
    print(f"Unique diseases: {df['MeSH Disease Term'].nunique()}")
    return df


def calculate_tfidf_weights(df):
    """
    Calculate TF-IDF weights for symptom-disease associations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'MeSH Symptom Term', 'MeSH Disease Term', 'PubMed occurrence'
        
    Returns:
    --------
    disease_vectors : dict
        Dictionary mapping disease names to symptom vectors (as dictionaries)
    all_symptoms : list
        List of all unique symptoms
    """
    # Get all unique diseases and symptoms
    all_diseases = df['MeSH Disease Term'].unique()
    all_symptoms = df['MeSH Symptom Term'].unique()
    
    N = len(all_diseases)  # Total number of diseases
    print(f"\nCalculating TF-IDF weights for {N} diseases and {len(all_symptoms)} symptoms...")
    
    # Calculate ni (number of diseases where symptom i appears)
    symptom_disease_count = df.groupby('MeSH Symptom Term')['MeSH Disease Term'].nunique()
    
    # Initialize disease vectors
    disease_vectors = defaultdict(dict)
    
    # Calculate TF-IDF weights: wi,j = Wi,j * log(N/ni)
    for _, row in df.iterrows():
        symptom = row['MeSH Symptom Term']
        disease = row['MeSH Disease Term']
        Wi_j = row['PubMed occurrence']  # Co-occurrence count
        ni = symptom_disease_count[symptom]  # Number of diseases with this symptom
        
        # TF-IDF weight
        wi_j = Wi_j * np.log(N / ni)
        disease_vectors[disease][symptom] = wi_j
    
    print(f"Created vectors for {len(disease_vectors)} diseases")
    return disease_vectors, list(all_symptoms)


def create_disease_feature_matrix(disease_vectors, all_symptoms):
    """
    Create a matrix representation of disease vectors.
    
    Parameters:
    -----------
    disease_vectors : dict
        Dictionary mapping disease names to symptom vectors
    all_symptoms : list
        List of all unique symptoms
        
    Returns:
    --------
    matrix : numpy.ndarray
        Matrix where rows are diseases and columns are symptoms
    disease_names : list
        List of disease names corresponding to matrix rows
    """
    disease_names = list(disease_vectors.keys())
    n_diseases = len(disease_names)
    n_symptoms = len(all_symptoms)
    
    # Create symptom to index mapping
    symptom_to_idx = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
    
    # Initialize matrix
    matrix = np.zeros((n_diseases, n_symptoms))
    
    # Fill matrix with TF-IDF weights
    for disease_idx, disease in enumerate(disease_names):
        for symptom, weight in disease_vectors[disease].items():
            symptom_idx = symptom_to_idx[symptom]
            matrix[disease_idx, symptom_idx] = weight
    
    print(f"\nCreated feature matrix of shape {matrix.shape}")
    return matrix, disease_names


def calculate_cosine_similarity(matrix):
    """
    Calculate pairwise cosine similarity between all disease vectors.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Matrix where rows are disease vectors
        
    Returns:
    --------
    similarity_matrix : numpy.ndarray
        Symmetric matrix of cosine similarities
    """
    n_diseases = matrix.shape[0]
    similarity_matrix = np.zeros((n_diseases, n_diseases))
    
    print(f"\nCalculating cosine similarity for {n_diseases} diseases...")
    
    # Calculate pairwise cosine similarity
    for i in range(n_diseases):
        for j in range(i, n_diseases):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity = 1 - cosine distance
                # If either vector is all zeros, similarity is 0
                norm_i = np.linalg.norm(matrix[i])
                norm_j = np.linalg.norm(matrix[j])
                
                if norm_i == 0 or norm_j == 0:
                    similarity = 0.0
                else:
                    similarity = 1 - cosine(matrix[i], matrix[j])
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_diseases} diseases...")
    
    return similarity_matrix


def create_disease_network(similarity_matrix, disease_names, threshold=0.0):
    """
    Create a disease-disease network from similarity matrix.
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Symmetric matrix of disease similarities
    disease_names : list
        List of disease names
    threshold : float
        Minimum similarity threshold for edge creation (default: 0.0)
        
    Returns:
    --------
    edgelist : pandas.DataFrame
        DataFrame with columns: disease1, disease2, similarity
    """
    print(f"\nCreating disease network with threshold {threshold}...")
    
    edges = []
    n_diseases = len(disease_names)
    
    for i in range(n_diseases):
        for j in range(i + 1, n_diseases):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                edges.append({
                    'disease1': disease_names[i],
                    'disease2': disease_names[j],
                    'similarity': similarity
                })
    
    edgelist = pd.DataFrame(edges)
    print(f"Created network with {len(edgelist)} edges")
    
    if len(edgelist) > 0:
        print(f"Similarity range: [{edgelist['similarity'].min():.6f}, {edgelist['similarity'].max():.6f}]")
        print(f"Mean similarity: {edgelist['similarity'].mean():.6f}")
    
    return edgelist


def save_network(edgelist, output_path):
    """
    Save disease network to file.
    
    Parameters:
    -----------
    edgelist : pandas.DataFrame
        DataFrame with disease-disease edges
    output_path : str
        Path to save the edgelist
    """
    edgelist.to_csv(output_path, sep='\t', index=False)
    print(f"\nNetwork saved to: {output_path}")


def generate_network_statistics(edgelist, disease_names):
    """
    Generate basic network statistics.
    
    Parameters:
    -----------
    edgelist : pandas.DataFrame
        DataFrame with disease-disease edges
    disease_names : list
        List of all disease names
    """
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    print(f"Total diseases: {len(disease_names)}")
    print(f"Total edges: {len(edgelist)}")
    
    if len(edgelist) > 0:
        # Create NetworkX graph for additional statistics
        G = nx.Graph()
        for _, row in edgelist.iterrows():
            G.add_edge(row['disease1'], row['disease2'], weight=row['similarity'])
        
        print(f"Connected components: {nx.number_connected_components(G)}")
        print(f"Average degree: {sum(dict(G.degree()).values()) / len(G):.2f}")
        print(f"Density: {nx.density(G):.6f}")
        
        # Largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Largest connected component size: {len(largest_cc)}")


def main():
    """Main execution function."""
    
    # Setup paths
    data_path = Path(__file__).parent.parent / 'data' / 'replication' / 'data3.txt'
    output_path = Path(__file__).parent.parent / 'data' / 'replication' / 'disease_similarity_network.tsv'
    
    # Load data
    print("="*60)
    print("DISEASE-DISEASE SIMILARITY NETWORK GENERATOR")
    print("="*60)
    df = load_symptom_disease_data(data_path)
    
    # Calculate TF-IDF weights
    disease_vectors, all_symptoms = calculate_tfidf_weights(df)
    
    # Create feature matrix
    matrix, disease_names = create_disease_feature_matrix(disease_vectors, all_symptoms)
    
    # Calculate cosine similarity
    similarity_matrix = calculate_cosine_similarity(matrix)
    
    # Create network (using threshold 0 to include all edges)
    # You can adjust the threshold to filter weak connections
    edgelist = create_disease_network(similarity_matrix, disease_names, threshold=0.0)
    
    # Save network
    save_network(edgelist, output_path)
    
    # Generate statistics
    generate_network_statistics(edgelist, disease_names)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
