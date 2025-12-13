import pandas as pd
from pathlib import Path
import requests
import json
from collections import defaultdict, Counter
import networkx as nx
from pyvis.network import Network
import time

# ---------------------------
# Config
# ---------------------------
HPO_ANNOTATIONS = Path("data/HPO/phenotype.hpoa")  # HPO phenotype-disease file
UMLS_API_KEY = ""  # Get from https://uts.nlm.nih.gov/home.html (free registration)
SYMPTOM_SEMANTIC_TYPE = "T184"  # UMLS semantic type for 'Sign or Symptom'

OUTPUT_DIR = Path("data")

# ---------------------------
# 1. Load HPO annotations
# ---------------------------
def load_hpo_annotations(path):
    """Load HPO phenotype annotation file."""
    # Typical format: DB, DB_ID (OMIM:XXXX), Qualifier, HPO_ID, Reference, Evidence, Onset, Frequency, Sex, Modifier, Aspect, Biocuration
    df = pd.read_csv(path, sep='\t', comment='#', dtype={'database_id': str, 'hpo_id': str})
    print(f"Loaded HPO annotations: {len(df)} rows")
    print(df.head())
    return df

# ---------------------------
# 2. Filter to symptoms via UMLS API
# ---------------------------
def get_umls_semantic_type(hpo_id, api_key, cache=None):
    """
    Query UMLS REST API for semantic type of an HPO term.
    Cache results to avoid repeated API calls.
    """
    if cache is None:
        cache = {}
    if hpo_id in cache:
        return cache[hpo_id]
    
    try:
        # Search UMLS for HPO code
        url = "https://uts-ws.nlm.nih.gov/rest/search"
        params = {
            "string": hpo_id,
            "searchType": "exact",
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        if data.get('result', {}).get('results'):
            # Get semantic type from first result
            result = data['result']['results'][0]
            sem_type = result.get('semanticTypes', [])
            cache[hpo_id] = sem_type
            return sem_type
        return []
    except Exception as e:
        print(f"  Error querying {hpo_id}: {e}")
        return []

# ---------------------------
# Alternative: Load pre-cached UMLS mappings (if available)
# ---------------------------
def load_umls_cache_file(path):
    """Load a pre-built file of HPO -> UMLS semantic types (to avoid API hammering)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep='\t')
    # Assume columns: hpo_id, semantic_type
    result = {}
    for _, row in df.iterrows():
        hpo = row['hpo_id']
        sem = row['semantic_type']
        if hpo not in result:
            result[hpo] = []
        result[hpo].append(sem)
    return result

# ---------------------------
# 3. Extract symptoms & OMIM diseases
# ---------------------------
def filter_to_symptoms(df, api_key=None, umls_cache_file=None):
    """
    Filter HPO records to symptoms only (UMLS semantic type T184).
    Requires either API key or pre-cached UMLS file.
    """
    if umls_cache_file:
        print(f"Loading UMLS cache from {umls_cache_file}...")
        umls_map = load_umls_cache_file(umls_cache_file)
    else:
        umls_map = {}
    
    symptom_records = []
    unique_hpos = df['hpo_id'].unique()
    print(f"Checking {len(unique_hpos)} unique HPO terms for symptom classification...")
    
    for i, hpo_id in enumerate(unique_hpos):
        if i % 100 == 0:
            print(f"  {i}/{len(unique_hpos)}")
        
        sem_types = umls_map.get(hpo_id)
        if sem_types is None and api_key:
            # Query API (rate-limited)
            sem_types = get_umls_semantic_type(hpo_id, api_key, cache=umls_map)
            time.sleep(0.1)  # Rate limiting
        
        if sem_types and SYMPTOM_SEMANTIC_TYPE in sem_types:
            # This HPO term is a symptom; keep all records with it
            symptom_records.append(hpo_id)
    
    symptom_df = df[df['hpo_id'].isin(symptom_records)].copy()
    print(f"\nFiltered to {len(symptom_df)} symptom records")
    print(f"  Unique diseases: {symptom_df['database_id'].nunique()}")
    print(f"  Unique symptoms: {len(symptom_records)}")
    return symptom_df, symptom_records

# ---------------------------
# 4. Extract OMIM IDs
# ---------------------------
def extract_omim_diseases(df):
    """Extract OMIM disease IDs from database_id column."""
    # Assume format: 'OMIM:XXXXX'
    omim_records = df[df['database_id'].str.startswith('OMIM:', na=False)].copy()
    omim_records['omim_id'] = omim_records['database_id'].str.replace('OMIM:', '')
    print(f"\nOMIM records: {len(omim_records)}")
    print(f"  Unique OMIM IDs: {omim_records['omim_id'].nunique()}")
    return omim_records

# ---------------------------
# 5. Map OMIM to MeSH (placeholder)
# ---------------------------
def load_omim_mesh_mapping(path=None):
    """
    Load a pre-built OMIM -> MeSH mapping.
    You may need to build this separately using UMLS Metathesaurus or other sources.
    Format: TSV with columns 'omim_id', 'mesh_id', 'mesh_name'
    """
    if path and Path(path).exists():
        df = pd.read_csv(path, sep='\t')
        mapping = {}
        for _, row in df.iterrows():
            omim = str(row['omim_id'])
            mesh = row['mesh_id']
            if omim not in mapping:
                mapping[omim] = []
            mapping[omim].append(mesh)
        return mapping
    else:
        print("WARNING: No OMIM->MeSH mapping file provided.")
        print("  You need to build or download this separately.")
        return {}

# ---------------------------
# 6. Build disease-symptom network (on HPO or MeSH)
# ---------------------------
def build_disease_symptom_network(df, omim_mesh_map=None):
    """
    Build a bipartite disease-symptom network.
    If omim_mesh_map provided, use MeSH IDs; otherwise use OMIM.
    """
    G = nx.Graph()
    
    for _, row in df.iterrows():
        disease_id = row['database_id']
        symptom_id = row['hpo_id']
        
        # Optionally map to MeSH
        if omim_mesh_map:
            omim = disease_id.replace('OMIM:', '')
            mesh_ids = omim_mesh_map.get(omim, [])
            if not mesh_ids:
                continue
            disease_id = mesh_ids[0]  # Use first mapping
        
        G.add_node(disease_id, node_type='disease', label=disease_id)
        G.add_node(symptom_id, node_type='symptom', label=symptom_id)
        G.add_edge(disease_id, symptom_id, weight=1)
    
    return G

# ---------------------------
# 7. Project to disease-disease
# ---------------------------
def project_to_disease_disease(G):
    """Project bipartite to disease-disease (shared symptoms)."""
    disease_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'disease']
    symptom_neighbors = {d: set(G.neighbors(d)) for d in disease_nodes}
    
    D = nx.Graph()
    for d in disease_nodes:
        D.add_node(d)
    
    n = len(disease_nodes)
    for i in range(n):
        if i % 500 == 0:
            print(f"  projecting {i}/{n}")
        d1 = disease_nodes[i]
        s1 = symptom_neighbors[d1]
        for j in range(i+1, n):
            d2 = disease_nodes[j]
            s2 = symptom_neighbors[d2]
            shared = s1 & s2
            if len(shared) >= 3:  # min shared symptoms
                D.add_edge(d1, d2, weight=len(shared))
    
    return D

# ---------------------------
# 8. Visualize
# ---------------------------
def visualize_network(G, out_html, top_k=50000):
    """Create PyVis visualization."""
    net = Network(height='900px', width='100%', bgcolor='#ffffff', notebook=False)
    
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get('weight', 1), reverse=True)
    edges = edges[:top_k]
    
    nodes_set = set()
    for u, v, _ in edges:
        nodes_set.add(u)
        nodes_set.add(v)
    
    for n in nodes_set:
        deg = G.degree(n)
        net.add_node(n, label=n, size=10 + deg*0.3, color='#ff6b6b', title=f"{n} (degree={deg})")
    
    for u, v, d in edges:
        net.add_edge(u, v, value=d.get('weight', 1))
    
    net.write_html(out_html)

# ---------------------------
# Main
# ---------------------------
def main():
    print("="*60)
    print("HPO-based Disease-Symptom Network Builder")
    print("="*60)
    
    # Step 1: Load HPO annotations
    if not HPO_ANNOTATIONS.exists():
        print(f"ERROR: {HPO_ANNOTATIONS} not found.")
        print("Download from http://www.human-phenotype-ontology.org/")
        return
    
    df = load_hpo_annotations(HPO_ANNOTATIONS)
    
    # Step 2: Filter to symptoms
    # Option A: Use UMLS API (requires key)
    # Option B: Use pre-cached UMLS semantic types file
    umls_cache = Path("data/umls_semantic_types.tsv")  # You need to create this
    
    if umls_cache.exists():
        symptom_df, symptom_ids = filter_to_symptoms(df, umls_cache_file=umls_cache)
    else:
        print("\nWARNING: No UMLS cache file found.")
        print("  To use UMLS API, set UMLS_API_KEY in this script.")
        print("  Alternatively, create data/umls_semantic_types.tsv with columns: hpo_id, semantic_type")
        print("\nProceeding with all HPO terms as 'symptoms' for now...")
        symptom_df = df
        symptom_ids = df['hpo_id'].unique()
    
    # Step 3: Extract OMIM
    omim_df = extract_omim_diseases(symptom_df)
    
    # Step 4: Load OMIM->MeSH mapping (if available)
    omim_mesh_path = Path("data/omim_mesh_mapping.tsv")
    omim_mesh_map = load_omim_mesh_mapping(omim_mesh_path) if omim_mesh_path.exists() else None
    
    if omim_mesh_map:
        print(f"Loaded OMIM->MeSH mapping: {len(omim_mesh_map)} OMIM IDs")
    
    # Step 5: Build bipartite
    print("\nBuilding bipartite disease-symptom network...")
    G = build_disease_symptom_network(omim_df, omim_mesh_map)
    print(f"Bipartite: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 6: Project
    print("\nProjecting to disease-disease network...")
    D = project_to_disease_disease(G)
    print(f"Disease-disease: {D.number_of_nodes()} nodes, {D.number_of_edges()} edges")
    
    # Step 7: Export
    print("\nExporting...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, OUTPUT_DIR / "hpo_bipartite.gexf")
    nx.write_gexf(D, OUTPUT_DIR / "hpo_disease_network.gexf")
    
    print("\nVisualizing...")
    visualize_network(D, OUTPUT_DIR / "hpo_disease_network.html")
    
    print("\nDone!")
    print(f"  hpo_bipartite.gexf")
    print(f"  hpo_disease_network.gexf")
    print(f"  hpo_disease_network.html")

if __name__ == "__main__":
    main()
