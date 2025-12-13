import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np

# ---------------------------
# Config
# ---------------------------
PUBMED_FILE = Path("data/pubmed25n0001.xml.gz")  # change to your downloaded file
DISEASES_TSV = Path("data/mesh_diseases.tsv")
SYMPTOMS_TSV = Path("data/mesh_symptoms.tsv")
MIN_ARTICLES_PER_SYMPTOM = 3        # drop very rare symptoms
MIN_SHARED_SYMPTOMS_EDGE = 3         # min shared symptoms to create disease-disease edge
VIS_EDGE_TOP_K = 50000               # cap edges for visualization

# ---------------------------
# Load MeSH sets
# ---------------------------
def load_mesh_sets():
    diseases = pd.read_csv(DISEASES_TSV, sep="\t")['mesh_ui'].tolist()
    symptoms = pd.read_csv(SYMPTOMS_TSV, sep="\t")['mesh_ui'].tolist()
    return set(diseases), set(symptoms)

# ---------------------------
# Parse PubMed XML (gz) streaming
# ---------------------------
def iter_pubmed_records(gz_path: Path):
    with gzip.open(gz_path, 'rb') as f:
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag == 'PubmedArticle':
                yield elem
                elem.clear()

# Extract MeSH descriptors for an article
# Returns sets of disease UIs, symptom UIs

def extract_mesh_sets(article_elem, disease_set, symptom_set, require_major=False):
    diseases = set()
    symptoms = set()
    for mh in article_elem.findall('.//MeshHeading'):
        desc = mh.find('DescriptorName')
        if desc is None:
            continue
        ui = desc.get('UI')
        if not ui:
            continue
        if require_major and desc.get('MajorTopicYN') != 'Y':
            continue
        if ui in disease_set:
            diseases.add(ui)
        if ui in symptom_set:
            symptoms.add(ui)
    return diseases, symptoms

# ---------------------------
# Build counts
# ---------------------------
def build_counts(pubmed_file, disease_set, symptom_set, require_major=False):
    d2s = defaultdict(Counter)  # disease -> symptom -> count
    symptom_freq = Counter()
    disease_freq = Counter()
    total_articles = 0

    for article in iter_pubmed_records(pubmed_file):
        total_articles += 1
        d_set, s_set = extract_mesh_sets(article, disease_set, symptom_set, require_major=require_major)
        if not d_set or not s_set:
            continue
        for d in d_set:
            disease_freq[d] += 1
            for s in s_set:
                d2s[d][s] += 1
                symptom_freq[s] += 1
    return d2s, disease_freq, symptom_freq, total_articles

# ---------------------------
# Build bipartite graph
# ---------------------------
def build_bipartite(d2s, symptom_freq, min_symptom_freq):
    B = nx.Graph()
    for d, s_counts in d2s.items():
        B.add_node(d, node_type='disease', label=d)
        for s, w in s_counts.items():
            if symptom_freq[s] < min_symptom_freq:
                continue
            if s not in B:
                B.add_node(s, node_type='symptom', label=s)
            B.add_edge(d, s, weight=w)
    return B

# ---------------------------
# Disease-disease projection (Jaccard on symptom sets)
# ---------------------------
def project_disease_network(B, min_shared):
    # precompute disease -> symptoms
    disease_nodes = [n for n, d in B.nodes(data=True) if d.get('node_type') == 'disease']
    symptom_neighbors = {d: set(B.neighbors(d)) for d in disease_nodes}
    G = nx.Graph()
    for d in disease_nodes:
        G.add_node(d)
    n = len(disease_nodes)
    for i in range(n):
        if i % 500 == 0:
            print(f"  projecting {i}/{n}")
        d1 = disease_nodes[i]
        s1 = symptom_neighbors[d1]
        for j in range(i+1, n):
            d2 = disease_nodes[j]
            s2 = symptom_neighbors[d2]
            inter = s1 & s2
            if len(inter) < min_shared:
                continue
            union = s1 | s2
            jaccard = len(inter) / len(union) if union else 0
            G.add_edge(d1, d2, weight=len(inter), jaccard=jaccard)
    return G

# ---------------------------
# Visualization subset
# ---------------------------
def make_pyvis(G, out_html, top_k_edges=50000):
    net = Network(height='900px', width='100%', bgcolor='#ffffff', font_color='#000000', notebook=False)
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get('jaccard', e[2].get('weight', 1)), reverse=True)
    edges = edges[:top_k_edges]

    nodes_in_edges = set()
    for u, v, _ in edges:
        nodes_in_edges.add(u)
        nodes_in_edges.add(v)

    for n in nodes_in_edges:
        deg = G.degree(n)
        net.add_node(n, label=n, title=f"{n}\nDegree: {deg}", size=10 + deg*0.2, color='#ff6b6b')
    for u, v, d in edges:
        w = d.get('jaccard', d.get('weight', 1))
        net.add_edge(u, v, value=w, title=f"shared={d.get('weight', '?')}, jaccard={d.get('jaccard', 0):.3f}")
    net.write_html(out_html)

# ---------------------------
# Main
# ---------------------------
def main():
    if not PUBMED_FILE.exists():
        raise FileNotFoundError(f"Missing PubMed file: {PUBMED_FILE}")
    disease_set, symptom_set = load_mesh_sets()
    print(f"Loaded MeSH sets: diseases={len(disease_set)}, symptoms={len(symptom_set)}")

    print(f"\nParsing PubMed: {PUBMED_FILE}")
    d2s, disease_freq, symptom_freq, total_articles = build_counts(PUBMED_FILE, disease_set, symptom_set, require_major=False)
    print(f"Articles processed: {total_articles}")
    print(f"Diseases with symptoms: {len(d2s)}")

    print("\nBuilding bipartite graph ...")
    B = build_bipartite(d2s, symptom_freq, MIN_ARTICLES_PER_SYMPTOM)
    print(f"Bipartite nodes: {B.number_of_nodes()}, edges: {B.number_of_edges()}")

    print("\nProjecting disease-disease network ...")
    G = project_disease_network(B, MIN_SHARED_SYMPTOMS_EDGE)
    print(f"Disease network: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    print("\nCreating visualization ...")
    make_pyvis(G, "mesh_disease_network.html", top_k_edges=VIS_EDGE_TOP_K)
    print("Saved mesh_disease_network.html")

    print("Saving GEXF ...")
    nx.write_gexf(B, "mesh_bipartite.gexf")
    nx.write_gexf(G, "mesh_disease_network.gexf")
    print("Done.")


if __name__ == "__main__":
    main()
