import pandas as pd
import networkx as nx
from pyvis.network import Network
import obonet

# -------------------------------
# 1. Load HPO ontology
# -------------------------------
print("Loading HPO ontology...")
hpo_graph = obonet.read_obo("data/HPO/hp.obo")
print(f"Loaded HPO: {hpo_graph.number_of_nodes()} terms")

# -------------------------------
# 2. Load disease-phenotype annotations
# -------------------------------
print("\nLoading disease-phenotype annotations...")
df = pd.read_csv('data/HPO/phenotype.hpoa', sep='\t', comment='#')
print(f"Total annotations: {len(df)}")
print(f"Unique diseases: {df['database_id'].nunique()}")
print(f"Unique phenotypes: {df['hpo_id'].nunique()}")

# -------------------------------
# 3. Build disease-phenotype network
# -------------------------------
print("\nBuilding disease-phenotype network...")
G = nx.Graph()

# Add disease and phenotype nodes with their attributes
disease_phenotype_pairs = df[['database_id', 'disease_name', 'hpo_id']].dropna().drop_duplicates()
print(disease_phenotype_pairs.head())
for _, row in disease_phenotype_pairs.iterrows():
    disease_id = row['database_id']
    disease_name = row['disease_name']
    hpo_id = row['hpo_id']
    
    # Skip if HPO ID is not in the ontology
    if hpo_id not in hpo_graph:
        continue
    
    # Add disease node
    if disease_id not in G:
        G.add_node(disease_id, 
                   node_type='disease',
                   label=disease_name,
                   name=disease_name)
    
    # Add phenotype node
    if hpo_id not in G:
        phenotype_name = hpo_graph.nodes[hpo_id].get('name', hpo_id)
        G.add_node(hpo_id, 
                   node_type='phenotype',
                   label=phenotype_name,
                   name=phenotype_name)
    
    # Add edge between disease and phenotype
    G.add_edge(disease_id, hpo_id)

print(f"\nNetwork statistics:")
print(f"  Total nodes: {G.number_of_nodes()}")
print(f"  Total edges: {G.number_of_edges()}")

# Count node types
disease_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'disease']
phenotype_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'phenotype']
print(f"  Disease nodes: {len(disease_nodes)}")
print(f"  Phenotype nodes: {len(phenotype_nodes)}")

# -------------------------------
# 4. Network analysis
# -------------------------------
print("\nNetwork analysis:")
# Most connected diseases
disease_degrees = [(n, G.degree(n)) for n in disease_nodes]
top_diseases = sorted(disease_degrees, key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 diseases by number of phenotypes:")
for disease_id, degree in top_diseases:
    print(f"  {disease_id}: {G.nodes[disease_id]['name']} - {degree} phenotypes")

# Most connected phenotypes
phenotype_degrees = [(n, G.degree(n)) for n in phenotype_nodes]
top_phenotypes = sorted(phenotype_degrees, key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 most common phenotypes:")
for hpo_id, degree in top_phenotypes:
    print(f"  {hpo_id}: {G.nodes[hpo_id]['name']} - {degree} diseases")

# Save bipartite network for reference as TSV
print("\nSaving bipartite disease-phenotype network to 'disease_phenotype_network.tsv'...")
with open('disease_phenotype_network.tsv', 'w') as f:
    f.write("Source\tTarget\n")
    for u, v in G.edges():
        f.write(f"{u}\t{v}\n")

# -------------------------------
# 5. Create disease-disease projection network (OPTIMIZED)
# -------------------------------
print("\nCreating disease-disease projection network...")

# Optimized projection: manually compute shared phenotypes
# Much faster than NetworkX's bipartite projection for large networks
from collections import defaultdict

print("Computing shared phenotypes between diseases...")
disease_disease_net = nx.Graph()

# Build disease -> phenotypes mapping
disease_to_phenotypes = defaultdict(set)
for disease in disease_nodes:
    disease_to_phenotypes[disease] = set(G.neighbors(disease))
    # Copy node attributes
    disease_disease_net.add_node(disease, **G.nodes[disease])

# Compute shared phenotypes for each disease pair
disease_list = list(disease_nodes)
total_pairs = len(disease_list) * (len(disease_list) - 1) // 2
print(f"Processing {len(disease_list)} diseases ({total_pairs:,} potential pairs)...")

# Only connect diseases with minimum shared phenotypes (filter early)
min_shared = 20  # Minimum shared phenotypes to create an edge (increased for meaningful connections)
edges_added = 0

for i in range(len(disease_list)):
    if i % 500 == 0:
        print(f"  Processed {i}/{len(disease_list)} diseases, {edges_added:,} edges added")
    
    disease1 = disease_list[i]
    phenotypes1 = disease_to_phenotypes[disease1]
    
    for j in range(i + 1, len(disease_list)):
        disease2 = disease_list[j]
        phenotypes2 = disease_to_phenotypes[disease2]
        
        # Count shared phenotypes
        shared = len(phenotypes1 & phenotypes2)
        
        # Only add edge if sufficient overlap
        if shared >= min_shared:
            disease_disease_net.add_edge(disease1, disease2, weight=shared)
            edges_added += 1

# save disease-disease network as TSV
print("\nSaving disease-disease network to 'disease_disease_network.tsv'...")
with open('disease_disease_network.tsv', 'w') as f:
    f.write("disease1\tdisease2\tweight\n")
    for u, v, data in disease_disease_net.edges(data=True):
        f.write(f"{u}\t{v}\t{data['weight']}\n")

print(f"\nDisease-disease network statistics:")
print(f"  Diseases: {disease_disease_net.number_of_nodes()}")
print(f"  Disease pairs connected: {disease_disease_net.number_of_edges()}")

# Calculate network metrics (skip expensive betweenness for large networks)
print("\nCalculating network metrics...")
degree_centrality = nx.degree_centrality(disease_disease_net)

# Skip betweenness if network is too large
if disease_disease_net.number_of_nodes() < 2000:
    betweenness = nx.betweenness_centrality(disease_disease_net, k=min(1000, len(disease_nodes)))
else:
    print("  Skipping betweenness centrality (network too large)")
    betweenness = None

# Find most central diseases
top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
print("\nTop 20 most connected diseases (by degree centrality):")
for disease_id, centrality in top_central:
    name = disease_disease_net.nodes[disease_id].get('name', disease_id)
    degree = disease_disease_net.degree(disease_id)
    print(f"  {disease_id}: {name}")
    print(f"    Connections: {degree}, Centrality: {centrality:.4f}")

# Detect communities
print("\nDetecting disease communities...")
from networkx.algorithms import community
communities = community.louvain_communities(disease_disease_net, seed=42)
print(f"Found {len(communities)} disease communities")

# Assign community colors
community_map = {}
for idx, comm in enumerate(communities):
    for disease in comm:
        community_map[disease] = idx

exit()
# -------------------------------
# 6. Visualize disease-disease network
# -------------------------------
print("\nCreating disease-disease visualization...")

# For better visualization, use adaptive filtering based on network density
# Calculate percentile threshold for edge weights
edge_weights = [d['weight'] for _, _, d in disease_disease_net.edges(data=True)]
if len(edge_weights) > 0:
    import numpy as np
    # Use 75th percentile or minimum of 15 shared phenotypes, whichever is higher
    weight_threshold = max(15, int(np.percentile(edge_weights, 75)))
else:
    weight_threshold = 15

print(f"Using edge weight threshold: {weight_threshold} shared phenotypes")

filtered_edges = [(u, v, d) for u, v, d in disease_disease_net.edges(data=True) 
                  if d['weight'] >= weight_threshold]

print(f"Filtered to {len(filtered_edges)} edges (>= {weight_threshold} shared phenotypes)")

# Further filter: only keep largest connected component if still too large
if len(filtered_edges) > 50000:
    print("Network still large, keeping only top connections...")
    # Sort edges by weight and keep top 50k
    filtered_edges = sorted(filtered_edges, key=lambda x: x[2]['weight'], reverse=True)[:50000]
    print(f"Reduced to top {len(filtered_edges)} strongest connections")

# Create filtered network for visualization
viz_net = nx.Graph()
for u, v, d in filtered_edges:
    viz_net.add_edge(u, v, weight=d['weight'])
    # Copy node attributes
    if u not in viz_net.nodes or 'name' not in viz_net.nodes[u]:
        viz_net.nodes[u].update(disease_disease_net.nodes[u])
    if v not in viz_net.nodes or 'name' not in viz_net.nodes[v]:
        viz_net.nodes[v].update(disease_disease_net.nodes[v])

print(f"Visualization network: {viz_net.number_of_nodes()} diseases, {viz_net.number_of_edges()} edges")

# Initialize PyVis network
net = Network(
    height="900px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#000000",
    notebook=False
)

# Configure physics for community clustering
net.set_options("""
{
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.015,
      "springLength": 150,
      "springConstant": 0.05,
      "damping": 0.5,
      "avoidOverlap": 0.8
    },
    "solver": "forceAtlas2Based",
    "stabilization": {
      "enabled": true,
      "iterations": 2000
    }
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true,
    "tooltipDelay": 100,
    "multiselect": true
  },
  "nodes": {
    "font": {
      "size": 11
    },
    "borderWidth": 2,
    "borderWidthSelected": 4
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "continuous"
    },
    "color": {
      "inherit": false
    }
  }
}
""")

# Community colors
community_colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
    '#27ae60', '#2980b9', '#8e44ad', '#d35400', '#7f8c8d'
]

# Add nodes with community coloring
for node_id in viz_net.nodes():
    name = viz_net.nodes[node_id].get('name', node_id)
    degree = viz_net.degree(node_id)
    community_id = community_map.get(node_id, 0)
    
    # Calculate total shared phenotypes
    total_shared = sum([viz_net[node_id][neighbor]['weight'] 
                       for neighbor in viz_net.neighbors(node_id)])
    
    # Create detailed tooltip
    tooltip = f"<b>{name}</b><br>"
    tooltip += f"ID: {node_id}<br>"
    tooltip += f"Connected to: {degree} diseases<br>"
    tooltip += f"Total shared phenotypes: {total_shared}<br>"
    tooltip += f"Community: {community_id + 1}"
    
    # Node color by community
    color = community_colors[community_id % len(community_colors)]
    
    # Node size by degree
    size = 15 + (degree * 0.8)
    
    net.add_node(
        node_id,
        label=name,
        title=tooltip,
        color=color,
        size=min(size, 60),
        borderWidth=2
    )

# Add edges with width based on shared phenotypes
for u, v, data in viz_net.edges(data=True):
    weight = data['weight']
    # Edge width proportional to shared phenotypes
    edge_width = min(1 + (weight * 0.3), 10)
    # Edge color based on weight
    if weight >= 10:
        edge_color = '#e74c3c'  # Red for strong connections
    elif weight >= 5:
        edge_color = '#f39c12'  # Orange for medium
    else:
        edge_color = '#95a5a6'  # Gray for weak
    
    net.add_edge(
        u, v,
        width=edge_width,
        color=edge_color,
        title=f"{weight} shared phenotypes"
    )

# Save visualization
output_file = "disease_disease_network.html"
net.write_html(output_file)

print(f"\n✓ Disease-disease network saved to '{output_file}'")
print(f"\nVisualization details:")
print(f"  Node colors = Disease communities (detected by Louvain algorithm)")
print(f"  Node size = Number of disease connections")
print(f"  Edge width = Number of shared phenotypes")
print(f"  Edge colors:")
print(f"    Red = Strong similarity (>=10 shared phenotypes)")
print(f"    Orange = Medium similarity (5-9 shared phenotypes)")
print(f"    Gray = Weak similarity (3-4 shared phenotypes)")
print(f"\nControls:")
print(f"  - Zoom with mouse wheel")
print(f"  - Pan by clicking and dragging")
print(f"  - Hover over nodes/edges for details")
print(f"  - Click and drag nodes to rearrange")

# -------------------------------
# 7. Save network data
# -------------------------------
print("\nSaving network data...")
nx.write_gexf(G, "bipartite_disease_phenotype.gexf")
nx.write_gexf(disease_disease_net, "disease_disease_full.gexf")
nx.write_gexf(viz_net, "disease_disease_filtered.gexf")
print("✓ Networks saved:")
print("  - bipartite_disease_phenotype.gexf (full bipartite network)")
print("  - disease_disease_full.gexf (all disease-disease connections)")
print("  - disease_disease_filtered.gexf (filtered for visualization)")
