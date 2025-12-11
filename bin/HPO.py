import obonet
import networkx as nx
import os
import matplotlib.pyplot as plt
from pyvis.network import Network

# -------------------------------
# 1. Load local HPO OBO file
# -------------------------------
obo_file = "hp.obo"  # Path to your file
if not os.path.exists(obo_file):
    raise FileNotFoundError(f"{obo_file} not found.")

print("Loading HPO ontology from local file...")
graph = obonet.read_obo(obo_file)
print(f"Loaded HPO: {graph.number_of_nodes()} terms, {graph.number_of_edges()} relationships.")

# -------------------------------
# 2. Helper functions
# -------------------------------
def get_children(term_id, g):
    """Return direct children of a term"""
    return list(g.predecessors(term_id))

def get_descendants(term_id, g):
    """Recursively get all descendants of a term"""
    descendants = []
    children = get_children(term_id, g)
    for child in children:
        descendants.append(child)
        descendants.extend(get_descendants(child, g))
    return descendants

# -------------------------------
# 3. Top-level symptom categories
# -------------------------------
root = 'HP:0000118'  # Phenotypic abnormality
top_level_terms = get_children(root, graph)

print("\nTop-level symptom categories:")
for tid in top_level_terms:
    name = graph.nodes[tid].get('name', '(no name)')
    print(f"{tid} - {name}")

# -------------------------------
# 4. Extract descendants for a category (optional)
# -------------------------------
# Example: nervous system
# nervous_system = 'HP:0000707'
# nerv_descendants = get_descendants(nervous_system, graph)
# print(f"\n'{graph.nodes[nervous_system]['name']}' has {len(nerv_descendants)} descendants.")

# -------------------------------
# 5a. Visualize top-level categories + first-level children (Matplotlib)
# -------------------------------
viz_graph = nx.DiGraph()
viz_graph.add_node(root, label=graph.nodes[root]['name'])
for tid in top_level_terms:
    viz_graph.add_node(tid, label=graph.nodes[tid]['name'])
    viz_graph.add_edge(root, tid)
    # for c in get_children(tid, graph):
    #     viz_graph.add_node(c, label=graph.nodes[c]['name'])
    #     viz_graph.add_edge(tid, c)

# plt.figure(figsize=(15, 10))
# pos = nx.spring_layout(viz_graph, seed=42)
# labels = nx.get_node_attributes(viz_graph, 'label')
# nx.draw(viz_graph, pos, with_labels=True, labels=labels, node_size=1500, node_color='skyblue', font_size=10, arrowsize=20)
# plt.title("HPO Top-Level Categories and First-Level Children")
# plt.show()

# -------------------------------
# 5b. Interactive visualization with PyVis
# -------------------------------
net = Network(height="800px", width="100%", directed=True)
for node in viz_graph.nodes():
    net.add_node(node, label=viz_graph.nodes[node]['label'])
for source, target in viz_graph.edges():
    net.add_edge(source, target)
    
# color nodes by level
for node in net.nodes:
    if node['id'] == root:
        node['color'] = 'red'
    elif node['id'] in top_level_terms:
        node['color'] = 'orange'
    else:
        node['color'] = 'lightblue'
net.write_html("hpo_top_level.html")
print("\nInteractive graph saved to 'hpo_top_level.html'. Open in a browser to explore.")
