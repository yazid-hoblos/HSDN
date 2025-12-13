import obonet
import networkx as nx
from pyvis.network import Network

# -------------------------------
# 1. Load HPO ontology
# -------------------------------
print("Loading HPO ontology from local file...")
graph = obonet.read_obo("hp.obo")
print(f"Loaded HPO: {graph.number_of_nodes()} terms, {graph.number_of_edges()} relationships.")

# -------------------------------
# 2. Create interactive visualization
# -------------------------------
print("\nBuilding interactive network visualization...")

# Initialize PyVis network with physics settings for better layout
net = Network(
    height="900px",
    width="100%",
    directed=True,
    bgcolor="#ffffff",
    font_color="#000000",
    notebook=False
)

# Configure physics for hierarchical layout
net.set_options("""
{
  "physics": {
    "enabled": true,
    "hierarchicalRepulsion": {
      "centralGravity": 0.0,
      "springLength": 150,
      "springConstant": 0.01,
      "nodeDistance": 200,
      "damping": 0.09
    },
    "solver": "hierarchicalRepulsion"
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed",
      "levelSeparation": 150,
      "nodeSpacing": 200
    }
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true,
    "tooltipDelay": 100
  },
  "nodes": {
    "font": {
      "size": 14,
      "face": "arial"
    },
    "borderWidth": 2,
    "borderWidthSelected": 3
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.5
      }
    },
    "smooth": {
      "enabled": true,
      "type": "cubicBezier"
    }
  }
}
""")

# Add all nodes with metadata
print("Adding nodes...")
for node_id, node_data in graph.nodes(data=True):
    name = node_data.get('name', node_id)
    
    # Create tooltip with additional info
    tooltip = f"<b>{name}</b><br>ID: {node_id}"
    if 'def' in node_data:
        definition = node_data['def'].split('"')[1] if '"' in node_data['def'] else node_data['def']
        tooltip += f"<br><br>{definition[:200]}..."
    
    # Determine node color based on hierarchy level
    # Root is red, top-level categories are orange, rest are blue gradients
    if node_id == 'HP:0000001':
        color = '#ff0000'
        size = 30
    elif node_id == 'HP:0000118':
        color = '#ff6600'
        size = 25
    else:
        # Calculate depth from root for color gradient
        try:
            depth = nx.shortest_path_length(graph, source=node_id, target='HP:0000001')
            # Blue gradient based on depth
            blue_value = max(100, 255 - depth * 15)
            color = f'#0000{blue_value:02x}'
            size = max(10, 20 - depth)
        except:
            color = '#6699ff'
            size = 15
    
    net.add_node(
        node_id,
        label=name,
        title=tooltip,
        color=color,
        size=size,
        shape='dot'
    )

# Add all edges
print("Adding edges...")
for source, target in graph.edges():
    net.add_edge(source, target)

# Save the visualization
output_file = "hpo_full_network.html"
net.write_html(output_file)
print(f"\n✓ Interactive HPO network saved to '{output_file}'")
print(f"  Total nodes: {graph.number_of_nodes()}")
print(f"  Total edges: {graph.number_of_edges()}")
print(f"\nOpen '{output_file}' in your browser to explore the network.")
print("  - Zoom with mouse wheel")
print("  - Pan by clicking and dragging")
print("  - Hover over nodes for details")
print("  - Click nodes to highlight connections")
