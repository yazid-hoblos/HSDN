"""
Interactive PyVis plot of backbone networks with community detection.

Inputs:
- Backbone edges (default): data/replication/consensus_symptom_geneppi_backbone.tsv
  Columns: Source, Target, weight, alpha
- Mapping edges (optional): data/replication/consensus_symptom_geneppi_edges.tsv
  Columns include: Source, Target, source_name, target_name

Outputs:
- HTML: Interactive network visualization
- TSV: Community assignments (node, community_id, label)

Usage:
  # Consensus GenePPI backbone
  python bin/visualize/plot_backbone_pyvis.py \
    --backbone data/replication/consensus_symptom_geneppi_backbone.tsv \
    --edges data/replication/consensus_symptom_geneppi_edges.tsv \
    --output presentation/plots/consensus_geneppi_network.html \
    --communities presentation/plots/consensus_geneppi_communities.tsv
  
  # Original backbone
  python bin/visualize/plot_backbone_pyvis.py \
    --backbone data/replication/filtering/disease_network_backbone.csv \
    --output presentation/plots/original_backbone_network.html \
    --communities presentation/plots/original_backbone_communities.tsv
"""

import argparse
from itertools import cycle
from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from pyvis.network import Network


def load_names_map(edges_path):
    df = pd.read_csv(edges_path, sep='\t')
    name_map = {}
    if 'source_name' in df.columns and 'target_name' in df.columns:
        for _, row in df.iterrows():
            sid = row['Source']
            tid = row['Target']
            if pd.notna(row['source_name']):
                name_map.setdefault(sid, str(row['source_name']))
            if pd.notna(row['target_name']):
                name_map.setdefault(tid, str(row['target_name']))
    return name_map


def select_top_nodes(df, top_n):
    strength = df.groupby('Source')['weight'].sum() + df.groupby('Target')['weight'].sum()
    top_nodes = set(strength.sort_values(ascending=False).head(top_n).index)
    return df[df['Source'].isin(top_nodes) & df['Target'].isin(top_nodes)]


def build_graph(df):
    return nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=['weight'], create_using=nx.Graph())


def detect_communities(G):
    if G.number_of_nodes() == 0:
        return {}
    comms = list(greedy_modularity_communities(G))
    node2comm = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            node2comm[n] = cid
    return node2comm


def palette():
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f",
        "#cab2d6", "#ffff99"
    ]
    # infinite cycle to avoid index errors
    return cycle(colors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=Path, required=True, help='Path to backbone TSV/CSV file')
    parser.add_argument('--edges', type=Path, default=None, help='Path to edges file with name mappings (optional)')
    parser.add_argument('--output', type=Path, required=True, help='Output HTML file path')
    parser.add_argument('--communities', type=Path, default=None, help='Output TSV file for community assignments')
    parser.add_argument('--top-n', type=int, default=300, help='Top-N nodes by strength to plot')
    parser.add_argument('--physics', action='store_true', help='Enable physics (default off for stable layout)')
    parser.add_argument('--no-community', action='store_true', help='Disable community coloring')
    args = parser.parse_args()

    if not args.backbone.exists():
        raise FileNotFoundError(f"Backbone file not found: {args.backbone}")
    
    # Auto-detect separator (CSV vs TSV)
    if args.backbone.suffix == '.csv':
        df_backbone = pd.read_csv(args.backbone)
    else:
        df_backbone = pd.read_csv(args.backbone, sep='\t')
    
    # Standardize column names
    df_backbone = df_backbone.rename(columns={'Weight': 'weight', 'Alpha': 'alpha'})
    
    # Load name map if edges file provided
    name_map = {}
    if args.edges and args.edges.exists():
        name_map = load_names_map(args.edges)
    else:
        print("Note: No edges file provided, using node IDs as labels")

    G = build_graph(df_backbone)

    net = Network(height='900px', width='100%', bgcolor='#ffffff', font_color='#222222', notebook=False, directed=False)
    net.barnes_hut() if args.physics else net.force_atlas_2based(gravity=-35, central_gravity=0.01, spring_length=110, spring_strength=0.04, damping=0.88, overlap=0)

    node2comm = detect_communities(G) if not args.no_community else {}
    color_cycle = palette()
    comm_colors = {}

    # Add nodes with labels
    max_edge_w = max((data['weight'] for _, _, data in G.edges(data=True)), default=1.0)

    for n in G.nodes():
        label = name_map.get(n, n)
        strength = sum(data['weight'] for _, _, data in G.edges(n, data=True))
        size = 10 + 6 * (strength / max(1e-9, max_edge_w))

        comm_id = node2comm.get(n)
        if comm_id is not None:
            if comm_id not in comm_colors:
                comm_colors[comm_id] = next(color_cycle)
            color = comm_colors[comm_id]
        else:
            color = '#4e79a7'

        net.add_node(
            n,
            label=label,
            title=f"ID: {n}<br>Strength: {strength:.3f}<br>Community: {comm_id if comm_id is not None else 'N/A'}",
            value=strength,
            size=size,
            color=color,
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        net.add_edge(u, v, value=w, title=f"weight={w:.4f}", width=0.5 + 3.0 * (w / max_edge_w))

    net.show_buttons(filter_=['physics'])
    net.write_html(str(args.output))
    print(f"Saved PyVis HTML to {args.output}")
    print("Open it in a browser and screenshot for slides if needed.")
    
    # Save community assignments if requested
    if args.communities:
        community_data = []
        for node_id in G.nodes():
            label = name_map.get(node_id, node_id)
            comm_id = node2comm.get(node_id, -1)
            degree = G.degree(node_id)
            strength = sum(data['weight'] for _, _, data in G.edges(node_id, data=True))
            community_data.append({
                'node_id': node_id,
                'label': label,
                'community': comm_id,
                'degree': degree,
                'strength': strength,
            })
        
        comm_df = pd.DataFrame(community_data)
        comm_df = comm_df.sort_values(['community', 'strength'], ascending=[True, False])
        args.communities.parent.mkdir(parents=True, exist_ok=True)
        comm_df.to_csv(args.communities, sep='\t', index=False)
        print(f"Saved community assignments to {args.communities}")
        print(f"   Total nodes: {len(comm_df)}")
        print(f"   Communities detected: {comm_df['community'].nunique()}")


if __name__ == '__main__':
    main()
