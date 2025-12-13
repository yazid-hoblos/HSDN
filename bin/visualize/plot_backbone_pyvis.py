"""
Interactive PyVis plot of the consensus backbone using disease names instead of IDs.

Inputs:
- Backbone edges (default): data/replication/consensus_symptom_geneppi_backbone.tsv
  Columns: Source, Target, weight, alpha
- Mapping edges (default): data/replication/consensus_symptom_geneppi_edges.tsv
  Columns include: Source, Target, source_name, target_name

Outputs:
- HTML: data/replication/consensus_backbone_pyvis.html

Usage:
  python bin/plot_backbone_pyvis.py \
    --backbone data/replication/consensus_symptom_geneppi_backbone.tsv \
    --edges data/replication/consensus_symptom_geneppi_edges.tsv \
    --top-n 300 \
    --output data/replication/consensus_backbone_pyvis.html
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
    parser.add_argument('--backbone', type=Path, default=Path('data/replication/consensus_symptom_geneppi_backbone.tsv'))
    parser.add_argument('--edges', type=Path, default=Path('data/replication/consensus_symptom_geneppi_edges.tsv'))
    parser.add_argument('--output', type=Path, default=Path('data/replication/consensus_backbone_pyvis.html'))
    parser.add_argument('--top-n', type=int, default=300, help='Top-N nodes by strength to plot')
    parser.add_argument('--physics', action='store_true', help='Enable physics (default off for stable layout)')
    parser.add_argument('--no-community', action='store_true', help='Disable community coloring')
    args = parser.parse_args()

    if not args.backbone.exists():
        raise FileNotFoundError(f"Backbone file not found: {args.backbone}")
    if not args.edges.exists():
        raise FileNotFoundError(f"Edges file with names not found: {args.edges}")

    df_backbone = pd.read_csv(args.backbone, sep='\t')
    # if args.top_n:
        # df_backbone = select_top_nodes(df_backbone, args.top_n)

    name_map = load_names_map(args.edges)

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


if __name__ == '__main__':
    main()
