"""
Plot OMIM backbone with disease names using PyVis.

Inputs:
- Backbone CSV: e.g., data/replication/disease_disease_network_backbone.csv
  Columns: Source,Target,Weight,Alpha (IDs like OMIM:618505)
- Optional mapping TSV/CSV: columns 'omim_id','name' (or 'OMIM','name') to label nodes.
  If absent, labels fall back to the OMIM IDs.

Usage:
  python bin/plot_omim_backbone_pyvis.py \
    --backbone data/replication/disease_disease_network_backbone.csv \
    --mapping data/replication/omim_name_map.tsv \
    --output data/replication/omim_backbone.html \
    --top-n 300
"""

import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities


def load_backbone(path: Path):
    df = pd.read_csv(path)
    for col in ['Source','Target','Weight']:
        if col not in df.columns:
            raise ValueError(f"Backbone missing column: {col}")
    return df


def load_mapping(map_path: Path):
    if not map_path or not map_path.exists():
        return {}
    # auto-detect sep
    sep = '\t' if map_path.suffix.lower() in ['.tsv','.txt'] else ','
    df = pd.read_csv(map_path, sep=sep)
    # normalize columns
    cols = set(df.columns.str.lower())
    if {'omim_id','name'}.issubset(cols):
        id_col = 'omim_id'
        name_col = 'name'
    elif {'omim','name'}.issubset(cols):
        id_col = 'omim'
        name_col = 'name'
    else:
        # try first two columns
        id_col = df.columns[0]
        name_col = df.columns[1]
    mapping = {}
    for _, row in df.iterrows():
        omim = str(row[id_col]).strip()
        if not omim.startswith('OMIM:'):
            omim = f'OMIM:{omim}'
        mapping[omim] = str(row[name_col]).strip()
    return mapping


def select_top_nodes(df, top_n):
    strength = df.groupby('Source')['Weight'].sum() + df.groupby('Target')['Weight'].sum()
    top_nodes = set(strength.sort_values(ascending=False).head(top_n).index)
    return df[df['Source'].isin(top_nodes) & df['Target'].isin(top_nodes)]


def build_graph(df):
    return nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=['Weight','Alpha'], create_using=nx.Graph())


def detect_communities(G):
    if G.number_of_nodes() == 0:
        return {}
    comms = list(greedy_modularity_communities(G))
    node2comm = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            node2comm[n] = cid
    return node2comm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=False, default=None,
                        help='Optional OMIM→name mapping TSV/CSV (columns: omim_id,name)')
    parser.add_argument('--output', type=Path, default=Path('data/replication/omim_backbone.html'))
    parser.add_argument('--top-n', type=int, default=300)
    parser.add_argument('--physics', action='store_true')
    args = parser.parse_args()

    df = load_backbone(args.backbone)
    if args.top_n:
        df = select_top_nodes(df, args.top_n)

    name_map = load_mapping(args.mapping) if args.mapping else {}
    G = build_graph(df)

    # Community coloring
    node2comm = detect_communities(G)
    comm_colors = {}
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f",
        "#cab2d6", "#ffff99"
    ]

    net = Network(height='900px', width='100%', bgcolor='#ffffff', font_color='#222222', notebook=False, directed=False)
    net.barnes_hut() if args.physics else net.force_atlas_2based(gravity=-35, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.88, overlap=0)

    # precompute max weight
    max_w = max((data.get('Weight', 1.0) for _, _, data in G.edges(data=True)), default=1.0)

    # Add nodes
    for n in G.nodes():
        label = name_map.get(n, n)
        strength = sum(data.get('Weight', 0.0) for _, _, data in G.edges(n, data=True))
        size = 10 + 6 * (strength / max(1e-9, max_w))
        comm = node2comm.get(n)
        if comm is not None and comm not in comm_colors:
            comm_colors[comm] = palette[comm % len(palette)]
        color = comm_colors.get(comm, '#4e79a7')
        net.add_node(n, label=label, title=f"{label}<br>{n}<br>Strength: {strength:.2f}<br>Community: {comm}", value=strength, size=size, color=color)

    # Add edges
    for u, v, data in G.edges(data=True):
        w = float(data.get('Weight', 1.0))
        net.add_edge(u, v, value=w, title=f"Weight={w:.2f}", width=0.6 + 3.0 * (w / max_w))

    net.write_html(str(args.output))
    print(f"Saved PyVis HTML to {args.output}")


if __name__ == '__main__':
    main()
