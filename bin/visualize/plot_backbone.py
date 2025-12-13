"""
Plot consensus backbone network for slides.

Defaults:
- Input: data/replication/consensus_symptom_geneppi_backbone.tsv
- Output: data/replication/consensus_backbone_plot.png
- Keeps top-N nodes by weighted degree (strength) before plotting (default 200).

Usage:
    python bin/plot_backbone.py --top-n 200 --dpi 300
"""

import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def load_backbone(path):
    df = pd.read_csv(path, sep='\t')
    required = {'Source', 'Target', 'weight'}
    if not required.issubset(df.columns):
        raise ValueError(f"Input missing required columns: {required}")
    return df


def select_top_nodes(df, top_n):
    # strength = sum of weights per node
    strength = df.groupby('Source')['weight'].sum() + df.groupby('Target')['weight'].sum()
    top_nodes = set(strength.sort_values(ascending=False).head(top_n).index)
    # return df[df['Source'].isin(top_nodes) & df['Target'].isin(top_nodes)]
    # keep all nodes and edges whether in top-n or not
    return df[df['Source'].isin(top_nodes) | df['Target'].isin(top_nodes)]


def build_graph(df):
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr=['weight'], create_using=nx.Graph())
    return G


def largest_component(G):
    if G.number_of_nodes() == 0:
        return G
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(components[0]).copy()


def scale(values, min_size, max_size):
    if len(values) == 0:
        return []
    v = np.array(values, dtype=float)
    v_min, v_max = v.min(), v.max()
    if v_max == v_min:
        return np.full_like(v, (min_size + max_size) / 2)
    return min_size + (v - v_min) * (max_size - min_size) / (v_max - v_min)


def plot_graph(G, out_path, seed=42, dpi=300):
    plt.figure(figsize=(10, 8), dpi=dpi)

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=None)

    # Node sizes/colors
    strength = {n: sum(d['weight'] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    degrees = dict(G.degree())
    node_sizes = scale(list(strength.values()), 80, 1500)
    node_colors = scale(list(degrees.values()), 0.2, 1.0)

    # Edge widths
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    edge_widths = scale(weights, 0.5, 3.5)

    # Draw
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap='plasma', alpha=0.9)
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.25)

    # Optional labels: top 25 nodes by strength
    top_labels = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:25]
    label_nodes = {n: n for n, _ in top_labels}
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=7)

    plt.axis('off')
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values()) if degrees else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6)
    cbar.set_label('Degree')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/replication/consensus_symptom_geneppi_backbone.tsv'))
    parser.add_argument('--output', type=Path, default=Path('data/replication/consensus_backbone_plot.png'))
    parser.add_argument('--top-n', type=int, default=200, help='Top-N nodes by weighted degree to plot')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--seed', type=int, default=42, help='Layout seed')
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = load_backbone(args.input)
    df_filt = select_top_nodes(df, args.top_n)
    G = build_graph(df_filt)
    G = largest_component(G)

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} (after top-{args.top_n} + LCC)")
    plot_graph(G, args.output, seed=args.seed, dpi=args.dpi)
    print(f"Saved plot to {args.output}")


if __name__ == '__main__':
    main()
