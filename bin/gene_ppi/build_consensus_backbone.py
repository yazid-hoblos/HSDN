"""
Build consensus symptom+{gene|ppi} network and apply backbone filtering.

Steps:
1) Parse unified_disease_network.tsv in chunks to collect sources per disease pair.
   - Keep pairs where sources include 'symptom' AND ( 'gene' OR 'ppi' ).
2) Collect those pairs (by disease names) and fetch their symptom similarity from
   disease_symptoms_projection.tsv.
3) Build weighted edge list (weight = symptom similarity) using mesh IDs for stability.
4) Apply disparity filter (backbone) with alpha (default 0.05).
5) Export edge list and backbone to data/replication/consensus_symptom_geneppi_edges.tsv
   and ..._backbone.tsv.

Assumptions:
- unified_disease_network.tsv columns: disease1_mesh_id, disease1_name, disease2_mesh_id,
  disease2_name, network_source, similarity_score, shared_genes, shared_proteins, networks
- disease_symptoms_projection.tsv columns: disease1, disease2, similarity (MeSH disease names)
"""

import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict


def disparity_filter(G, alpha=0.05):
    backbone = nx.Graph()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) <= 1:
            for nbr in neighbors:
                if not backbone.has_edge(node, nbr):
                    backbone.add_edge(node, nbr, weight=G[node][nbr]['weight'])
            continue
        k = len(neighbors)
        total_w = sum(G[node][nbr]['weight'] for nbr in neighbors)
        for nbr in neighbors:
            w = G[node][nbr]['weight']
            p = w / total_w if total_w else 0
            alpha_ij = (1 - p) ** (k - 1)
            if alpha_ij < alpha:
                if not backbone.has_edge(node, nbr):
                    backbone.add_edge(node, nbr, weight=w, alpha=alpha_ij)
    return backbone


def collect_sources(unified_path, chunk_size=200000):
    pair_sources = defaultdict(set)
    pair_meta = {}
    use_cols = [
        'disease1_mesh_id', 'disease1_name', 'disease2_mesh_id', 'disease2_name',
        'network_source'
    ]
    for chunk in pd.read_csv(unified_path, sep='\t', usecols=use_cols, chunksize=chunk_size, dtype=str):
        for _, row in chunk.iterrows():
            d1_id = row['disease1_mesh_id']
            d2_id = row['disease2_mesh_id']
            key = tuple(sorted([d1_id, d2_id]))
            pair_sources[key].add(row['network_source'])
            if key not in pair_meta:
                # store names for later lookup
                pair_meta[key] = (
                    row['disease1_mesh_id'], row['disease1_name'],
                    row['disease2_mesh_id'], row['disease2_name']
                )
    return pair_sources, pair_meta


def filter_pairs(pair_sources):
    keep = set()
    for key, sources in pair_sources.items():
        if 'symptom' in sources and ('gene' in sources or 'ppi' in sources):
            keep.add(key)
    return keep


def load_symptom_similarity(sim_path, needed_pairs, chunk_size=200000):
    # Map by disease names (MeSH disease names)
    sim_lookup = {}
    cols = ['disease1', 'disease2', 'similarity']
    for chunk in pd.read_csv(sim_path, sep='\t', usecols=cols, chunksize=chunk_size, dtype={'disease1':'string','disease2':'string','similarity':'float64'}):
        for _, row in chunk.iterrows():
            k = tuple(sorted([row['disease1'], row['disease2']]))
            # Only keep if we need it
            if k in needed_pairs:
                sim_lookup[k] = row['similarity']
    return sim_lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.05, help='Backbone alpha (default 0.05)')
    parser.add_argument('--chunk', type=int, default=200000, help='Chunk size for CSV reading')
    args = parser.parse_args()

    base = Path(__file__).parent.parent.parent
    data_dir = base / 'data' / 'replication'
    unified_path = data_dir / 'unified_disease_network.tsv'
    sim_path = data_dir / 'disease_symptoms_projection.tsv'

    out_edges = 'consensus_symptom_geneppi_edges.tsv'
    out_backbone = 'consensus_symptom_geneppi_backbone.tsv'

    if not unified_path.exists():
        print(f"Missing unified network: {unified_path}")
        return
    if not sim_path.exists():
        print(f"Missing symptom similarity: {sim_path}")
        return

    print("Collecting sources from unified network...")
    pair_sources, pair_meta = collect_sources(unified_path, chunk_size=args.chunk)

    print("Filtering pairs with symptom + (gene or ppi)...")
    keep_pairs = filter_pairs(pair_sources)
    print(f"Pairs kept: {len(keep_pairs):,}")

    # Build needed pairs by disease names (for similarity lookup)
    needed_name_pairs = set()
    for key in keep_pairs:
        d1_id, d1_name, d2_id, d2_name = pair_meta[key]
        needed_name_pairs.add(tuple(sorted([d1_name, d2_name])))

    print("Loading symptom similarities for kept pairs...")
    sim_lookup = load_symptom_similarity(sim_path, needed_name_pairs, chunk_size=args.chunk)
    print(f"Similarities found: {len(sim_lookup):,}")

    # Build edge list with similarity weight
    rows = []
    missing_sim = 0
    for key in keep_pairs:
        d1_id, d1_name, d2_id, d2_name = pair_meta[key]
        name_key = tuple(sorted([d1_name, d2_name]))
        sim = sim_lookup.get(name_key)
        if sim is None:
            missing_sim += 1
            continue
        rows.append({
            'Source': d1_id,
            'Target': d2_id,
            'source_name': d1_name,
            'target_name': d2_name,
            'weight': sim
        })

    print(f"Edges with similarity: {len(rows):,}; missing similarity: {missing_sim:,}")
    df_edges = pd.DataFrame(rows)
    df_edges.to_csv(out_edges, sep='\t', index=False)
    print(f"Saved filtered edges to {out_edges}")

    # Build graph and backbone
    print("Building graph and applying disparity filter...")
    G = nx.from_pandas_edgelist(df_edges, source='Source', target='Target', edge_attr=['weight'], create_using=nx.Graph())
    backbone = disparity_filter(G, alpha=args.alpha)

    # Export backbone
    backbone_rows = []
    for u, v, data in backbone.edges(data=True):
        backbone_rows.append({
            'Source': u,
            'Target': v,
            'weight': data.get('weight', np.nan),
            'alpha': data.get('alpha', np.nan)
        })
    pd.DataFrame(backbone_rows).to_csv(out_backbone, sep='\t', index=False)
    print(f"Backbone edges: {len(backbone_rows):,} -> {out_backbone}")


if __name__ == '__main__':
    main()
