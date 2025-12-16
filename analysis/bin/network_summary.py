#!/usr/bin/env python3
"""
Summarize and cross-compare disease networks produced in this project.
Outputs are written to analysis/.
- Node/edge counts and weight stats for symptom projection, SGPDN edges/backbone, and HPO projection/backbone.
- Edge overlap sizes (Jaccard and raw intersections) across layers.
- Weight correlations on shared edges between symptom-based and HPO-based projections.
- Plots: weight histograms and correlation scatter.
"""

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=sep)


def undirected_edges(df: pd.DataFrame, cols: Tuple[str, str]) -> pd.Series:
    return df[list(cols)].apply(lambda r: tuple(sorted((r[0], r[1]))), axis=1)


def weight_stats(weights: pd.Series) -> Dict[str, float]:
    w = weights.dropna().astype(float)
    if w.empty:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "p95": np.nan}
    return {
        "min": w.min(),
        "max": w.max(),
        "mean": w.mean(),
        "median": w.median(),
        "p95": w.quantile(0.95),
    }


def stats_row(name: str, df: pd.DataFrame, cols: Tuple[str, str], weight_col: str) -> Dict[str, object]:
    nodes = set(df[cols[0]]) | set(df[cols[1]])
    weights = df[weight_col]
    row = {
        "layer": name,
        "nodes": len(nodes),
        "edges": len(df),
    }
    row.update(weight_stats(weights))
    return row


def compute_overlaps(edge_sets: Dict[str, set]) -> List[Dict[str, object]]:
    rows = []
    names = list(edge_sets)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            inter = set.intersection(*[edge_sets[c] for c in combo])
            union = set.union(*[edge_sets[c] for c in combo])
            rows.append({
                "layers": "+".join(combo),
                "k": len(combo),
                "intersection_size": len(inter),
                "union_size": len(union),
                "jaccard": len(inter) / len(union) if union else np.nan,
            })
    return rows


def plot_histograms(weight_dict: Dict[str, pd.Series], out_path: Path) -> None:
    fig, axes = plt.subplots(len(weight_dict), 1, figsize=(8, 3 * len(weight_dict)), constrained_layout=True)
    if len(weight_dict) == 1:
        axes = [axes]
    for ax, (name, series) in zip(axes, weight_dict.items()):
        s = series.dropna().astype(float)
        ax.hist(s, bins=50, color="#4c72b0", alpha=0.8)
        ax.set_title(f"Weight distribution: {name}")
        ax.set_xlabel("weight")
        ax.set_ylabel("count")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(df[x_col], df[y_col], s=6, alpha=0.4, color="#dd8452")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Shared-edge weight correlation")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize disease networks and overlaps")
    parser.add_argument("--outdir", type=Path, default=Path("analysis"), help="Directory for outputs")
    parser.add_argument("--sym_proj", type=Path, default=Path("data/replication/disease_symptoms_projection.tsv"))
    parser.add_argument("--sym_edges", type=Path, default=Path("data/replication/consensus_symptom_geneppi_edges.tsv"))
    parser.add_argument("--sym_backbone", type=Path, default=Path("data/replication/consensus_symptom_geneppi_backbone.tsv"))
    parser.add_argument("--hpo_proj", type=Path, default=Path("data/HPO/disease_disease_network.tsv"))
    parser.add_argument("--hpo_backbone", type=Path, default=Path("data/HPO/disease_disease_network_backbone.csv"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sym_proj = load_table(args.sym_proj, sep="\t")
    sym_edges = load_table(args.sym_edges, sep="\t")
    sym_backbone = load_table(args.sym_backbone, sep="\t")
    hpo_proj = load_table(args.hpo_proj, sep="\t")
    hpo_backbone = load_table(args.hpo_backbone, sep="," if args.hpo_backbone.suffix == ".csv" else "\t")

    stats = []
    stats.append(stats_row("symptom_projection", sym_proj, ("disease1", "disease2"), "similarity"))
    stats.append(stats_row("sgpdn_edges", sym_edges, ("Source", "Target"), "weight"))
    stats.append(stats_row("sgpdn_backbone", sym_backbone, ("Source", "Target"), "weight"))
    stats.append(stats_row("hpo_projection", hpo_proj, ("disease1", "disease2"), "weight"))
    stats.append(stats_row("hpo_backbone", hpo_backbone, ("Source", "Target"), "Weight"))
    pd.DataFrame(stats).to_csv(outdir / "network_stats.tsv", sep="\t", index=False)

    edge_sets = {
        "sym_proj": set(undirected_edges(sym_proj, ("disease1", "disease2"))),
        "sym_backbone": set(undirected_edges(sym_backbone, ("Source", "Target"))),
        "hpo_backbone": set(undirected_edges(hpo_backbone, ("Source", "Target"))),
    }
    pd.DataFrame(compute_overlaps(edge_sets)).to_csv(outdir / "edge_overlaps.tsv", sep="\t", index=False)

    sym_edges["key"] = undirected_edges(sym_edges, ("Source", "Target"))
    hpo_proj["key"] = undirected_edges(hpo_proj, ("disease1", "disease2"))
    merged = sym_edges.merge(hpo_proj[["key", "weight"]], on="key", how="inner", suffixes=("_sym", "_hpo"))
    if not merged.empty:
        corr = merged[["weight_sym", "weight_hpo"]].corr(method="spearman")
        corr.to_csv(outdir / "weight_correlations.tsv", sep="\t")
        plot_scatter(merged, "weight_sym", "weight_hpo", outdir / "correlation_scatter.png")
    else:
        pd.DataFrame().to_csv(outdir / "weight_correlations.tsv", sep="\t")

    weight_dict = {
        "sym_proj": sym_proj["similarity"],
        "sgpdn_edges": sym_edges["weight"],
        "sgpdn_backbone": sym_backbone["weight"],
        "hpo_proj": hpo_proj["weight"],
        "hpo_backbone": hpo_backbone["Weight"],
    }
    plot_histograms(weight_dict, outdir / "weight_histograms.png")

    top_edges = sym_backbone.sort_values("weight", ascending=False).head(20)
    top_edges.to_csv(outdir / "top20_sgpdn_backbone.tsv", sep="\t", index=False)

    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
