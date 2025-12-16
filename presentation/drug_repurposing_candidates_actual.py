from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

DATA_PATH = Path("analysis/repurposing_drug_proposals.tsv")
BACKBONE_PATH = Path("data/replication/filtering/disease_network_backbone.csv")
OUTPUT_DIR = Path("presentation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Diseases from the schematic that have real drug entries
DISEASES = [
    "Pulmonary Fibrosis",
    "Proteinuria",
    "Metabolic Syndrome X",
]


def load_candidates():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df.rename(columns={
        "repurposing_candidate": "disease",
        "known_disease": "approved_for",
    })
    df = df[df["disease"].isin(DISEASES)].copy()
    return df


def load_backbone() -> nx.Graph:
    # Build SGPDN backbone graph with weight as inverse distance
    backbone = pd.read_csv(BACKBONE_PATH)
    g = nx.Graph()
    for _, row in backbone.iterrows():
        w = float(row["Weight"])
        dist = 1.0 / max(w, 1e-6)
        g.add_edge(row["Source"], row["Target"], weight=w, dist=dist)
    return g


def score_candidates(df: pd.DataFrame, g: nx.Graph) -> pd.DataFrame:
    evidence_weight = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    df["evidence_w"] = df["evidence_strength"].map(evidence_weight).fillna(0.3)

    def proximity(row):
        src = row["approved_for"]
        dst = row["disease"]
        if src not in g or dst not in g:
            return None
        try:
            # Shortest path using inverse weight as distance
            return nx.shortest_path_length(g, src, dst, weight="dist")
        except nx.NetworkXNoPath:
            return None

    df["proximity"] = df.apply(proximity, axis=1)

    # Normalize components
    sim = df["similarity"].clip(0, 1)
    tgt = df["target_overlap"].fillna(0)
    ev = df["evidence_w"]
    prox = df["proximity"]
    # Convert proximity to a decaying score (smaller distance -> higher score)
    prox_score = prox.copy()
    max_prox = prox.dropna().max() if prox.dropna().size else None
    if max_prox and max_prox > 0:
        prox_score = prox_score.apply(lambda x: 1 - (x / max_prox) if pd.notna(x) else 0)
    else:
        prox_score = pd.Series(0, index=df.index)

    df["composite_score"] = (
        0.45 * sim
        + 0.25 * tgt
        + 0.20 * prox_score
        + 0.10 * ev
    )
    return df


def plot_top_candidates(df: pd.DataFrame, top_n: int = 5):
    fig, axes = plt.subplots(len(DISEASES), 1, figsize=(8, 10), sharex=True)
    if len(DISEASES) == 1:
        axes = [axes]

    palette = {"High": "#27AE60", "Medium": "#F39C12", "Low": "#C0392B"}

    for ax, disease in zip(axes, DISEASES):
        sub = df[df["disease"] == disease].sort_values("composite_score", ascending=False).head(top_n)
        if sub.empty:
            ax.text(0.5, 0.5, "No candidates found", ha="center", va="center")
            ax.set_yticks([])
            ax.set_xlim(0, 1.1)
            ax.set_title(disease, loc="left", fontsize=11, fontweight="bold")
            continue
        ax.barh(
            y=sub["drug"],
            width=sub["composite_score"],
            color=sub["evidence_strength"].map(palette),
            edgecolor="black",
            alpha=0.85,
        )
        for y, (score, sim, ev, prox) in enumerate(zip(sub["composite_score"], sub["similarity"], sub["evidence_strength"], sub["proximity"])):
            prox_txt = f", prox={prox:.2f}" if pd.notna(prox) else ""
            ax.text(score + 0.01, y, f"{score:.2f} (sim={sim:.2f}{prox_txt}, {ev})", va="center", fontsize=8)
        ax.set_title(disease, loc="left", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Composite score (sim + target overlap + proximity + evidence)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    labels = [f"Evidence: {k}" for k in palette.keys()]
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Top repurposing candidates (real drugs, network-aware)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png = OUTPUT_DIR / "drug_repurposing_candidates_actual.png"
    pdf = OUTPUT_DIR / "drug_repurposing_candidates_actual.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved:", png)
    print("Saved:", pdf)


def save_table(df: pd.DataFrame, top_n: int = 8):
    rows = []
    for disease in DISEASES:
        sub = df[df["disease"] == disease].sort_values("composite_score", ascending=False).head(top_n)
        rows.append(sub)
    table = pd.concat(rows, axis=0)
    out = OUTPUT_DIR / "drug_repurposing_candidates_actual.tsv"
    table.to_csv(out, sep="\t", index=False)
    print("Saved:", out)


if __name__ == "__main__":
    df = load_candidates()
    g = load_backbone()
    df = score_candidates(df, g)
    plot_top_candidates(df, top_n=5)
    save_table(df, top_n=8)
