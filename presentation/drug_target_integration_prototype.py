from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

OUTPUT_DIR = Path("presentation/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_TARGETS = Path("data/demo/drug_targets_demo.tsv")
DISEASE_GENES = Path("data/demo/disease_genes_demo.tsv")
BACKBONE = Path("data/replication/filtering/disease_network_backbone.csv")

DISEASES = ["Pulmonary Fibrosis", "Proteinuria", "Metabolic Syndrome X"]


def load_data():
    dt = pd.read_csv(DRUG_TARGETS, sep="\t")
    dg = pd.read_csv(DISEASE_GENES, sep="\t")
    bb = pd.read_csv(BACKBONE)
    return dt, dg, bb


def build_backbone(bb: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in bb.iterrows():
        w = float(row["Weight"]) if "Weight" in row else float(row[2])
        dist = 1.0 / max(w, 1e-6)
        g.add_edge(row["Source"], row["Target"], weight=w, dist=dist)
    return g


def compute_direct_coverage(dt: pd.DataFrame, dg: pd.DataFrame) -> pd.DataFrame:
    dg_map = dg.groupby("disease")["gene_symbol"].apply(set)
    dt_map = dt.groupby("drug")["gene_symbol"].apply(set)
    rows = []
    for disease in DISEASES:
        dgenes = dg_map.get(disease, set())
        for drug, targets in dt_map.items():
            overlap = dgenes.intersection(targets)
            rows.append({
                "disease": disease,
                "drug": drug,
                "direct_overlap": len(overlap),
                "overlap_genes": ",".join(sorted(overlap)) if overlap else "",
            })
    return pd.DataFrame(rows)


def compute_proximity(bb_graph: nx.Graph) -> pd.DataFrame:
    # Use approved_for approximations from repurposing proposals file if present; else demo assumptions
    # For this prototype: drug approved_for anchor diseases
    approved_map = {
        "Metformin": "Diabetes Mellitus",
        "Lisinopril": "Hypertension",
        "Atorvastatin": "Hyperlipidemias",
        "Thalidomide": "Multiple Myeloma",
    }
    rows = []
    for disease in DISEASES:
        for drug, approved_for in approved_map.items():
            if disease in bb_graph and approved_for in bb_graph:
                try:
                    d = nx.shortest_path_length(bb_graph, approved_for, disease, weight="dist")
                except nx.NetworkXNoPath:
                    d = None
            else:
                d = None
            rows.append({"disease": disease, "drug": drug, "proximity": d})
    return pd.DataFrame(rows)


def assemble_scores(direct_df: pd.DataFrame, prox_df: pd.DataFrame) -> pd.DataFrame:
    df = direct_df.merge(prox_df, on=["disease", "drug"], how="left")
    # Proximity to score: smaller distance → higher score; normalize per disease
    def norm_prox(sub: pd.DataFrame):
        if sub["proximity"].dropna().empty:
            sub["prox_score"] = 0.0
            return sub
        m = sub["proximity"].dropna().max()
        sub["prox_score"] = sub["proximity"].apply(lambda x: 1 - (x/m) if pd.notna(x) and m > 0 else 0.0)
        return sub
    df = df.groupby("disease", group_keys=False).apply(norm_prox)
    # Composite: 0.6 direct coverage + 0.4 proximity
    df["composite_score"] = 0.6 * (df["direct_overlap"].clip(0, None)) + 0.4 * df["prox_score"].fillna(0)
    return df


def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(len(DISEASES), 1, figsize=(8, 9), sharex=True)
    if len(DISEASES) == 1:
        axes = [axes]
    for ax, disease in zip(axes, DISEASES):
        sub = df[df["disease"] == disease].sort_values("composite_score", ascending=False)
        ax.barh(sub["drug"], sub["composite_score"], color="#7FB3D5", edgecolor="black")
        for y, (score, ovl, genes, prox) in enumerate(zip(sub["composite_score"], sub["direct_overlap"], sub["overlap_genes"], sub["proximity"])):
            txt = f"{score:.2f} (direct={ovl}, prox={'NA' if pd.isna(prox) else prox:.2f})"
            if genes:
                txt += f" | {genes}"
            ax.text(score + 0.02, y, txt, va="center", fontsize=8)
        ax.set_title(disease, loc="left", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, max(2, sub["composite_score"].max() + 0.5))
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Composite (0.6×direct overlap + 0.4×proximity)")
    fig.suptitle("Prototype: Protein–drug integration over SGPDN", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = OUTPUT_DIR / "drug_target_integration_prototype.png"
    pdf = OUTPUT_DIR / "drug_target_integration_prototype.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("Saved:", png)
    print("Saved:", pdf)

    out = OUTPUT_DIR / "drug_target_integration_prototype.tsv"
    df.sort_values(["disease", "composite_score"], ascending=[True, False]).to_csv(out, sep="\t", index=False)
    print("Saved:", out)


if __name__ == "__main__":
    dt, dg, bb = load_data()
    g = build_backbone(bb)
    direct = compute_direct_coverage(dt, dg)
    prox = compute_proximity(g)
    scores = assemble_scores(direct, prox)
    plot_results(scores)
