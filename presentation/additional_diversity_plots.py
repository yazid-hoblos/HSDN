"""
Generate additional informative plots for disease-gene diversity analysis.
Creates:
1. Correlation heatmap (pairwise)
2. Joint distribution plots
3. Violin/distribution plots
4. Regression plots with confidence intervals
5. Summary statistics tables
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, pearsonr, linregress

OUTPUT_DIR = Path("presentation/plots")
METRICS_FILE = OUTPUT_DIR / "disease_gene_diversity_metrics.tsv"


def load_metrics():
    """Load the computed metrics"""
    df = pd.read_csv(METRICS_FILE, sep="\t")
    return df


def plot_correlation_heatmap(df: pd.DataFrame):
    """Create a correlation matrix heatmap"""
    # Z-score all metrics
    metrics = [
        "disease_diversity", "gene_diversity",
        "disease_diversity_shannon", "gene_diversity_shannon",
        "disease_betweenness", "gene_betweenness"
    ]
    
    z_df = df[metrics].copy()
    for col in z_df.columns:
        z_df[col] = zscore(z_df[col])
    
    # Compute correlation matrix
    corr_matrix = z_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", 
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    
    # Improve labels
    labels = [
        "Disease Div\n(Bridging)",
        "Gene Div\n(Bridging)",
        "Disease Div\n(Shannon)",
        "Gene Div\n(Shannon)",
        "Disease\nBetweenness",
        "Gene\nBetweenness"
    ]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    ax.set_title("Correlation Matrix: Disease and Gene Diversity Metrics", 
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    
    png = OUTPUT_DIR / "diversity_correlation_heatmap.png"
    pdf = OUTPUT_DIR / "diversity_correlation_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_joint_distributions(df: pd.DataFrame):
    """Create joint distribution plots with marginals"""
    metrics_pairs = [
        ("disease_diversity", "gene_diversity", "Bridging Coefficient"),
        ("disease_diversity_shannon", "gene_diversity_shannon", "Shannon Entropy"),
        ("disease_betweenness", "gene_betweenness", "Betweenness Centrality"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (disease_col, gene_col, title) in zip(axes, metrics_pairs):
        x = zscore(df[disease_col])
        y = zscore(df[gene_col])
        
        # Hexbin plot with density
        hb = ax.hexbin(x, y, gridsize=15, cmap='YlOrRd', mincnt=1, edgecolors='gray', linewidths=0.2)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "b-", linewidth=2, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Correlation
        pcc, pval = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {pcc:.2f}\nP < 1e-{int(-np.log10(max(pval, 1e-10)))}", 
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Disease metric (z-score)", fontsize=10)
        ax.set_ylabel("Gene metric (z-score)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Count", fontsize=9)
    
    fig.suptitle("Joint Distributions: Disease vs Gene Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "diversity_joint_distributions.png"
    pdf = OUTPUT_DIR / "diversity_joint_distributions.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_distributions_violin(df: pd.DataFrame):
    """Create violin plots for each metric"""
    metrics = [
        ("disease_diversity", "Disease Diversity\n(Bridging)"),
        ("gene_diversity", "Gene Diversity\n(Bridging)"),
        ("disease_diversity_shannon", "Disease Diversity\n(Shannon)"),
        ("gene_diversity_shannon", "Gene Diversity\n(Shannon)"),
        ("disease_betweenness", "Disease Betweenness"),
        ("gene_betweenness", "Gene Betweenness"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for ax, (col, label) in zip(axes, metrics):
        data = zscore(df[col])
        parts = ax.violinplot([data], positions=[0], widths=0.7, 
                              showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('#8DD3C7')
            pc.set_alpha(0.7)
        
        ax.scatter(np.random.normal(0, 0.04, len(data)), data, alpha=0.3, s=20, color='black')
        
        ax.set_ylabel("z-score", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add stats
        mean_val = data.mean()
        std_val = data.std()
        ax.text(0.98, 0.97, f"μ={mean_val:.2f}\nσ={std_val:.2f}", 
                transform=ax.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    fig.suptitle("Distribution of Metrics (Violin Plots)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "diversity_distributions_violin.png"
    pdf = OUTPUT_DIR / "diversity_distributions_violin.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_regression_with_ci(df: pd.DataFrame):
    """Create regression plots with confidence intervals"""
    from scipy import stats
    
    metric_pairs = [
        ("disease_diversity", "gene_diversity", "#5DADE2", "Bridging Coefficient"),
        ("disease_diversity_shannon", "gene_diversity_shannon", "#F4A460", "Shannon Entropy"),
        ("disease_betweenness", "gene_betweenness", "#58D68D", "Betweenness"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (disease_col, gene_col, color, title) in zip(axes, metric_pairs):
        x = zscore(df[disease_col].values)
        y = zscore(df[gene_col].values)
        
        # Scatter
        ax.scatter(x, y, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', linewidth=2.5, label=f'y = {slope:.2f}x + {intercept:.2f}')
        
        # Confidence interval (95%)
        predict_y = slope * x + intercept
        residuals = y - predict_y
        std_residuals = np.std(residuals)
        ci = 1.96 * std_residuals
        ax.fill_between(line_x, line_y - ci, line_y + ci, alpha=0.2, color='red', label='95% CI')
        
        # Stats
        ax.text(0.05, 0.95, f"r² = {r_value**2:.3f}\nP = {p_value:.2e}\nn = {len(x)}", 
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        
        ax.set_xlabel("Disease metric (z-score)", fontsize=10)
        ax.set_ylabel("Gene metric (z-score)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    fig.suptitle("Linear Regression with 95% Confidence Intervals", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    png = OUTPUT_DIR / "diversity_regression_ci.png"
    pdf = OUTPUT_DIR / "diversity_regression_ci.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def generate_summary_table(df: pd.DataFrame):
    """Generate summary statistics table"""
    metrics = [
        "disease_diversity", "gene_diversity",
        "disease_diversity_shannon", "gene_diversity_shannon",
        "disease_betweenness", "gene_betweenness"
    ]
    
    summary_stats = []
    for col in metrics:
        z_data = zscore(df[col])
        summary_stats.append({
            "Metric": col.replace("_", " ").title(),
            "Mean (z)": f"{z_data.mean():.3f}",
            "Std (z)": f"{z_data.std():.3f}",
            "Min": f"{df[col].min():.2e}",
            "Max": f"{df[col].max():.2e}",
            "Median": f"{df[col].median():.2e}",
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save as TSV
    out_tsv = OUTPUT_DIR / "diversity_summary_statistics.tsv"
    summary_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv}")
    
    # Also create a visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    fig.suptitle("Summary Statistics: Diversity Metrics", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    png = OUTPUT_DIR / "diversity_summary_table.png"
    pdf = OUTPUT_DIR / "diversity_summary_table.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    print("Loading metrics...")
    df = load_metrics()
    print(f"Loaded {len(df)} diseases with complete metrics\n")
    
    print("Generating correlation heatmap...")
    plot_correlation_heatmap(df)
    
    print("\nGenerating joint distributions...")
    plot_joint_distributions(df)
    
    print("\nGenerating violin plots...")
    plot_distributions_violin(df)
    
    print("\nGenerating regression plots with CI...")
    plot_regression_with_ci(df)
    
    print("\nGenerating summary statistics...")
    generate_summary_table(df)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
