"""
Generate publication volume imbalance plot for presentation intro.
Shows the gap between molecular disease research and symptom-level integration.
"""

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez
import time

# Set your email for PubMed queries (required by NCBI)
Entrez.email = "yazidhoblos4@gmail.com"

def get_pubmed_count(query):
    """Query PubMed and return publication count."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
        record = Entrez.read(handle)
        handle.close()
        count = int(record["Count"])
        print(f"Query: {query[:60]:<60} | Count: {count:,}")
        time.sleep(0.4)  # Rate limiting - be nice to NCBI
        return count
    except Exception as e:
        print(f"Error querying '{query}': {e}")
        return 0

def main():
    print("Querying PubMed for publication counts...\n")
    
    # Define search queries
    # queries = {
    #     "Disease + Genes": "disease AND (gene OR genomics OR genetic)",
    #     "Disease + Protein Interactions": "disease AND (protein interaction OR PPI)",
    #     "Disease + Pathways": "disease AND (pathway OR signaling)",
    #     "Disease + Symptoms": "disease AND symptom",
    #     "Disease + Symptoms + Molecular": "disease AND symptom AND (gene OR protein OR molecular)"
    # }
    queries = {
    "Disease + Genes":
        "disease AND (gene OR genomics OR genetic)",

    "Disease + Protein Interactions":
        "disease AND (\"protein interaction\" OR \"protein-protein interaction\" OR PPI)",

    "Disease + Pathways":
        "disease AND (pathway OR signaling)",

    "Disease Networks":
        "\"disease network\" OR \"disease-disease network\"",

    "Disease–Symptom Relations":
        "\"disease symptom\" OR \"disease–symptom\" OR \"disease-symptom association\"",

    "Symptom-based Disease Networks":
        "\"symptom-based\" AND disease AND network"
    }

    queries = {
        # Post-genomic era: abundance of molecular data
        "Molecular-only Disease Studies":
            "disease AND (gene OR genomics OR genetic OR \"protein interaction\" OR pathway) "
            "NOT (symptom OR phenotype)",

        # Clinical-level focus without molecular mechanisms
        "Clinical-only Disease Studies":
            "disease AND (symptom OR phenotype OR \"clinical manifestation\") "
            "NOT (gene OR protein OR molecular)",

        # Explicit molecular–clinical integration (the gap)
        "Molecular–Clinical Integration":
            "disease AND (symptom OR phenotype) AND "
            "(gene OR protein OR molecular) AND "
            "(network OR interaction OR pathway)"
    }
    
    # Get counts
    counts = {}
    for label, query in queries.items():
        counts[label] = get_pubmed_count(query)
    
    print(f"\n{'='*80}")
    print("Publication Counts Summary:")
    print(f"{'='*80}")
    for label, count in counts.items():
        print(f"{label:<40} {count:>12,}")
    print(f"{'='*80}\n")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = list(counts.keys())
    values = list(counts.values())
    
    # Create bar chart with log scale
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_yscale('log')
    ax.set_ylabel('Number of Publications (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Research Focus Area', fontsize=14, fontweight='bold')
    ax.set_title('Molecular Disease Research Vastly Outweighs Symptom-Level Integration',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels with rotation
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11)
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Add annotation to highlight the gap
    # if len(values) >= 5:
    #     ratio = values[0] / values[4] if values[4] > 0 else 0
    #     ax.annotate(f'{ratio:.0f}× more publications',
    #                xy=(4, values[4]), xytext=(2.5, values[0]/10),
    #                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
    #                fontsize=12, color='red', fontweight='bold',
    #                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'plots/slide1_publication_imbalance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for presentation
    output_pdf = 'plots/slide1_publication_imbalance.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")
    
    plt.show()
    
    # Save counts to file for reference
    with open('data/publication_counts.txt', 'w') as f:
        f.write("PubMed Publication Counts\n")
        f.write("=" * 80 + "\n\n")
        for label, count in counts.items():
            f.write(f"{label:<40} {count:>12,}\n")
        f.write("\n" + "=" * 80 + "\n")
        if len(values) >= 5 and values[4] > 0:
            f.write(f"\nRatio (Disease+Genes : Disease+Symptoms+Molecular): {values[0]/values[4]:.1f}:1\n")
    
    print("Counts saved to: publication_counts.txt")

if __name__ == "__main__":
    main()
