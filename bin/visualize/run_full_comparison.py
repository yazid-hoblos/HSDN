"""
Master script to run full backbone and community comparison analysis.

Executes in order:
1. Generate PyVis plots and community assignments for both networks
2. Compare backbone network structures
3. Compare community structures

Outputs all results to presentation/plots/
"""

import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: Command failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✓ SUCCESS")
        return True


def main():
    print("="*70)
    print("BACKBONE & COMMUNITY COMPARISON PIPELINE")
    print("="*70)
    
    # Ensure output directory exists
    Path("presentation/plots").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate PyVis plot for consensus genePPI backbone
    success = run_command(
        ["python", "bin/visualize/plot_backbone_pyvis.py",
         "--backbone", "data/replication/consensus_symptom_geneppi_backbone.tsv",
         "--edges", "data/replication/consensus_symptom_geneppi_edges.tsv",
         "--output", "presentation/plots/consensus_geneppi_network.html",
         "--communities", "presentation/plots/consensus_geneppi_communities.tsv"],
        "STEP 1: Generate Consensus GenePPI Network Visualization"
    )
    if not success:
        print("\n⚠ Warning: Could not generate consensus network plot")
    
    # Step 2: Generate PyVis plot for original backbone
    success = run_command(
        ["python", "bin/visualize/plot_backbone_pyvis.py",
         "--backbone", "data/replication/filtering/disease_network_backbone.csv",
         "--output", "presentation/plots/original_backbone_network.html",
         "--communities", "presentation/plots/original_backbone_communities.tsv"],
        "STEP 2: Generate Original Backbone Network Visualization"
    )
    if not success:
        print("\n⚠ Warning: Could not generate original network plot")
    
    # Step 3: Compare backbones
    success = run_command(
        ["python", "bin/visualize/compare_backbones.py"],
        "STEP 3: Compare Backbone Network Structures"
    )
    if not success:
        print("\n⚠ Warning: Could not complete backbone comparison")
    
    # Step 4: Compare communities
    success = run_command(
        ["python", "bin/visualize/compare_communities.py"],
        "STEP 4: Compare Community Structures"
    )
    if not success:
        print("\n⚠ Warning: Could not complete community comparison")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nAll outputs saved to presentation/plots/")
    print("\nGenerated files:")
    print("  - consensus_geneppi_network.html (interactive)")
    print("  - original_backbone_network.html (interactive)")
    print("  - consensus_geneppi_communities.tsv")
    print("  - original_backbone_communities.tsv")
    print("  - backbone_comparison_*.png/pdf")
    print("  - community_comparison_*.png/pdf")
    print("  - *_stats.tsv (various statistics files)")
    print("\nOpen HTML files in browser to explore networks interactively!")


if __name__ == "__main__":
    main()
