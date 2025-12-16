# Backbone and Community Comparison Analysis

## Overview
This analysis compares two disease network backbones:
1. **Original Backbone**: From paper replication (symptom-disease co-occurrence)
2. **Consensus GenePPI Backbone**: From your integrated symptom-gene-PPI network

## Generated Outputs

### 1. Interactive Network Visualizations
- `consensus_geneppi_network.html` - Interactive PyVis plot (618 nodes, 44 communities)
- `original_backbone_network.html` - Interactive PyVis plot (1,035 nodes, 87 communities)
  * Open in browser to explore
  * Color-coded by community
  * Hover for node details
  * Drag to rearrange

### 2. Community Assignments (TSV)
- `consensus_geneppi_communities.tsv` - All nodes with community IDs, degrees, strength
- `original_backbone_communities.tsv` - All nodes with community IDs, degrees, strength
  * Columns: node_id, label, community, degree, strength
  * Sorted by community and strength
  * Ready for further analysis

### 3. Backbone Network Comparison

#### Files in `presentation/plots/backbone/`:
- **Network Statistics**:
  * `backbone_comparison_network_stats.tsv` - Comprehensive network metrics table
  * `backbone_comparison_summary_tables.png/pdf` - Visual summary of statistics
  
- **Edge Analysis**:
  * `backbone_comparison_overlap_stats.tsv` - Edge overlap metrics
  * `backbone_comparison_overlap.png/pdf` - Overlap summary visualization
  * Note: **Zero edge overlap** (different node ID systems)
  
- **Degree Distribution**:
  * `backbone_comparison_degree.png/pdf` - Histogram and log-log degree comparison

#### Key Findings:
- **Original**: 1,035 nodes, 2,199 edges, avg degree 4.25, density 0.0041
- **Consensus**: 618 nodes, 1,722 edges, avg degree 5.57, density 0.0090
- **Edge Overlap**: 0 (networks use different node identifiers)
- **Avg Clustering**: Original (0.389) vs Consensus (0.231)

### 4. Community Structure Comparison

#### Files in `presentation/plots/`:
- **Community Sizes**:
  * `community_comparison_sizes.png/pdf` - Bar chart and histogram of community sizes
  
- **Overlap Analysis**:
  * `community_comparison_overlap_heatmap.png/pdf` - Shows node overlap (none in this case)
  * Explains why: different node ID systems (disease names vs MeSH IDs)
  
- **Top Nodes**:
  * `community_comparison_top_nodes.png/pdf` - Top 5 nodes per community for first 10 communities
  * Shows key disease hubs per community
  
- **Statistics**:
  * `community_comparison_detailed_stats.tsv` - Per-community statistics (size, avg degree, etc.)
  * `community_comparison_node_overlap.tsv` - Node overlap metrics
  * `community_comparison_summary.png/pdf` - Visual summary table

#### Key Findings:
- **Original**: 87 communities, avg size 11.9 nodes, largest 109 nodes
- **Consensus**: 44 communities, avg size 14.0 nodes, largest 72 nodes
- **Node Overlap**: 0 (different ID systems prevent direct mapping)
- **Structural**: Consensus has fewer but larger communities on average

## Important Note on Node Overlap

The two networks use **different node identifier systems**:
- **Original backbone**: Disease names (e.g., "Aneurysm", "Behcet Syndrome")
- **Consensus backbone**: MeSH IDs (e.g., "D007003", "D006501")

This prevents direct node-to-node and community-to-community mapping. However, the **structural comparisons** (degree distributions, community size distributions, network statistics) remain valid and informative.

## Presentation-Ready Plots

All plots are saved in both PNG (300 DPI) and PDF formats for presentation use:

### For Slides - Network Structure:
1. `backbone/backbone_comparison_summary_tables.png` - Side-by-side network statistics
2. `backbone/backbone_comparison_degree.png` - Degree distribution comparison
3. `community_comparison_sizes.png` - Community size distribution

### For Slides - Communities:
1. `community_comparison_summary.png` - Community detection summary table
2. `community_comparison_top_nodes.png` - Key nodes per community
3. Interactive HTMLs can be screenshot for visual appeal

### For Interactive Exploration:
1. `consensus_geneppi_network.html` - Zoom, pan, explore your network
2. `original_backbone_network.html` - Zoom, pan, explore original

## Usage

### Regenerate All Analyses:
```bash
python bin/visualize/run_full_comparison.py
```

### Generate Individual Components:
```bash
# Plot consensus network only
python bin/visualize/plot_backbone_pyvis.py \
  --backbone data/replication/consensus_symptom_geneppi_backbone.tsv \
  --edges data/replication/consensus_symptom_geneppi_edges.tsv \
  --output presentation/plots/consensus_geneppi_network.html \
  --communities presentation/plots/consensus_geneppi_communities.tsv

# Compare backbones only
python bin/visualize/compare_backbones.py

# Compare communities only
python bin/visualize/compare_communities.py
```

## Next Steps

1. **Open HTML files** in browser to explore communities visually
2. **Review community_*.tsv files** to identify key disease hubs per community
3. **Use comparison plots** in presentation to show methodological differences
4. **Consider**: Map MeSH IDs to disease names for direct node overlap comparison (requires mapping file)
