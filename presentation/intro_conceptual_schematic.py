"""
Generate conceptual schematic for presentation intro slide.
Shows: Existing networks → Missing bridge (symptoms) → Research question
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, ConnectionPatch
import numpy as np

def draw_simple_network(ax, positions, edges, node_color='lightblue', edge_color='gray', 
                        node_size=150, title="", title_size=10, label_nodes=False):
    """Draw a simple network schematic."""
    # Draw edges
    for edge in edges:
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        ax.plot(x, y, color=edge_color, linewidth=2, alpha=0.6, zorder=1)
    
    # Draw nodes
    for i, pos in enumerate(positions):
        circle = Circle(pos, node_size, color=node_color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        if label_nodes:
            ax.text(pos[0], pos[1], f'D{i+1}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', zorder=3)
    
    # Add title
    if title:
        ax.text(0.5, -0.15, title, transform=ax.transAxes, 
               ha='center', fontsize=title_size, fontweight='bold')

def draw_bipartite_network(ax, disease_pos, symptom_pos, edges, title=""):
    """Draw bipartite disease-symptom network."""
    # Draw edges
    for edge in edges:
        d_pos = disease_pos[edge[0]]
        s_pos = symptom_pos[edge[1]]
        ax.plot([d_pos[0], s_pos[0]], [d_pos[1], s_pos[1]], 
               color='gray', linewidth=1.5, alpha=0.4, zorder=1)
    
    # Draw disease nodes (circles)
    for pos in disease_pos:
        circle = Circle(pos, 120, color='#E8C4C4', ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
    
    # Draw symptom nodes (squares)
    for pos in symptom_pos:
        square = mpatches.Rectangle((pos[0]-100, pos[1]-100), 200, 200,
                                    color='#C4D7E8', ec='black', linewidth=2, zorder=2)
        ax.add_patch(square)
    
    # Add labels
    ax.text(disease_pos[1][0], disease_pos[0][1] + 400, 'Diseases', 
           ha='center', fontsize=9, style='italic')
    ax.text(symptom_pos[1][0], symptom_pos[0][1] + 400, 'Symptoms',
           ha='center', fontsize=9, style='italic')
    
    if title:
        ax.text(0.5, -0.15, title, transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold')

def draw_overlay_network(ax, positions, edges, overlay_edges, title=""):
    """Draw network with primary and overlay edges."""
    # Draw overlay edges (molecular) - dashed
    for edge in overlay_edges:
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        ax.plot(x, y, color='#E74C3C', linewidth=2.5, alpha=0.7, 
               linestyle='--', zorder=1, label='Molecular')
    
    # Draw primary edges (symptom-based) - solid
    for edge in edges:
        x = [positions[edge[0]][0], positions[edge[1]][0]]
        y = [positions[edge[0]][1], positions[edge[1]][1]]
        ax.plot(x, y, color='#3498DB', linewidth=3, alpha=0.8, 
               zorder=2, label='Symptom-based')
    
    # Draw nodes
    for pos in positions:
        circle = Circle(pos, 150, color='#E8C4C4', ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
    
    if title:
        ax.text(0.5, -0.15, title, transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold')

def main():
    # Create figure with three panels
    fig = plt.figure(figsize=(18, 6))
    
    # Define positions for networks
    # Simple circular layouts for small networks
    def circular_layout(n, radius=600, center=(0, 0)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) 
                for a in angles]
    
    # ========== LEFT PANEL: Existing Approaches ==========
    ax_left = plt.subplot(1, 3, 1)
    ax_left.set_xlim(-2500, 2500)
    ax_left.set_ylim(-3000, 3000)
    ax_left.axis('off')
    ax_left.set_aspect('equal')
    
    # Network 1: Gene-based (top)
    pos1 = circular_layout(5, radius=400, center=(0, 1800))
    edges1 = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,4)]
    
    for i, edge in enumerate(edges1):
        x = [pos1[edge[0]][0], pos1[edge[1]][0]]
        y = [pos1[edge[0]][1], pos1[edge[1]][1]]
        ax_left.plot(x, y, color='#7FB3D5', linewidth=2, alpha=0.6, zorder=1)
    
    for pos in pos1:
        circle = Circle(pos, 120, color='#AED6F1', ec='black', linewidth=1.5, zorder=2)
        ax_left.add_patch(circle)
    
    ax_left.text(0, 2500, 'Gene-based', ha='center', fontsize=9, fontweight='bold')
    ax_left.text(0, 2300, 'disease network', ha='center', fontsize=8, style='italic')
    
    # Network 2: PPI-based (middle)
    pos2 = circular_layout(5, radius=400, center=(0, 0))
    edges2 = [(0,1), (1,2), (2,3), (0,3), (1,3), (2,4)]
    
    for edge in edges2:
        x = [pos2[edge[0]][0], pos2[edge[1]][0]]
        y = [pos2[edge[0]][1], pos2[edge[1]][1]]
        ax_left.plot(x, y, color='#C39BD3', linewidth=2, alpha=0.6, zorder=1)
    
    for pos in pos2:
        circle = Circle(pos, 120, color='#D7BDE2', ec='black', linewidth=1.5, zorder=2)
        ax_left.add_patch(circle)
    
    ax_left.text(0, 700, 'PPI-based', ha='center', fontsize=9, fontweight='bold')
    ax_left.text(0, 500, 'disease network', ha='center', fontsize=8, style='italic')
    
    # Network 3: Comorbidity (bottom)
    pos3 = circular_layout(5, radius=400, center=(0, -1800))
    edges3 = [(0,2), (1,3), (2,4), (0,4), (1,2)]
    
    for edge in edges3:
        x = [pos3[edge[0]][0], pos3[edge[1]][0]]
        y = [pos3[edge[0]][1], pos3[edge[1]][1]]
        ax_left.plot(x, y, color='#F8B4B4', linewidth=2, alpha=0.6, zorder=1)
    
    for pos in pos3:
        circle = Circle(pos, 120, color='#FADBD8', ec='black', linewidth=1.5, zorder=2)
        ax_left.add_patch(circle)
    
    ax_left.text(0, -1100, 'Comorbidity', ha='center', fontsize=9, fontweight='bold')
    ax_left.text(0, -1300, 'network', ha='center', fontsize=8, style='italic')
    
    # Panel title
    ax_left.text(0, -2700, 'Existing disease networks', ha='center', 
                fontsize=13, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    # ========== MIDDLE PANEL: Missing Bridge ==========
    ax_mid = plt.subplot(1, 3, 2)
    ax_mid.set_xlim(-2500, 2500)
    ax_mid.set_ylim(-3000, 3000)
    ax_mid.axis('off')
    ax_mid.set_aspect('equal')
    
    # Bipartite: diseases on left, symptoms on right
    disease_pos = [(-800, 1500), (-800, 500), (-800, -500), (-800, -1500)]
    symptom_pos = [(800, 1500), (800, 500), (800, -500), (800, -1500)]
    
    # Only vertical connections, no disease-disease edges
    connections = [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3), (3,3)]
    
    for conn in connections:
        d_pos = disease_pos[conn[0]]
        s_pos = symptom_pos[conn[1]]
        ax_mid.plot([d_pos[0], s_pos[0]], [d_pos[1], s_pos[1]],
                   color='gray', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Draw diseases (circles, not connected)
    for pos in disease_pos:
        circle = Circle(pos, 150, color='#E8C4C4', ec='black', linewidth=2, zorder=2)
        ax_mid.add_patch(circle)
    
    # Draw symptoms (squares)
    for pos in symptom_pos:
        square = mpatches.Rectangle((pos[0]-120, pos[1]-120), 240, 240,
                                    color='#FEF5E7', ec='black', linewidth=2, zorder=2)
        ax_mid.add_patch(square)
    
    # Labels
    ax_mid.text(-800, 2100, 'Diseases', ha='center', fontsize=10, style='italic')
    ax_mid.text(800, 2100, 'Symptoms', ha='center', fontsize=10, style='italic')
    
    # Highlight the gap - big X or "missing" indicator
    # ax_mid.text(0, -2200, 'No disease–disease', ha='center', fontsize=11, 
    #            color='#C0392B', fontweight='bold')
    # ax_mid.text(0, -2450, 'edges from symptoms', ha='center', fontsize=11,
    #            color='#C0392B', fontweight='bold')
    
    # Panel title
    ax_mid.text(0, -2700, 'Symptoms', ha='center',
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', alpha=0.5))
    
    # ========== RIGHT PANEL: Research Question ==========
    ax_right = plt.subplot(1, 3, 3)
    ax_right.set_xlim(-2500, 2500)
    ax_right.set_ylim(-3000, 3000)
    ax_right.axis('off')
    ax_right.set_aspect('equal')
    
    # Disease network with symptom-based and molecular overlay
    pos_final = circular_layout(6, radius=700, center=(0, 200))
    
    # Symptom-based edges (solid blue)
    symptom_edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (1,4)]
    for edge in symptom_edges:
        x = [pos_final[edge[0]][0], pos_final[edge[1]][0]]
        y = [pos_final[edge[0]][1], pos_final[edge[1]][1]]
        ax_right.plot(x, y, color='#3498DB', linewidth=3.5, alpha=0.8, zorder=1)
    
    # Molecular overlay edges (dashed red)
    molecular_edges = [(0,2), (1,5), (2,4)]
    for edge in molecular_edges:
        x = [pos_final[edge[0]][0], pos_final[edge[1]][0]]
        y = [pos_final[edge[0]][1], pos_final[edge[1]][1]]
        ax_right.plot(x, y, color='#E74C3C', linewidth=3, alpha=0.7,
                     linestyle='--', zorder=2)
    
    # Draw nodes
    for pos in pos_final:
        circle = Circle(pos, 150, color='#A9DFBF', ec='black', linewidth=2, zorder=3)
        ax_right.add_patch(circle)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498DB', linewidth=3, label='Symptom similarity'),
        Line2D([0], [0], color='#E74C3C', linewidth=3, linestyle='--', 
               label='Shared genes/PPIs')
    ]
    ax_right.legend(handles=legend_elements, loc='upper right', fontsize=9,
                   frameon=True, fancybox=True, shadow=True)
    
    # Question text
    ax_right.text(0, -1400, 'Do clinically similar', ha='center', fontsize=12,
                 fontweight='bold', color='#117A65')
    ax_right.text(0, -1650, 'diseases share', ha='center', fontsize=12,
                 fontweight='bold', color='#117A65')
    ax_right.text(0, -1900, 'molecular interactions?', ha='center', fontsize=12,
                 fontweight='bold', color='#117A65')
    
    # Panel title
    ax_right.text(0, -2700, 'This work: symptom-based network', ha='center',
                 fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F4E6', alpha=0.7))
    
    # Add arrows between panels to show progression
    # Arrow 1: Left to Middle
    arrow1 = FancyArrowPatch((2200, 0), (-2200, 0),
                            transform=ax_left.transData,
                            connectionstyle="arc3", 
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='gray', alpha=0.5, zorder=0)
    # Adjust for subplot positioning
    
    # Main title
    fig.suptitle('From Isolated Networks to Integrated Understanding',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save outputs
    output_png = 'plots/slide2_conceptual_schematic.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG saved to: {output_png}")
    
    output_pdf = 'plots/slide2_conceptual_schematic.pdf'
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {output_pdf}")
    
    plt.show()

if __name__ == "__main__":
    main()
