import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_final_architecture_v4():
    # --- 1. Global Aesthetic Config ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm' 
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Palette
    blue_fill = '#E3F2FD'
    blue_edge = '#1565C0'
    green_fill = '#E3F2FD'
    green_edge = '#1565C0'
    purple_fill = '#FFF' # Light Purple
    purple_edge = '#1565C0' # Dark Purple
    orange_fill = '#FFF'
    orange_edge = '#1565C0'
    text_color = '#212121'

    # --- 2. Inputs (Left) ---
    # Box s
    rect_s = patches.FancyBboxPatch((0.5, 6.0), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor=blue_fill, edgecolor=blue_edge, linewidth=1.5)
    ax.add_patch(rect_s)
    ax.text(1.25, 6.4, "$s$", fontsize=32, ha='center', va='center', color=text_color)

    # Box g
    rect_g = patches.FancyBboxPatch((0.5, 4.0), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor=blue_fill, edgecolor=blue_edge, linewidth=1.5)
    ax.add_patch(rect_g)
    ax.text(1.25, 4.4, "$g$", fontsize=32, ha='center', va='center', color=text_color)

    # --- 3. Main Architecture (Center) ---
    rect_main = patches.FancyBboxPatch((3.5, 3.0), 4.0, 4.5, boxstyle="round,pad=0.2", 
                                       facecolor=blue_fill, edgecolor=blue_edge, linewidth=1.5)
    ax.add_patch(rect_main)
    ax.text(5.5, 7.1, "Dual Goal\nArchitecture", fontsize=24, fontweight='bold', ha='center', va='center', color=text_color)
    
    # Star Graph Icon
    center_x, center_y = 5.5, 5.0
    for angle in np.linspace(0, 2*np.pi, 6)[:-1]:
        lx = center_x + 1.5 * np.cos(angle)
        ly = center_y + 1.5 * np.sin(angle)
        ax.plot([center_x, lx], [center_y, ly], color=blue_edge, lw=1.5, zorder=1)
        circle_l = patches.Circle((lx, ly), 0.3, facecolor='white', edgecolor=blue_edge, zorder=2)
        ax.add_patch(circle_l)
    circle_c = patches.Circle((center_x, center_y), 0.6, facecolor=blue_edge, edgecolor=blue_edge, zorder=3)
    ax.add_patch(circle_c)
    ax.text(center_x, center_y, "$g$", color='white', fontsize=16, ha='center', va='center', zorder=4)

    # --- 4. Gaussian Sampling (Restored, Bottom Center) ---
    rect_sample = patches.FancyBboxPatch((3.5, 0.5), 4.0, 1.5, boxstyle="round,pad=0.1", 
                                         facecolor=purple_fill, edgecolor=purple_edge, linewidth=1.5)
    ax.add_patch(rect_sample)
    ax.text(5.5, 1.5, "Gaussian Sampling", fontsize=24, fontweight='bold', ha='center', va='center', color=text_color)
    ax.text(5.5, 0.9, "$s' \sim \mathcal{N}(s, \lambda I)$", fontsize=25, ha='center', va='center', color=text_color)

    # --- 5. Output V (Center Right) ---
    rect_v = patches.FancyBboxPatch((8.2, 4.5), 1.8, 1.5, boxstyle="round,pad=0.1", 
                                    facecolor=green_fill, edgecolor=green_edge, linewidth=1.5)
    ax.add_patch(rect_v)
    ax.text(9.1, 5.25, "$V(s, g)$", fontsize=32, ha='center', va='center', color=text_color)

    # --- 6. Loss Modules (Right) ---
    
    # Top: Viscous Relaxation
    rect_l1 = patches.FancyBboxPatch((11.0, 4.8), 4.8, 3.2, boxstyle="round,pad=0.1", 
                                     facecolor=orange_fill, edgecolor=orange_edge, linewidth=1.5)
    ax.add_patch(rect_l1)
    ax.text(13.4, 7.5, "1. Viscous Relaxation", fontsize=24, fontweight='bold', ha='center', va='center', color=text_color)
    
    eq1_text = r"$\hat V(s,g) \leftarrow -2\nu\log\!\left(\frac{1}{K}\sum_k e^{-c(s,s'_k)+\frac{V(s'_k,g)}{2\nu}}\right)$"
    eq2_text = r"$\mathcal{L}_{\text{visc}}=\|V(s,g)-\mathrm{sg}(\hat V(s,g))\|^2$"

    
    ax.text(13.4, 6.4, eq1_text, fontsize=20, ha='center', va='center', color=text_color)
    ax.text(13.4, 5.5, eq2_text, fontsize=20, ha='center', va='center', color=text_color)

    # Bottom: Metric Slope Limit
    rect_l2 = patches.FancyBboxPatch((11.0, 0.5), 4.8, 3.2, boxstyle="round,pad=0.1", 
                                     facecolor=orange_fill, edgecolor=orange_edge, linewidth=1.5)
    ax.add_patch(rect_l2)
    ax.text(13.4, 3.2, "2. Metric Slope Limit", fontsize=24, fontweight='bold', ha='center', va='center', color=text_color)
    
    eq3_text = r"$\text{Slope} \approx \frac{|V - V'|}{\|s - s'\|}$"
    eq4_text = r"$\mathcal{L}_{slope} = \mathbb{E} \left[ (\text{Slope} > C(s))^2 \right]$"
    
    ax.text(13.4, 2.1, eq3_text, fontsize=25, ha='center', va='center', color=text_color)
    ax.text(13.4, 1.2, eq4_text, fontsize=25, ha='center', va='center', color=text_color)

    # --- 7. Wiring ---
    
    # s -> Main
    ax.annotate("", xy=(3.5, 5.5), xytext=(2.0, 6.4), 
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    
    # g -> Main
    ax.annotate("", xy=(3.5, 5.0), xytext=(2.0, 4.4), 
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5, connectionstyle="arc3,rad=0.1"))
    
    # s -> Gaussian Sampling (Dashed)
    ax.plot([1.25, 1.25], [6.0, 1.25], color="#888", ls="--", lw=1.5) # Drop down from s
    ax.annotate("", xy=(3.5, 1.25), xytext=(1.25, 1.25), 
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5, ls="--"))

    # Gaussian Sampling -> Main (Eval at s')
    # Dashed arrow going up
    ax.annotate("", xy=(5.5, 3.0), xytext=(5.5, 2.0), 
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5, ls="--"))
    # Label placed cleanly to the right
    ax.text(5.6, 2.5, "eval at $s'$", fontsize=22, color="#555", ha='left', va='center')
    
    # Main -> V
    ax.annotate("", xy=(8.2, 5.25), xytext=(7.5, 5.25), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # V -> Split
    ax.plot([10.0, 10.5], [5.25, 5.25], color="#555", lw=1.5) # stem
    ax.plot([10.5, 10.5], [2.1, 6.4], color="#555", lw=1.5)   # vertical bar
    ax.annotate("", xy=(11.0, 6.4), xytext=(10.5, 6.4), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5)) # to top
    ax.annotate("", xy=(11.0, 2.1), xytext=(10.5, 2.1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5)) # to bottom

    plt.tight_layout()
    plt.savefig('architecture_diagram_v4.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    draw_final_architecture_v4()