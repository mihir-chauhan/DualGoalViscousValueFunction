import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# --- Final Style Config ---
plt.rcParams.update({
    "font.family": "sans-serif",
    # Try Montserrat first, fall back to safe sans-serifs
    "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "text.usetex": False,
    "mathtext.fontset": "cm" # Computer Modern for math (looks professional)
})

# ==========================================
# 1. 2x2 Physics Comparison (ICML Format)
# ==========================================

def solve_pde_grid(mode='eikonal', size=50, obstacle_cost=100.0, viscosity=2.5):
    # Initialize
    V = np.ones((size, size)) * 50.0
    goal = (size//2, size-5)
    V[goal] = 0.0
    
    costs = np.ones((size, size)) * 1.0
    ox, oy = size//2, size//2
    # Obstacle
    costs[oy-6:oy+6, ox-6:ox+6] = obstacle_cost
    
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Value Iteration
    for _ in range(600):
        V_new = V.copy()
        for r in range(1, size-1):
            for c in range(1, size-1):
                if (r, c) == goal: continue
                
                vals = []
                for dr, dc in actions:
                    vals.append(V[r+dr, c+dc] + costs[r, c])
                vals = np.array(vals)
                
                if mode == 'eikonal':
                    V_new[r, c] = np.min(vals)
                elif mode == 'viscous':
                    # Soft Min
                    m = np.min(vals)
                    soft_min = m - viscosity * np.log(np.sum(np.exp(-(vals - m)/viscosity)))
                    V_new[r, c] = soft_min

        if np.max(np.abs(V - V_new)) < 1e-3: break
        V = V_new
    return V

def get_gradients(V):
    dy, dx = np.gradient(-V)
    norm = np.sqrt(dx**2 + dy**2)
    mask = norm > 0
    dx[mask] /= norm[mask]
    dy[mask] /= norm[mask]
    return dx, dy

def plot_icml_grid():
    N = 50
    # Solve
    V_eik = solve_pde_grid('eikonal', size=N)
    V_visc = solve_pde_grid('viscous', size=N, viscosity=4.0)
    
    # 2x2 Layout
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    goal_x, goal_y = N//2, N-5
    
    # --- Style Settings ---
    cmap_eik = plt.cm.Blues_r
    cmap_visc = plt.cm.GnBu_r
    
    # ---------------------------------------
    # Panel (a): Eikonal Value (Top Left)
    # ---------------------------------------
    ax = axes[0, 0]
    im = ax.imshow(V_eik, cmap=cmap_eik, origin='upper')
    # Add Contours to show gradient texture
    ax.contour(V_eik, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    ax.set_title("(a) Eikonal Value Surface", color='#0d47a1', fontweight='bold')
    ax.text(goal_x, goal_y, 'G', color='white', ha='center', va='center', fontweight='bold', 
            path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------
    # Panel (b): Viscous Value (Top Right)
    # ---------------------------------------
    ax = axes[0, 1]
    im = ax.imshow(V_visc, cmap=cmap_visc, origin='upper')
    # Add Contours
    ax.contour(V_visc, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    ax.set_title("(b) Feynman-Kac Value Surface", color='#006064', fontweight='bold')
    ax.text(goal_x, goal_y, 'G', color='white', ha='center', va='center', fontweight='bold',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------
    # Panel (c): Eikonal Policy (Bottom Left)
    # ---------------------------------------
    ax = axes[1, 0]
    # Obstacle
    rect = plt.Rectangle((N//2-6, N//2-6), 12, 12, color='#90caf9', alpha=0.5)
    ax.add_patch(rect)
    
    U, V_vec = get_gradients(V_eik)
    skip = 3
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], -V_vec[::skip, ::skip], 
              color='#01579b', scale=22, headwidth=4, alpha=0.9)
    
    ax.set_title("(c) Eikonal Policy", color='#0d47a1')
    ax.set_xlabel("Grazes Obstacle Boundary", fontsize=9, color='#555555')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------
    # Panel (d): Viscous Policy (Bottom Right)
    # ---------------------------------------
    ax = axes[1, 1]
    # Obstacle
    rect = plt.Rectangle((N//2-6, N//2-6), 12, 12, color='#80deea', alpha=0.5)
    ax.add_patch(rect)
    
    U, V_vec = get_gradients(V_visc)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], -V_vec[::skip, ::skip], 
              color='#006064', scale=22, headwidth=4, alpha=0.9)
    
    ax.set_title("(d) Viscous Policy", color='#006064')
    ax.set_xlabel("Safe Margin (Smoothed)", fontsize=9, color='#555555')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig("icml_physics_2x2.png")
    print("Generated icml_physics_2x2.png")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_rounded_box(ax, center, w, h, text, color, edge, label_c='black', fontsize=10, box=True):
    if box:
        box_patch = patches.FancyBboxPatch(
            (center[0]-w/2, center[1]-h/2), w, h,
            boxstyle="round,pad=0.1",
            linewidth=2, edgecolor=edge, facecolor=color, zorder=2
        )
        ax.add_patch(box_patch)
    if text:
        lines = text.split('\n')
        y_offset = 0.15 * (len(lines) - 1) if len(lines) > 1 else 0
        for i, line in enumerate(lines):
            ax.text(center[0], center[1] + y_offset - (i*0.3), line, ha='center', va='center', 
                    fontsize=fontsize, color=label_c, zorder=4, fontweight='bold')

def draw_star_graph(ax, center, radius=0.6):
    gx, gy = center
    # Edges
    for i in range(5):
        angle = (2 * np.pi / 5) * i - np.pi/2
        sx = gx + radius * np.cos(angle)
        sy = gy + radius * np.sin(angle)
        ax.plot([gx, sx], [gy, sy], color='#0288d1', linestyle='-', linewidth=1.5, zorder=3)
        circle = patches.Circle((sx, sy), 0.12, facecolor='white', edgecolor='#555555', zorder=4)
        ax.add_patch(circle)

    # Center Node (Goal)
    circle_g = patches.Circle((gx, gy), 0.18, facecolor='#b3e5fc', edgecolor='#0277bd', linewidth=2, zorder=5)
    ax.add_patch(circle_g)
    ax.text(gx, gy, "$g$", ha='center', va='center', fontsize=11, fontweight='bold', color='#01579b', zorder=6)

def generate_final_arch_fixed():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Palette
    c_bg      = '#e3f2fd'  # Light Blue (Network)
    c_border  = '#1565c0'  # Dark Blue
    c_val     = '#e0f2f1'  # Teal (Value)
    c_val_b   = '#00695c'
    c_loss    = '#fff3e0'  # Orange (Losses)
    c_loss_b  = '#ef6c00'
    c_text    = '#37474f'

    # -----------------------------
    # 1. INPUTS (Left Column)
    # -----------------------------
    ax.text(0.8, 5.0, "$s$", fontsize=16, fontweight='bold', ha='center', color=c_border)
    ax.text(0.8, 4.0, "$g$", fontsize=16, fontweight='bold', ha='center', color=c_border)
    ax.text(0.8, 1.0, "Speed $\sigma(s)$", fontsize=11, fontweight='bold', ha='center', color='#546e7a')

    # -----------------------------
    # 2. MAIN NETWORK (Center)
    # -----------------------------
    # Container Box
    draw_rounded_box(ax, (3.5, 4.5), 3.0, 2.5, "", c_bg, c_border)
    ax.text(3.5, 5.5, "Dual Goal\nArchitecture", fontsize=11, ha='center', color=c_border, fontweight='bold')
    
    # Star Graph
    draw_star_graph(ax, (3.5, 4.3), radius=0.7)
    
    # Value Output Box
    draw_rounded_box(ax, (6.0, 4.5), 1.2, 0.8, "$V(s,g)$", c_val, c_val_b, fontsize=12)

    # -----------------------------
    # 3. SAMPLING BLOCK (Bottom Center)
    # -----------------------------
    draw_rounded_box(ax, (3.5, 2.0), 2.5, 1.0, "Gaussian Sampling\n$s' \sim \mathcal{N}(s, 2\\nu)$", '#f3e5f5', '#8e24aa', fontsize=9)

    # -----------------------------
    # 4. LOSSES (Right Column)
    # -----------------------------
    
    # Top: Viscous Relaxation
    draw_rounded_box(ax, (9.5, 5.0), 3.5, 2.0, "", c_loss, c_loss_b)
    ax.text(9.5, 5.7, "1. Viscous Relaxation", fontsize=10, fontweight='bold', color=c_loss_b, ha='center')
    ax.text(9.5, 5.1, r"$V_{targ} = V(s) + \lambda (\mathrm{LSE}(\Delta V) - \log K)$", fontsize=9, color=c_text, ha='center')
    ax.text(9.5, 4.5, r"$\mathcal{L}_{visc} \approx |\mathrm{ReLU}(V(s) - V_{targ})|^2$", fontsize=9, color=c_text, ha='center')

    # Bottom: Slope Limit
    draw_rounded_box(ax, (9.5, 2.0), 3.5, 2.0, "", c_loss, c_loss_b)
    ax.text(9.5, 2.7, "2. Metric Slope Limit", fontsize=10, fontweight='bold', color=c_loss_b, ha='center')
    ax.text(9.5, 2.1, r"$|\nabla V| \approx \frac{|V(s') - V(s)|}{\|s' - s\|}$", fontsize=9, color=c_text, ha='center')
    ax.text(9.5, 1.5, r"$\mathcal{L}_{slope} \approx |\mathrm{ReLU}(|\nabla V| - \frac{\kappa}{\sigma(s)})|^2$", fontsize=9, color=c_text, ha='center')

    # -----------------------------
    # 5. CONNECTIONS (Orthogonal Fix)
    # -----------------------------
    
    # s -> Net (Right then Up)
    ax.annotate("", xy=(2.0, 4.7), xytext=(1.0, 5.0), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64", 
                                connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    
    # g -> Net (Right then Up)
    ax.annotate("", xy=(2.0, 4.3), xytext=(1.0, 4.0), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64", 
                                connectionstyle="angle,angleA=0,angleB=90,rad=5"))

    # Net -> Value (Straight)
    ax.annotate("", xy=(5.4, 4.5), xytext=(5.0, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64"))

    # Value -> Viscous (Top)
    # Leave Right (0), Arrive Up (90)
    ax.annotate("", xy=(7.75, 5.0), xytext=(6.6, 4.5), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64", 
                                connectionstyle="angle,angleA=0,angleB=90,rad=5"))

    # Value -> Slope (Bottom)
    # Leave Right (0), Arrive Down (-90)
    ax.annotate("", xy=(7.75, 2.0), xytext=(6.6, 4.5), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#455a64", 
                                connectionstyle="angle,angleA=0,angleB=-90,rad=5"))

    # Sampling Logic (Dashed)
    # s -> Sampling Block (Down then Right)
    # Note: angleA=-90 (Down), angleB=0 (Right) guarantees intersection at (1.0, 2.0)
    ax.annotate("", xy=(2.2, 2.0), xytext=(1.0, 5.0), 
                arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color="#8e24aa", 
                                connectionstyle="angle,angleA=-90,angleB=180,rad=5"))
    
    # Sampling Block -> Net (Ghost)
    ax.annotate("", xy=(3.5, 3.2), xytext=(3.5, 2.5), 
                arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color="#8e24aa"))
    ax.text(3.6, 2.8, "eval at $s'$", fontsize=8, color="#8e24aa")

    # Speed -> Slope Limit (FIXED HERE)
    # Changed to: Right (0) then Up (90) to create an L-shape that intersects
    ax.annotate("", xy=(7.75, 1.5), xytext=(1.5, 1.0), 
                arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color="#546e7a", 
                                connectionstyle="angle,angleA=0,angleB=90,rad=5"))

    plt.tight_layout()
    plt.savefig("viscous_arch_fixed.png")
    print("Generated viscous_arch_fixed.png")


if __name__ == "__main__":
    #plot_icml_grid()
    generate_final_arch_fixed()