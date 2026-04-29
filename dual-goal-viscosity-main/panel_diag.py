import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# --- Professional Style Config ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "text.usetex": False 
})

def solve_pde_grid(mode='eikonal', size=40, obstacle_cost=100.0, viscosity=2.5):
    # Initialize Grid
    V = np.ones((size, size)) * 50.0
    goal = (size//2, size-4)
    V[goal] = 0.0
    
    costs = np.ones((size, size)) * 1.0
    ox, oy = size//2, size//2
    # Obstacle
    costs[oy-5:oy+5, ox-5:ox+5] = obstacle_cost
    
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for _ in range(600):
        V_new = V.copy()
        for r in range(1, size-1):
            for c in range(1, size-1):
                if (r, c) == goal: continue
                
                neighbor_vals = []
                for dr, dc in actions:
                    val = V[r+dr, c+dc]
                    step_cost = costs[r, c]
                    neighbor_vals.append(val + step_cost)
                neighbor_vals = np.array(neighbor_vals)
                
                if mode == 'eikonal':
                    V_new[r, c] = np.min(neighbor_vals)
                elif mode == 'viscous':
                    # Soft Min (Log-Sum-Exp)
                    m = np.min(neighbor_vals)
                    # Note: We use a slightly different update for visualization stability
                    soft_min = m - viscosity * np.log(np.sum(np.exp(-(neighbor_vals - m)/viscosity)))
                    V_new[r, c] = soft_min

        if np.max(np.abs(V - V_new)) < 1e-3:
            break
        V = V_new
    return V

def get_gradients(V):
    dy, dx = np.gradient(-V)
    norm = np.sqrt(dx**2 + dy**2)
    mask = norm > 0
    dx[mask] /= norm[mask]
    dy[mask] /= norm[mask]
    return dx, dy

def plot_icml_2x2():
    N = 40
    # Solve Physics (Exact parameters from your "correct" snippet)
    V_eik = solve_pde_grid('eikonal', size=N)
    V_visc = solve_pde_grid('viscous', size=N, viscosity=4.0)
    
    # 2x2 Layout for ICML
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    goal_x, goal_y = N//2, N-4
    
    # --- Colormaps ---
    # Using reversed maps so Low Value (Goal) is Light, High Value is Dark
    cmap_eik = plt.cm.Blues
    cmap_visc = plt.cm.GnBu

    # --- Independent Normalization ---
    # This ensures the Feynman-Kac plot is NOT blank.
    # We normalize each plot to its own range so the gradient is visible in both.
    def normalize(V):
        v_min, v_max = V.min(), np.percentile(V, 90)
        return mcolors.Normalize(vmin=v_min, vmax=v_max)

    norm_eik = normalize(V_eik)
    norm_visc = normalize(V_visc)

    # ---------------------------------------
    # Panel (a): Eikonal Value (Top Left)
    # ---------------------------------------
    ax = axes[0, 0]
    # Note: We invert the data logic for the cmap so Goal (Low V) = Light Color
    # Actually, standard cmap (Blues) maps Low->Light, High->Dark.
    # So we plot V directly.
    im = ax.imshow(V_eik, cmap=cmap_eik, norm=norm_eik, origin='upper')
    
    # Add Contours (White lines for the concentric look)
    ax.contour(V_eik, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    ax.set_title("(a) Eikonal Value Surface", color='#0d47a1', fontweight='bold')
    ax.scatter([goal_x], [goal_y], c='#ff5252', s=40, zorder=10, edgecolors='white', linewidth=1)
    ax.text(goal_x, goal_y-3, 'G', color='black', ha='center', va='center', fontweight='bold',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------
    # Panel (b): Viscous Value (Top Right)
    # ---------------------------------------
    ax = axes[0, 1]
    # Independent normalization ensures this isn't blank!
    im = ax.imshow(V_visc, cmap=cmap_visc, norm=norm_visc, origin='upper')
    
    # Add Contours
    ax.contour(V_visc, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    ax.set_title("(b) Feynman-Kac Value Surface", color='#006064', fontweight='bold')
    ax.scatter([goal_x], [goal_y], c='#ff5252', s=40, zorder=10, edgecolors='white', linewidth=1)
    ax.text(goal_x, goal_y-3, 'G', color='black', ha='center', va='center', fontweight='bold',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.set_xticks([]); ax.set_yticks([])

    # ---------------------------------------
    # Panel (c): Eikonal Policy (Bottom Left)
    # ---------------------------------------
    ax = axes[1, 0]
    rect = plt.Rectangle((N//2-5, N//2-5), 10, 10, color='#90caf9', alpha=0.5) # Light Blue Obstacle
    ax.add_patch(rect)
    
    U, V_vec = get_gradients(V_eik)
    skip = 2
    # Dark Navy Arrows
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
    rect = plt.Rectangle((N//2-5, N//2-5), 10, 10, color='#80deea', alpha=0.5) # Cyan Obstacle
    ax.add_patch(rect)
    
    U, V_vec = get_gradients(V_visc)
    # Dark Teal Arrows
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], -V_vec[::skip, ::skip], 
              color='#006064', scale=22, headwidth=4, alpha=0.9)
    
    ax.set_title("(d) Viscous Policy", color='#006064')
    ax.set_xlabel("Safe Margin (Smoothed)", fontsize=9, color='#555555')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig("icml_physics_2x2.pdf")
    print("Generated icml_physics_2x2.pdf")

if __name__ == "__main__":
    plot_icml_2x2()