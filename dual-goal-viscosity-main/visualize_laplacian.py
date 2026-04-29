import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# --- Style ---
plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 10,
    "axes.titlesize": 11,
    "text.usetex": False
})

# ------------------------------------------------------------
# 1. Physics Setup (Speed Profile)
# ------------------------------------------------------------
def get_physics_grid(size=100):
    # Speed s(x): 1.0 in free space, approaches 0 in obstacle
    speed = np.ones((size, size))
    
    ox, oy = size // 2, size // 2
    r = 15
    
    # Eikonal assumes 1/speed. Feynman-Kac assumes cost in exp(-cost).
    # We use a hard obstacle for Eikonal (speed -> 0 implies cost -> inf)
    # For numerical stability in FK, we assign a very high cost to the obstacle.
    speed[oy-r:oy+r, ox-r:ox+r] = 1e-4 
    
    # Cost field: f(x) = 1/s(x)
    # Free space cost = 1.0, Obstacle cost = 10000.0
    cost_field = 1.0 / speed
    
    goal = (size - 15, size // 2)
    obstacle_mask = np.zeros((size, size), dtype=bool)
    obstacle_mask[oy-r:oy+r, ox-r:ox+r] = True
    
    return cost_field, goal, obstacle_mask, (ox, oy, r)

# ------------------------------------------------------------
# 2. Eikonal Solver (Godunov / Rouy-Tourin)
# ------------------------------------------------------------
def solve_eikonal_godunov(size=100):
    cost_map, goal, mask, _ = get_physics_grid(size)
    
    V = np.ones((size, size)) * 2000.0
    V[goal] = 0.0
    
    for _ in range(2000): # More iterations to ensure convergence
        V_prev = V.copy()
        
        # Neumann Neighbors
        V_up    = np.roll(V, -1, axis=0); V_up[-1, :] = V[-1, :]
        V_down  = np.roll(V, 1, axis=0);  V_down[0, :] = V[0, :]
        V_left  = np.roll(V, -1, axis=1); V_left[:, -1] = V[:, -1]
        V_right = np.roll(V, 1, axis=1);  V_right[:, 0] = V[:, 0]
        
        v_min_y = np.minimum(V_up, V_down)
        v_min_x = np.minimum(V_left, V_right)
        
        # Quadratic Godunov update
        diff = np.abs(v_min_x - v_min_y)
        is_1d = diff >= cost_map
        
        V_new_1d = np.minimum(v_min_x, v_min_y) + cost_map
        
        discriminant = 2 * cost_map[~is_1d]**2 - diff[~is_1d]**2
        discriminant = np.maximum(discriminant, 0)
        V_new_2d = 0.5 * (v_min_x[~is_1d] + v_min_y[~is_1d] + np.sqrt(discriminant))
        
        V_new = V_new_1d
        V_new[~is_1d] = V_new_2d
        V_new[goal] = 0.0
        
        V = V_new
        if np.max(np.abs(V - V_prev)) < 1e-4:
            break
            
    return V, mask

# ------------------------------------------------------------
# 3. Feynman-Kac Solver (Linear Space)
# ------------------------------------------------------------
def solve_feynman_kac_linear(size=100, nu=5.0): # Increased nu to 15.0
    """
    Higher nu (temperature) allows diffusion to propagate further 
    before the signal decays to machine epsilon.
    """
    cost_map, goal, mask, _ = get_physics_grid(size)
    
    Psi = np.zeros((size, size))
    Psi[goal] = 1.0 
    
    # Precompute decay. 
    # Free space decay = exp(-1.0 / 15.0) ~ 0.93 (preserves signal well)
    decay = np.exp(-cost_map / nu)
    
    for _ in range(3000): # More iterations for diffusion to wrap around
        Psi_prev = Psi.copy()
        
        Psi_up    = np.roll(Psi, -1, axis=0); Psi_up[-1, :] = Psi[-1, :]
        Psi_down  = np.roll(Psi, 1, axis=0);  Psi_down[0, :] = Psi[0, :]
        Psi_left  = np.roll(Psi, -1, axis=1); Psi_left[:, -1] = Psi[:, -1]
        Psi_right = np.roll(Psi, 1, axis=1);  Psi_right[:, 0] = Psi[:, 0]
        
        Psi_avg = 0.25 * (Psi_up + Psi_down + Psi_left + Psi_right)
        Psi_new = Psi_avg * decay
        Psi_new[goal] = 1.0
        
        Psi = Psi_new
        if np.max(np.abs(Psi - Psi_prev)) < 1e-8:
            break
            
    # Transform back: V = -nu * log(Psi)
    with np.errstate(divide='ignore'):
        V = -nu * np.log(Psi + 1e-100) # Lower floor to capture tail
    
    V[mask] = np.nan 
    return V, mask

# ------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------
def get_policy_gradient(V):
    dy, dx = np.gradient(-V) 
    norm = np.sqrt(dx**2 + dy**2)
    norm[norm < 1e-5] = 1.0
    return dx/norm, dy/norm

def plot_comparison():
    N = 100
    V_eik, mask = solve_eikonal_godunov(N)
    
    # nu=15.0 helps fill the space. 
    V_fk, _ = solve_feynman_kac_linear(N, nu=25.0)
    
    _, _, _, geom = get_physics_grid(N)
    ox, oy, r = geom

    fig, axs = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    
    # Calculate robust max for color scaling (ignoring obstacle NaNs)
    vmax_eik = np.nanpercentile(V_eik, 90)
    vmax_fk  = np.nanpercentile(V_fk, 90)
    
    # --- (a) Eikonal Value ---
    ax = axs[0, 0]
    norm_eik = mcolors.Normalize(vmin=0, vmax=vmax_eik)
    # Blue color scheme
    ax.imshow(V_eik, cmap="Blues", norm=norm_eik, origin='upper')
    ax.contour(V_eik, levels=np.linspace(0, vmax_eik, 20), colors='white', linewidths=0.5, alpha=0.5)
    
    ax.add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, edgecolor='white', facecolor='#1a237e')) # Dark Blue Obstacle
    ax.text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax.set_title("Eikonal Value ($|\nabla V| = 1/s$)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- (b) Feynman-Kac Value ---
    ax = axs[0, 1]
    norm_fk = mcolors.Normalize(vmin=0, vmax=vmax_fk)
    # Slightly different blue-green to distinguish, but still blue-based
    ax.imshow(V_fk, cmap="Blues", norm=norm_fk, origin='upper')
    ax.contour(V_fk, levels=np.linspace(0, vmax_fk, 20), colors='white', linewidths=0.5, alpha=0.5)
    
    ax.add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, edgecolor='white', facecolor='#004d40')) # Dark Teal Obstacle
    ax.text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax.set_title("Feynman-Kac Value (Linear in Log Space)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Policies ---
    stride = 5
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    
    # (c) Eikonal Policy
    ax = axs[1, 0]
    U, V_vec = get_policy_gradient(V_eik)
    ax.imshow(V_eik, cmap="Blues_r", alpha=0.3, norm=norm_eik, origin='upper')
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              U[::stride, ::stride], V_vec[::stride, ::stride], 
              scale=22, color='#0d47a1', width=0.0035, headwidth=4)
    ax.add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#1a237e'))
    ax.set_title("Eikonal Policy (Grazing)")
    ax.set_xticks([]); ax.set_yticks([])

    # (d) Feynman-Kac Policy
    ax = axs[1, 1]
    U_fk, V_vec_fk = get_policy_gradient(V_fk)
    ax.imshow(V_fk, cmap="GnBu", alpha=0.3, norm=norm_fk, origin='upper')
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              U_fk[::stride, ::stride], V_vec_fk[::stride, ::stride], 
              scale=22, color='#006064', width=0.0035, headwidth=4)
    ax.add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#004d40'))
    ax.set_title("Viscous Policy (Safe Margin)")
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig("eikonal_fk_blue_v3.png", dpi=300)
    print("Saved eikonal_fk_blue_v3.pdf")

if __name__ == "__main__":
    plot_comparison()