import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 20,
    "axes.titlesize": 20,
    "text.usetex": False
})

# ------------------------------------------------------------
# 1. Physics Setup
# ------------------------------------------------------------
def get_grid(size=100):
    speed = np.ones((size, size))
    ox, oy = size//2, size//2
    r = 15

    speed[oy-r:oy+r, ox-r:ox+r] = 1e-6
    cost = 1.0 / speed

    goal = (size-10, size//2)  # (row, col)
    mask = np.zeros((size,size), dtype=bool)
    mask[oy-r:oy+r, ox-r:ox+r] = True

    return cost, goal, mask, (ox, oy, r)

# ------------------------------------------------------------
# 2. Eikonal solver
# ------------------------------------------------------------
def solve_eikonal(size=100, iters=5000):
    cost, goal, mask, _ = get_grid(size)
    V = np.ones((size,size)) * 1e6
    V[mask] = np.inf
    V[goal] = 0.0

    for _ in range(iters):
        V_prev = V.copy()

        V_up = np.roll(V, -1, axis=0); V_up[-1,:] = V[-1,:]
        V_dn = np.roll(V,  1, axis=0); V_dn[0,:]  = V[0,:]
        V_l  = np.roll(V, -1, axis=1); V_l[:, -1] = V[:, -1]
        V_r  = np.roll(V,  1, axis=1); V_r[:, 0]  = V[:, 0]

        vmin_x = np.minimum(V_l, V_r)
        vmin_y = np.minimum(V_up, V_dn)
        diff = np.abs(vmin_x - vmin_y)

        is1d = diff >= cost
        V_new = np.minimum(vmin_x, vmin_y) + cost

        disc = 2*cost[~is1d]**2 - diff[~is1d]**2
        disc = np.maximum(disc, 0)
        V_new[~is1d] = 0.5*(vmin_x[~is1d]+vmin_y[~is1d] + np.sqrt(disc))

        V_new[mask] = np.inf
        V_new[goal] = 0.0
        V = V_new

        if np.max(np.abs(V - V_prev)) < 1e-6:
            break

    return V, mask

# ------------------------------------------------------------
# 3. FK solver (stationary linear PDE)
# ------------------------------------------------------------
def solve_fk(size=100, nu=25.0, iters=20000):
    cost, goal, mask, _ = get_grid(size)
    Psi = np.zeros((size,size))
    Psi[goal] = 1.0
    Psi[mask] = 0.0

    h = 1.0

    for _ in range(iters):
        Psi_prev = Psi.copy()

        up = np.roll(Psi, -1, axis=0); up[-1,:] = Psi[-1,:]
        dn = np.roll(Psi, 1, axis=0); dn[0,:] = Psi[0,:]
        lf = np.roll(Psi, -1, axis=1); lf[:, -1] = Psi[:, -1]
        rt = np.roll(Psi, 1, axis=1); rt[:, 0] = Psi[:, 0]

        denom = 4 + (h*h*cost)/nu
        Psi_new = (up + dn + lf + rt) / denom

        Psi_new[mask] = 0.0
        Psi_new[goal] = 1.0

        Psi = Psi_new
        if np.max(np.abs(Psi - Psi_prev)) < 1e-8:
            break

    V = -nu * np.log(Psi + 1e-100)
    V[mask] = np.nan
    return V, mask

# ------------------------------------------------------------
# 4. Gradient
# ------------------------------------------------------------
def gradient(V):
    dy, dx = np.gradient(V)
    return dx, dy

# ------------------------------------------------------------
# 5. Plot + save
# ------------------------------------------------------------
def plot_save():
    N = 100
    V_eik, mask = solve_eikonal(N)
    V_fk, _ = solve_fk(N, nu=25)

    _, _, _, geom = get_grid(N)
    ox, oy, r = geom

    X, Y = np.meshgrid(np.arange(N), np.arange(N))

    fig, ax = plt.subplots(2,2, figsize=(12,10), constrained_layout=True)

    # compute gradient fields
    dx_e, dy_e = gradient(V_eik)
    dx_f, dy_f = gradient(V_fk)
    mag_fk = np.sqrt(dx_f**2 + dy_f**2)
    mag_fk[mag_fk < 1e-8] = 1.0
    # normalized direction (policy)
    Ux_fk = dx_f / mag_fk
    Uy_fk = dy_f / mag_fk

    # normalize
    mag_e = np.sqrt(dx_e**2 + dy_e**2) + 1e-8
    mag_f = np.sqrt(dx_f**2 + dy_f**2) + 1e-8

    dx_e /= mag_e
    dy_e /= mag_e
    dx_f /= mag_f
    dy_f /= mag_f

    # --- (a) Eikonal Value ---
    ax[0,0].imshow(V_eik, origin='upper', cmap='Blues_r')
    ax[0,0].contour(V_eik, levels=20, colors='black', linewidths=0.5)
    ax[0,0].add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#0047AB'))
    ax[0,0].text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax[0,0].set_title("Eikonal Value")
    ax[0,0].axis('off')

    # --- (b) FK Value ---
    ax[0,1].imshow(V_fk, origin='upper', cmap='Blues_r')
    ax[0,1].contour(V_fk, levels=20, colors='black', linewidths=0.5)
    ax[0,1].add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#0047AB'))
    ax[0,1].text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax[0,1].set_title("FK Value")
    ax[0,1].axis('off')

    # --- (c) Eikonal Policy ---
    ax[1,0].imshow(V_eik, origin='upper', cmap='Blues_r', alpha=0.4)
    ax[1,0].quiver(
        X[::6, ::6], Y[::6, ::6],
        -dx_e[::6, ::6],  # x component
        +dy_e[::6, ::6],  # y component flipped
        scale=30, width=0.003, color='teal'
        )
    ax[1,0].add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#0047AB'))
    ax[1,0].text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax[1,0].set_title("Eikonal Policy (-∇V)")
    ax[1,0].axis('off')

    # --- (d) FK Policy ---
    ax[1,1].imshow(V_fk, origin='upper', cmap='Blues_r', alpha=0.4)

    stride = 6
    ax[1,1].quiver(
        X[::6, ::6], Y[::6, ::6],
        #-dx_f[::6, ::6],
        #+dy_f[::6, ::6],
        -Ux_fk[::stride, ::stride]*0.2 * mag_fk[::stride, ::stride],
        +Uy_fk[::stride, ::stride]*0.2 * mag_fk[::stride, ::stride],
        scale=30, width=0.003, color='teal'
    )
    ax[1,1].add_patch(patches.Rectangle((ox-r, oy-r), 2*r, 2*r, color='#0047AB'))
    ax[1,1].text(ox, oy, "OBSTACLE", color='white', ha='center', va='center', fontsize=8)
    ax[1,1].set_title("FK Policy (-∇V)")
    ax[1,1].axis('off')

    plt.savefig("eikonal_vs_fk_correct_4panel.png", dpi=300, bbox_inches="tight")
    print("Saved: eikonal_vs_fk_correct_4panel.png")

if __name__ == "__main__":
    plot_save()
