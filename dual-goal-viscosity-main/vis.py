from multiprocessing.sharedctypes import Value
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax
import jax.numpy as jnp
import mujoco
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.env_utils import (
    generate_obstacle_coordinates,
    compute_speed_profile,
    compute_exponential_speed_profile,
)
from rich import print
from rich.console import Console

from matplotlib import cm

cmap = cm.get_cmap("Greys_r", 20)
c = [cmap(i) for i in range(cmap.N)][::-1]  # reverse colors
#ax.contour(Y, X, V, levels=20, colors=c, linewidths=1, alpha=1.0, zorder=1)

console = Console()

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "antmaze-large-navigate-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0.0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_float("eval_goal_gaussian", None, "Goal Gaussian noise for evaluation.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

config_flags.DEFINE_config_file("agent", "agents/crl/id.py", lock_config=False)

plt.rcParams.update({
    "font.family": "sans-serif",
    # Try Montserrat first, fall back to safe sans-serifs
    "font.sans-serif": ["Montserrat", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "text.usetex": False,
    "mathtext.fontset": "cm" # Computer Modern for math (looks professional)
})

# --- JAX RNG Helper (Replicates your supply_rng) ---
def supply_rng(f, rng):
    """Wraps a function to automatically supply a split RNG key."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        # Pass the key as 'seed' which is standard for many Flax agents
        # If your agent expects 'rng', change this kwarg to 'rng'
        return f(*args, seed=key, **kwargs) 
    return wrapped

# --- Helper: Generate a real rollout (Aligned with your evaluate logic) ---
def get_rollout(env, agent, goal_pos, max_steps=200, temperature=0.0):
    """
    Runs the agent to get a trajectory.
    Mimics the logic of 'evaluate' but forces a specific goal for visualization.
    """
    # 1. Setup Actor with RNG
    # We use a random seed for the rollout to ensure stochasticity
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    actor_fn = supply_rng(agent.sample_actions, rng=rng)

    # 2. Reset and Handle API differences
    reset_res = env.reset()
    if isinstance(reset_res, tuple) and len(reset_res) == 2 and isinstance(reset_res[1], dict):
        obs = reset_res[0] # Gymnasium
    else:
        obs = reset_res    # Old Gym
        
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    
    trajectory = []
    
    # 3. Prepare Goal with Padding
    # We ignore the env's internal goal and FORCE the visualization goal [1, 1]
    
    # Determine full observation dimension for padding
    if hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None:
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = obs.shape[0] if not isinstance(obs, dict) else obs['observation'].shape[0]

    # Create base goal [1, 2] -> Scaled by maze unit
    # goal_xy shape becomes (1, 2)
    goal_xy = np.array([goal_pos], dtype=np.float32) * maze_unit
    
    # Pad to match observation size (e.g. 69)
    # The agent expects: (batch, dim)
    if obs_dim > 2:
        padding = np.zeros((1, obs_dim - 2), dtype=np.float32)
        # FIX: goal_xy is already (1, 2), so we concatenate directly
        goal_input = np.concatenate([goal_xy, padding], axis=-1)
    else:
        goal_input = goal_xy

    # 4. Rollout Loop
    for _ in range(max_steps):
        # -- Extract Position (x,y) for trajectory logging --
        if isinstance(obs, dict):
            curr_obs = obs['observation'] if 'observation' in obs else obs['achieved_goal']
            pos = curr_obs[:2]
        else:
            pos = obs[:2]
            
        trajectory.append(pos)
        
        # -- Check success (Euclidean distance on x,y only) --
        # Compare current pos to unpadded goal (extract from batch)
        if np.linalg.norm(pos - goal_xy[0]) < 0.5:
            break

        # -- Prepare Inputs for Agent --
        if isinstance(obs, dict):
            obs_data = obs['observation'] if 'observation' in obs else obs
            obs_batch = obs_data[None, :]
        else:
            obs_batch = obs[None, :]
        
        # -- Sample Action (Using the actor_fn wrapper) --
        # Matches your evaluate signature: observations, goals, temperature
        action = actor_fn(observations=obs_batch, goals=goal_input, temperature=temperature)
        
        # Unpack if necessary (handle batch dim and tuples)
        if isinstance(action, tuple): 
            action = action[0]
        
        # Ensure it's a flat numpy array for the env
        action = np.array(action)
        if action.ndim > 1:
            action = action[0]

        # -- Step Environment --
        step_res = env.step(action)
        
        # Handle 4-tuple (old) vs 5-tuple (new)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            obs, reward, done, info = step_res

        if done:
            # Capture final position
            if isinstance(obs, dict):
                curr_obs = obs['observation'] if 'observation' in obs else obs['achieved_goal']
                trajectory.append(curr_obs[:2])
            else:
                trajectory.append(obs[:2])
            break
            
    return np.array(trajectory)

def get_recovery_rollout(env, agent, goal_pos, perturb_step=-1, perturb_std=0.1, seed=0):
    """
    Runs a single trajectory. If perturb_step >= 0, adds Gaussian noise at that step.
    """
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    np_rng = np.random.default_rng(seed)
    actor_fn = supply_rng(agent.sample_actions, rng=rng)

    # 2. Reset and Handle API differences
    try:
        # Gymnasium API
        reset_res = env.reset(seed=seed)
    except TypeError:
        # Old Gym API
        env.seed(seed)
        reset_res = env.reset()

    if isinstance(reset_res, tuple): obs = reset_res[0]
    else: obs = reset_res
        
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    
    trajectory = []
    
    # 3. Prepare Goal with Padding
    # We ignore the env's internal goal and FORCE the visualization goal [1, 1]
    
    # Determine full observation dimension for padding
    if hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None:
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = obs.shape[0] if not isinstance(obs, dict) else obs['observation'].shape[0]

    # Create base goal [1, 2] -> Scaled by maze unit
    # goal_xy shape becomes (1, 2)
    goal_xy = np.array([goal_pos], dtype=np.float32) * maze_unit
    
    # Pad to match observation size (e.g. 69)
    # The agent expects: (batch, dim)
    if obs_dim > 2:
        padding = np.zeros((1, obs_dim - 2), dtype=np.float32)
        # FIX: goal_xy is already (1, 2), so we concatenate directly
        goal_input = np.concatenate([goal_xy, padding], axis=-1)
    else:
        goal_input = goal_xy
    max_steps = 300
    
    for t in range(max_steps):
        # --- A. Perturbation Event ---
        if isinstance(obs, dict):
            curr_pos = obs['observation'][:2]
            obs_full = obs['observation']
        else:
            curr_pos = obs[:2]
            obs_full = obs

        # Prepare input batch
        obs_input = obs_full.copy()
        
        # --- APPLY PERTURBATION (OBSERVATION ONLY) ---
        if t == perturb_step:
            # 1. Get real physics state
            qpos = env_u.data.qpos.copy()
            qvel = env_u.data.qvel.copy()
            
            # 2. Add physical noise to position
            noise = np_rng.normal(loc=0.0, scale=perturb_std, size=2)
            qpos[:2] += noise
            
            # 3. Set state (Teleport the robot)
            # We keep qvel the same, so it "skids" to the side
            env_u.set_state(qpos, qvel)
            
            # 4. CRITICAL: Re-fetch observation so agent sees the new spot
            # (In some envs, we might need a manual get_obs, 
            # but usually the next step handles it. We log the qpos manually here.)
            trajectory.append(qpos[:2].copy())
            
            # Update 'curr_pos' for the log below
            if isinstance(obs, dict): obs['observation'][:2] = qpos[:2]
            else: obs[:2] = qpos[:2]
        else:
            # Log real physical position
            trajectory.append(curr_pos.copy())

        # Check Success (Always check against REAL position)
        if np.linalg.norm(curr_pos - goal_xy[0]) < 0.5:
            break

        # --- Select Action ---
        # We pass the potentially perturbed 'obs_input' to the actor
        obs_batch = obs_input[None, :]
            
        action = actor_fn(observations=obs_batch, goals=goal_input, temperature=0.0)
        if isinstance(action, tuple): action = action[0]
        action = np.array(action).flatten()
        
        # --- Step Environment (Physics uses REAL action from potentially FAKE obs) ---
        step_res = env.step(action)
        if len(step_res) == 5: obs, _, terminated, truncated, _ = step_res
        else: obs, _, done, _ = step_res; terminated, truncated = done, done
            
        if terminated or truncated:
            # Log final point
            if isinstance(obs, dict): trajectory.append(obs['observation'][:2])
            else: trajectory.append(obs[:2])
            break
            
    return np.array(trajectory)

def vis_single_perturbation(env, agent, goal_pos=[1, 1], perturb_std=0.1, save_name="perturb_plot"):
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    dim_other = env.observation_space.shape[0] - 2

    # --- 1. Compute Value Grid (EXACT COPY OF YOUR ORIGINAL LOGIC) ---
    print("Computing Value Grid...")
    maze_size = env_u.maze_map.shape[0]
    h, w = env_u.maze_map.shape 

    # Calculate ranges independently (Exact match to your snippet)
    range_min = -maze_unit * 1.5
    range_max_x = (w - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    range_max_y = (h - 1) * maze_unit - maze_unit * 1.5 + maze_unit

    # Create linspaces that match the actual maze dimensions
    grid_size = 100
    x = np.linspace(range_min, range_max_x, grid_size)
    y = np.linspace(range_min, range_max_y, grid_size)

    X, Y = np.meshgrid(x, y)
    
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    goals = np.array([[1, 1]], dtype=np.float32) * maze_unit
    goals_all = np.repeat(goals, grid_points.shape[0], axis=0)

    if dim_other > 0:
        grid_other = np.zeros((grid_points.shape[0], dim_other), dtype=np.float32)
        grid_points = np.concatenate([grid_points, grid_other], axis=-1)
        goals_all = np.concatenate([goals_all, grid_other], axis=-1)

    # Vmap Function
    def value_fn_single(pt, g):
        pt = pt[None, :]
        g = g[None, :]
        rep = agent.network.select("rep_value")(g)
        v1, v2 = agent.network.select("value")(pt, rep)
        return ((v1 + v2) / 2.0).squeeze()
    
    value_fn_batched = jax.jit(jax.vmap(value_fn_single, in_axes=(0, 0)))

    chunk_size = 5000
    V_chunks = []
    for i in range(0, len(grid_points), chunk_size):
        V_chunks.append(value_fn_batched(grid_points[i:i+chunk_size], goals_all[i:i+chunk_size]))
    V = jnp.concatenate(V_chunks).reshape(X.shape)

    # --- 2. Run Trajectories ---
    print(f"Simulating Perturbation (std={perturb_std})...")
    
    # A. Nominal Trajectory (Seed 42)
    
    perturb_step = 120
    traj_skew = get_recovery_rollout(env, agent, goal_pos, perturb_step=perturb_step, perturb_std=perturb_std, seed=42)
    print("Len of traj_skew:", len(traj_skew))
    
    #traj_base = get_recovery_rollout(env, agent, goal_pos, perturb_step=-1, seed=42)
    
    # B. Perturbed Trajectory (Seed 42 + perturbation)
    # We use the same seed so the trajectory is identical UP TO the perturbation
    #perturb_step = len(traj_base) // 2
    #print("Len of traj_base:", len(traj_base))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # A. Value Contours
    vmin, vmax = -72, 0 
    contour = ax.contourf(X, Y, V, levels=grid_size, cmap="Blues_r", zorder=0, vmin=vmin, vmax=vmax, alpha=0.9)
    ax.contour(X, Y, V, levels=20, colors="k", linewidths=0.5, alpha=0.5, zorder=1)
    
    # B. Maze Walls (Using your helper function logic)
    # Assuming plot_maze_on_ax is available in your scope as per your snippet
    # If not, this function call is what you requested to keep.
    try:
        plot_maze_on_ax(env, ax)
    except NameError:
        print("Warning: 'plot_maze_on_ax' helper not found. Skipping wall render.")

    # C. Nominal Trajectory (Black Dotted)
    # Plot the FULL baseline
    ax.plot(
        traj_skew[:perturb_step, 0], traj_skew[:perturb_step, 1], 
        color='black', linestyle=':', linewidth=3.0, 
        alpha=0.9, zorder=5, label="Nominal"
    )
    ax.scatter(traj_skew[-1, 0], traj_skew[-1, 1], marker='x', s=100, color='black', zorder=5)

    # D. Perturbed Trajectory (Red Dotted)
    # ONLY plot from the perturbation point onwards to avoid overlapping the black line
    # We include one point before (perturb_step - 1) to connect the lines visually
    start_vis_idx = max(0, perturb_step - 1)
    
    ax.plot(
        traj_skew[start_vis_idx:, 0], traj_skew[start_vis_idx:, 1], 
        color='red', linestyle=':', linewidth=3.0, 
        alpha=0.9, zorder=5, label="Perturbed"
    )
    
    # E. The "Skew Point" Marker (Blue Circle)
    if perturb_step < len(traj_skew):
        ax.scatter(
            traj_skew[perturb_step, 0], traj_skew[perturb_step, 1], 
            s=150, marker='o', color='white', edgecolors='grey', linewidth=2, zorder=7, label="Skew Point"
        )

    # --- 4. Formatting ---
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Stochastic Recovery ($\sigma={perturb_std}$)")
    
    divider = make_axes_locatable(ax)
    cax_val = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(contour, cax=cax_val, label="Value $V(s)$")

    save_path = f"{save_name}_std{perturb_std}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Vis] Saved to {save_path}")

# --- The "Reviewer 2" Visualization Function ---
def vis_action_uncertainty(env, agent, trajectory, beta=1.0, save_name="combined"):
    """
    Overlays Lookahead Uncertainty 'Fans' onto the Value Function Heatmap.
    Uses V(s+a) to determine shape, allowing for non-circular (multi-modal) distributions.
    """
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    maze_map = env_u.maze_map
    dim_other = env.observation_space.shape[0] - 2

    # --- 1. Compute Value Function Grid ---
    print("Computing Value Grid...")
    maze_size = env_u.maze_map.shape[0]
    range_min = -maze_unit * 1.5
    range_max = (maze_size - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    grid_size = 100
    h, w = env_u.maze_map.shape 

    # Calculate ranges independently
    range_min = -maze_unit * 1.5
    range_max_x = (w - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    range_max_y = (h - 1) * maze_unit - maze_unit * 1.5 + maze_unit

    # Create linspaces that match the actual maze dimensions
    x = np.linspace(range_min, range_max_x, grid_size)
    y = np.linspace(range_min, range_max_y, grid_size)

    X, Y = np.meshgrid(x, y)
    
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    goals = np.array([[1, 1]], dtype=np.float32) * maze_unit
    goals_all = np.repeat(goals, grid_points.shape[0], axis=0)

    if dim_other > 0:
        grid_other = np.zeros((grid_points.shape[0], dim_other), dtype=np.float32)
        grid_points = np.concatenate([grid_points, grid_other], axis=-1)
        goals_all = np.concatenate([goals_all, grid_other], axis=-1)

    # JAX Funcs
    # FIX 1: We explicitly use vmap to ensure (N,) input gives (N,) output
    # preventing the (N, N) outer product bug.
    def value_fn_single(pt, g):
        # We must add batch dim for the network, then squeeze output
        pt = pt[None, :]
        g = g[None, :]
        rep = agent.network.select("rep_value")(g)
        v1, v2 = agent.network.select("value")(pt, rep)
        return ((v1 + v2) / 2.0).squeeze()

    # Vmap over the batch dimension (axis 0 for both inputs)
    value_fn_batched = jax.jit(jax.vmap(value_fn_single, in_axes=(0, 0)))

    # Compute V for grid (safely batched to avoid OOM)
    # We do simple chunking here just in case grid is huge
    chunk_size = 5000
    V_chunks = []
    for i in range(0, len(grid_points), chunk_size):
        V_chunks.append(value_fn_batched(grid_points[i:i+chunk_size], goals_all[i:i+chunk_size]))
    V = jnp.concatenate(V_chunks).reshape(X.shape)

    # --- 2. Compute Lookahead Uncertainty (The Shapes) ---
    print(f"Computing Lookahead Uncertainty (Beta={beta})...")
    
    # A. Setup Lookahead Circle
    n_angles = 72
    angles = np.linspace(0, 2*np.pi, n_angles)
    lookahead_dist = maze_unit * 0.6 
    circle_offsets = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * lookahead_dist

    # B. Subsample Trajectory
    step_size = max(1, len(trajectory) // 30) 
    traj_subset = trajectory[::step_size]
    
    # C. Prepare Batch Inputs
    candidate_states = traj_subset[:, None, :] + circle_offsets[None, :, :]
    flat_candidates = candidate_states.reshape(-1, 2)
    
    if dim_other > 0:
        flat_padding = np.zeros((flat_candidates.shape[0], dim_other), dtype=np.float32)
        flat_candidates = np.concatenate([flat_candidates, flat_padding], axis=-1)
    
    flat_goals = np.repeat(goals, flat_candidates.shape[0], axis=0)
    if dim_other > 0:
        goal_padding = np.zeros((flat_goals.shape[0], dim_other), dtype=np.float32)
        flat_goals = np.concatenate([flat_goals, goal_padding], axis=-1)

    # D. Query Values (Using fixed vmapped function)
    flat_values = value_fn_batched(flat_candidates, flat_goals)
    action_values = flat_values.reshape(len(traj_subset), n_angles)

    # E. Create Polygons
    polygons = []
    entropies = []
    fan_scale = maze_unit * 0.5 # Visual size
    
    for i in range(len(traj_subset)):
        pos = traj_subset[i]
        values = action_values[i]
        
        # Softmax over values -> Probabilities
        logits = beta * values
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

        # Draw Polygon: Radius is proportional to probability
        # This creates the "Blob" / "Needle" shapes, NOT circles.
        visual_radii = (probs / probs.max()) * fan_scale
        
        poly_points = []
        for j, angle in enumerate(angles):
            px = pos[0] + visual_radii[j] * np.cos(angle)
            py = pos[1] + visual_radii[j] * np.sin(angle)
            poly_points.append([px, py])
        polygons.append(patches.Polygon(poly_points, closed=True))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # A. Value Contours (Background)
    vmin, vmax = -72, 0 
    contour = ax.contourf(X, Y, V, levels=grid_size, cmap="Blues_r", zorder=0, vmin=vmin, vmax=vmax, alpha=0.9)
    ax.contour(X, Y, V, levels=20, colors="k", linewidths=0.5, alpha=0.5, zorder=1)
    
    # B. Maze Walls
    plot_maze_on_ax(env, ax)
    
    # C. Trajectory
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], 
        color='black', linestyle=':', linewidth=2.5, 
        alpha=0.9, zorder=5
    )
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', s=100, color='black', zorder=5)
    
    # D. Uncertainty Fans
    entropies = np.array(entropies)
    if len(entropies) > 0:
        norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-6)
    else:
        norm_entropy = entropies

    p = PatchCollection(polygons, cmap="RdPu", alpha=0.6, zorder=4)
    p.set_array(norm_entropy) 
    ax.add_collection(p)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    divider = make_axes_locatable(ax)
    # Create a slot on the Right for Value, and on the Left for Advantage
    cax_val = divider.append_axes("right", size="5%", pad=0.05)
    cax_adv = divider.append_axes("left", size="5%", pad=0.05)
    
    # Plot colorbars into those specific slots
    cbar1 = plt.colorbar(contour, cax=cax_val, label="Value")
    
    cbar2 = plt.colorbar(p, cax=cax_adv, label="Advantage")
    # Move ticks/label to the outer left side
    cbar2.ax.yaxis.set_ticks_position('left')
    cbar2.ax.yaxis.set_label_position('left')
    
    ax.set_title(f"Trajectory with Advantage")
    
    save_path = Path("plots") / f"{save_name}_beta{beta}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Vis] Saved plot to {save_path}")

def plot_maze_on_ax(env, ax):
    env_u = env.unwrapped
    maze_map = env_u.maze_map
    maze_unit = env_u._maze_unit

    for i in range(0, maze_map.shape[0]):
        for j in range(0, maze_map.shape[1]):
            if maze_map[i, j] == 1:
                wall = np.array([j, i])
                wall = wall * maze_unit - maze_unit * 1.5
                rect = patches.Rectangle(
                    wall, maze_unit, maze_unit,
                    edgecolor=None, facecolor="white", alpha=1, zorder=2
                )
                ax.add_patch(rect)

def vis_combined_heatmap_and_uncertainty(env, agent, trajectory, beta=1.0, save_name="combined"):
    """
    Overlays Uncertainty 'Fans' onto the Value Function Heatmap.
    Refined for less 'blobby' look and clearer trajectory.
    """
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    maze_map = env_u.maze_map
    dim_other = env.observation_space.shape[0] - 2

    # --- 1. Compute Value Function Grid ---
    print("Computing Value Grid...")
    maze_size = env_u.maze_map.shape[0]
    range_min = -maze_unit * 1.5
    range_max = (maze_size - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    grid_size = 100
    x = np.linspace(range_min, range_max, grid_size)
    y = np.linspace(range_min, range_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Masking
    M = maze_map.repeat(maze_unit, axis=0).repeat(maze_unit, axis=1)
    X_idx = ((X - range_min).astype(int)).clip(0, M.shape[1] - 1)
    Y_idx = ((Y - range_min).astype(int)).astype(int).clip(0, M.shape[0] - 1)
    M = 1 - M[Y_idx, X_idx]

    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    goals = np.array([[1, 1]], dtype=np.float32) * maze_unit
    goals_all = np.repeat(goals, grid_points.shape[0], axis=0)

    if dim_other > 0:
        grid_other = np.zeros((grid_points.shape[0], dim_other), dtype=np.float32)
        grid_points = np.concatenate([grid_points, grid_other], axis=-1)
        goals_all = np.concatenate([goals_all, grid_other], axis=-1)

    def value_fn(pts, gs):
        rep = agent.network.select("rep_value")(gs)
        v1, v2 = agent.network.select("value")(pts, rep)
        return (v1 + v2) / 2.0
    
    def value_sum_fn(pts, gs):
        return value_fn(pts, gs).sum()

    V = value_fn(grid_points, goals_all)
    V = V.reshape(X.shape)

    # --- 2. Compute Uncertainty ---
    print(f"Computing Uncertainty")
    value_grad_fn = jax.jit(jax.grad(value_sum_fn, argnums=0))
    
    traj_pts = trajectory.astype(np.float32)
    goals_traj = np.repeat(goals[:1], len(trajectory), axis=0)
    
    if dim_other > 0:
        padding = np.zeros((len(trajectory), dim_other), dtype=np.float32)
        traj_pts = np.concatenate([traj_pts, padding], axis=-1)
        goals_traj = np.concatenate([goals_traj, padding], axis=-1)
        
    grads = value_grad_fn(traj_pts, goals_traj)
    grads_xy = grads[:, :2]

    # Polygons
    n_angles = 72
    angles = np.linspace(0, 2*np.pi, n_angles)
    circle_actions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    
    polygons = []
    entropies = []
    
    # VISUAL TWEAK: Reduced scale to avoid 'blobs' dominating the plot
    fan_scale = maze_unit * 0.4 
    
    step_size = max(1, len(trajectory) // 35)
    
    for t in range(0, len(trajectory), step_size):
        pos = trajectory[t]
        grad = grads_xy[t]
        
        advantages = circle_actions @ grad 
        logits = beta * advantages
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

        visual_radii = (probs / probs.max()) * fan_scale
        
        poly_points = []
        for i, angle in enumerate(angles):
            r = visual_radii[i]
            px = pos[0] + r * np.cos(angle)
            py = pos[1] + r * np.sin(angle)
            poly_points.append([px, py])
        
        polygons.append(patches.Polygon(poly_points, closed=True))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # A. Value Contours
    # vmin/vmax fixed to ensure consistent background between runs
    vmin, vmax = -72, 0 
    contour = ax.contourf(X, Y, V, levels=grid_size, cmap="Blues_r", zorder=0, vmin=vmin, vmax=vmax, alpha=0.9)
    ax.contour(X, Y, V, levels=20, colors="k", linewidths=0.5, alpha=0.5, zorder=1)
    
    # B. Maze Walls
    plot_maze_on_ax(env, ax)
    
    # C. Trajectory (Dark Black Dotted)
    # High zorder to be on top of everything
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], 
        color='black', linestyle=':', linewidth=2.5, 
        alpha=0.9, zorder=5, label="Trajectory"
    )
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', s=100, color='black', zorder=5) # Mark end
    
    # D. Uncertainty Fans (RdPu)
    entropies = np.array(entropies)
    if len(entropies) > 0:
        norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-6)
    else:
        norm_entropy = entropies

    # VISUAL TWEAK: High transparency (alpha=0.5) to see contours underneath
    p = PatchCollection(polygons, cmap="RdPu", alpha=0.5, zorder=4)
    p.set_array(norm_entropy) 
    ax.add_collection(p)
    
    # Colorbars
    cbar1 = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="Value (Blue=Low, White=High)")
    cbar2 = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04, location='left', label="Action Entropy (Dark=unsure)")
    
    ax.set_title(f"Uncertainty in action selection over Value function")
    ax.set_aspect('equal')
    
    save_path = Path("plots") / f"{save_name}_beta{beta}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Vis] Saved plot to {save_path}")

def main(_):
    # ----- Set up environment and dataset -----
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config["frame_stack"], dataset_dir="tmp/data/"
    )
    env.reset()
    if "oraclerep" in FLAGS.env_name and config["oraclerep"] == False:
        raise ValueError(
            "Must enable oracle representation in config dictionary to use this environment!"
        )

    if "speed_profile" in config:
        env_u = env.unwrapped
        S = env_u._maze_unit
        obstacle_coordinates = generate_obstacle_coordinates(env_u, S, resolution=0.1)

        if config["speed_profile"] == "linear":
            speed_train, speed_min = compute_speed_profile(
                train_dataset["observations"], obstacle_coordinates
            )
            speed_val, _ = compute_speed_profile(
                val_dataset["observations"], obstacle_coordinates
            )

        elif config["speed_profile"] == "exponential":
            speed_train, speed_min = compute_exponential_speed_profile(
                train_dataset["observations"], obstacle_coordinates
            )
            speed_val, _ = compute_exponential_speed_profile(
                val_dataset["observations"], obstacle_coordinates
            )

        elif config["speed_profile"] == "constant":
            speed_train = np.ones((train_dataset["observations"].shape[0],))
            speed_val = np.ones((val_dataset["observations"].shape[0],))

            speed_min = 0.1

        else:
            NotImplementedError

        train_dataset["speed"] = speed_train
        val_dataset["speed"] = speed_val
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    # debugging only:
    # obs_all = train_dataset["observations"]
    # obs_all = np.concatenate([obs_all, val_dataset["observations"]], axis=0)
    obs_all = None

    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "VIPDataset": VIPDataset,
    }[config["dataset_class"]]
    train_dataset = dataset_class(
        Dataset.create(norm=config["norm"], **train_dataset), config
    )

    # ----- Initialize agent -----
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )

    agent_class = agents[config["agent_name"]]
    ex_goals = example_batch["value_goals"] if config["oraclerep"] else None
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
        ex_goals=ex_goals,
    )

    # ----- Restore agent -----
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # ----- visualize value function -----
    vis_value_function(env, agent, obs_all)
    
    # # --- EXECUTE VISUALIZATION ---
    print("\n--- Generating Rollout ---")
    # Generate one clean rollout to [1,1]
    # trajectory = get_rollout(env, agent, goal_pos=[1, 1], temperature=0.0)
    
    # if len(trajectory) < 5:
    #     print("Trajectory too short/failed. Check Agent.")
    # else:
    #     # Generate overlays for different betas
    #     save_prefix = Path(FLAGS.restore_path).name
    #     # Loop betas to find the best looking one for the paper
    #     for beta in [0.5]:
    #         # vis_combined_heatmap_and_uncertainty(
    #         #     env, agent, trajectory, beta=beta, save_name=f"{save_prefix}-overlay"
    #         # )
    #         vis_action_uncertainty(
    #             env, agent, trajectory, beta=beta, save_name=f"{save_prefix}-uncertainty"
    #         )
            
    vis_single_perturbation(env, agent, goal_pos=[1, 1],  perturb_std=4.5, save_name="recovery_heatmap")

    print("Done.")


def plot_maze(env):
    env_u = env.unwrapped
    maze_map = env_u.maze_map
    maze_unit = env_u._maze_unit

    # Draw maze walls
    for i in range(0, maze_map.shape[0]):
        for j in range(0, maze_map.shape[1]):
            if maze_map[i, j] == 1:  # 1 indicates a wall
                wall = np.array([i, j])
                wall = wall * maze_unit - maze_unit * 1.5
                rect = patches.Rectangle(
                    wall,
                    maze_unit,
                    maze_unit,
                    # linewidth=1,
                    edgecolor=None,
                    facecolor="white",
                    alpha=1,
                    zorder=2,
                )
                plt.gca().add_patch(rect)


def create_meshgrid(env, grid_size=100):
    env_u = env.unwrapped
    maze_size = env_u.maze_map.shape[0]
    maze_unit = env_u._maze_unit
    maze_map = env_u.maze_map

    # Create a grid of positions
    h, w = env_u.maze_map.shape 

    # Calculate ranges independently
    range_min = -maze_unit * 1.5
    range_max_x = (w - 1) * maze_unit - maze_unit * 1.5 + maze_unit
    range_max_y = (h - 1) * maze_unit - maze_unit * 1.5 + maze_unit

    # Create linspaces that match the actual maze dimensions
    x = np.linspace(range_min, range_max_x, grid_size)
    y = np.linspace(range_min, range_max_y, grid_size)

    X, Y = np.meshgrid(x, y)

    # maze mask at corresponding grid points
    M = maze_map.repeat(maze_unit, axis=0).repeat(maze_unit, axis=1)
    X_idx = ((X - range_min).astype(int)).clip(0, M.shape[1] - 1)
    Y_idx = ((Y - range_min).astype(int)).astype(int).clip(0, M.shape[0] - 1)
    M = 1 - M[Y_idx, X_idx]
    return X, Y, M


def vis_value_function(env, agent, obs_all=None):
    """Visualize the value function on a 2D grid for pointmaze environments."""
    env_u = env.unwrapped
    maze_unit = env_u._maze_unit
    dim_other = env.observation_space.shape[0] - 2
    env_name = env.spec.id
    rng = jax.random.PRNGKey(0)
    agent_name = FLAGS.agent["agent_name"]

    VAL = True
    GRD = False
    LAP = False
    assert sum([VAL, GRD, LAP]) == 1, "Only one of VAL, GRD, LAP should be True."

    goals = [[1, 1]]  # maze cell idx
    goals = np.array(goals, dtype=np.float32) * maze_unit

    # # Create a grid of positions
    grid_size = 100
    X, Y, M = create_meshgrid(env, grid_size=grid_size)
    # Evaluate value function at each grid point
    with console.status("[bold green] Querying value function on grid"):
        # batched query
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
        goals_all = np.repeat(goals, grid_points.shape[0], axis=0)
        # print(grid_points.shape, goals.shape)
        if dim_other > 0:
            # create more dimensions with zeros
            grid_other = np.zeros((1, dim_other), dtype=np.float32)
            # grid_other += np.random.randn(*grid_other.shape) * 0.1  # debug: add small noise
            grid_other = np.repeat(grid_other, grid_points.shape[0], axis=0)
            grid_points = np.concatenate([grid_points, grid_other], axis=-1)
            goals_all = np.concatenate([goals_all, grid_other], axis=-1)

        def value_fn(pts, gs):
            print("AGENT NAME: ", agent_name)
            if agent_name == 'gcivl':
                v1, v2 = agent.network.select("value")(pts, gs)
            elif "vib" in agent_name:
                gr, kl_loss, kl_info = agent.network.select('vib')(
                gs, rng, encoded=False)
                v1, v2 = agent.network.select("value")(pts, gr)
            else:
                gr = agent.network.select("rep_value")(gs)
                v1, v2 = agent.network.select("value")(pts, gr)
            
            
            
            return (v1 + v2) / 2.0

        def value_sum_fn(*args):
            return value_fn(*args).sum()

        if VAL:
            # query value function
            V = value_fn(grid_points, goals_all)
            V = V.reshape(X.shape)
            # ranges for vf:
            # vmin, vmax = -75, -0.25
            vmin, vmax = -94, 0
            save_prefix = "vf"

        if GRD:
            # query gradient of value function
            value_grad_fn = jax.jit(jax.grad(value_sum_fn, argnums=0))
            V_grad = value_grad_fn(grid_points, goals_all)
            V = jnp.linalg.norm(V_grad, axis=-1)
            V = V.reshape(X.shape)
            # ranges for vf grad:
            # vmin, vmax = 0.004, 12.3
            vmin, vmax = 0, 11.5
            save_prefix = "vf_grad"

        if LAP:
            # query laplacian of value function
            value_hess_fn = jax.jit(jax.vmap(jax.hessian(value_sum_fn, argnums=0)))
            # batched (for reduced memory usage)
            V_Hess = []
            batch_size = 100
            for i in range(0, grid_points.shape[0], batch_size):
                pts = grid_points[i : i + batch_size]
                gs = goals_all[i : i + batch_size]
                V_hess_batch = value_hess_fn(pts, gs)
                V_Hess.append(V_hess_batch)
            V_hess = jnp.concatenate(V_Hess, axis=0)
            V = jnp.trace(V_hess, axis1=1, axis2=2)
            V = V.reshape(X.shape)
            # ranges for vf lap:
            vmin, vmax = -37, 26
            save_prefix = "vf_lap"

    # np.save(f"{Path(FLAGS.restore_path).name}-vf.npy", np.stack([X, Y, V], axis=-1))
    V_msk = V * M
    print(f"V (org) range: min {V.min():.3f}, max {V.max():.3f}")
    print(f"V (msk) range: min {V_msk.min():.3f}, max {V_msk.max():.3f}")

    # # Plot the value function
    fig, ax = plt.subplots(figsize=(8, 8))
    print(f"- Using value range for plotting: vmin={vmin}, vmax={vmax}")
    
    # 1. The Filled Background
    contour = ax.contourf(
        Y, X, V, levels=grid_size, cmap="Blues_r", zorder=0
    )
    plt.colorbar(contour, ax=ax, label="Value (Goal-conditioned)")
    
    
    if VAL:
        # 2. SOFT CONTOURS (The "tiny ruler ticks")
        # We use 4x the density (80 levels) but thin and semi-transparent
        ax.contour(Y, X, V, levels=80, colors="k", linewidths=0.4, alpha=0.3, zorder=1)

        # 3. HARD CONTOURS (The "giant ruler ticks")
        # Kept exactly as your original code
        ax.contour(Y, X, V, levels=20, colors=c, linewidths=1.4, alpha=1.0, zorder=1, linestyles=[(0, (2, 1))])

    #ax.plot(goals[0, 0], goals[0, 1], "ro", markersize=15)  # Plot goal position
    plot_maze(env)

    ## Debugging:
    if obs_all is not None:
        obs_all = obs_all[:, :2]  # only plot x,y positions
        # plot states from dataset
        ax.plot(
            obs_all[:, 0],
            obs_all[:, 1],
            "k.",
            markersize=1,
            alpha=0.1,
        )
    plt.savefig(f"plots/{Path(FLAGS.restore_path).name}-{save_prefix}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    app.run(main)
