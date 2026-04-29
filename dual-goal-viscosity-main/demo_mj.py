import time
import random
from pathlib import Path
from turtle import width
from xml.parsers.expat import model

import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax
import ogbench
import mujoco
import mujoco.viewer
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset

from utils.flax_utils import restore_agent
from utils.evaluation import supply_rng
from utils.env_utils import FrameStackWrapper
from rich import print
from rich.console import Console

console = Console()

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "antmaze-large-navigate-v0", "Environment (dataset) name."
)
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0.0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_float("eval_goal_gaussian", None, "Goal Gaussian noise for evaluation.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

config_flags.DEFINE_config_file("agent", "agents/crl/id.py", lock_config=False)


def make_env_and_datasets(dataset_name, frame_stack=None, dataset_dir=None, **kwargs):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, dataset_dir=dataset_dir, **kwargs
    )
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()
    return env, train_dataset, val_dataset


def images_to_video(images, video_path, fps=30):
    import cv2

    height, width, _ = images[0].shape
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in images:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()
    print(f"Saved video to {video_path}")


def main(_):
    # ----- Set up environment and dataset -----
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config["frame_stack"],
        dataset_dir="tmp/data/",
        width=600,
        height=600,
    )

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

    # vcount = 0
    # while True:
    #     batch = train_dataset.sample(config["batch_size"])
    #     print(batch.keys())
    #     print(batch["observations"].shape)
    #     print(np.unique(batch["terminals"]))
    #     # exit()
    #     is_term = (batch["terminals"] == 1).any()
    #     print(batch["actions"].shape)
    #     if not is_term:
    #         continue
    #     # feed the batch to the visualization function
    #     # vis(env, batch, agent, task_id=1)
    #     image = vis_agent_v2(env, agent, config, batch, task_id=1)
    #     # images_to_video(renders, f"videos/vis_agent_{vcount}.mp4")
    #     imageio.imwrite(f"videos/traj_{vcount}.jpg", image)
    #     vcount += 1
    #     if vcount == 500:
    #         break

    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    print(f"Evaluating on {num_tasks} tasks...")
    for task_id in tqdm.trange(1, num_tasks + 1):
        task_name = task_infos[task_id - 1]["task_name"]
        image, info = vis_agent_v3(env, agent, config, task_id=task_id)
        imageio.imwrite(
            f"videos-visc/task_{task_id}_{task_name}_success{info['success']}.jpg",
            image,
        )


def vis_agent_v3(env, agent, config, task_id=1):
    # task-based
    model = env.unwrapped.model
    data = env.unwrapped.data
    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )

    width, height = 600, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    # sim
    observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get("goal")
    done = False
    step = 0

    # trajectory
    ee_trajectory = []
    ee_id = model.body("ur5e/robotiq/base").id

    # renders = []
    while not done:
        step_start = time.time()
        action = actor_fn(observations=observation, goals=goal, temperature=0)
        action = np.array(action)
        if not config.get("discrete"):
            action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("************** Done **************")
        step += 1

        # Record Trajectory
        ee_pos = data.body(ee_id).xpos.copy()
        ee_orn = data.body(ee_id).xmat.copy().reshape(3, 3)
        # offset point along approach vector
        ee_pos = ee_pos + 0.1 * ee_orn[:, 2]

        if (
            len(ee_trajectory) == 0
            or np.linalg.norm(ee_pos - ee_trajectory[-1]) > 0.005
        ):
            ee_trajectory.append(ee_pos)

        # frame = env.render()
        # renders.append(frame)
        observation = next_observation

    # Update the scene with the *current* physics state
    renderer.update_scene(data)
    scene = renderer.scene

    # Iterate through stored points and draw lines between them
    for i in range(len(ee_trajectory) - 1):
        # Check if we have space in the geom buffer
        if scene.ngeom >= scene.maxgeom:
            print("Scene geom buffer full!")
            break

        # Clean/Initialize the geom slot with defaults
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),  # size placeholder
            np.zeros(3),  # pos placeholder
            np.zeros(9),  # mat placeholder
            np.array([1.0, 0.0, 0.0, 1.0]).astype(np.float32),  # RGBA
        )

        # Initialize a new line geom in the scene
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],  # The geom object to modify
            mujoco.mjtGeom.mjGEOM_LINE,  # Type
            2.0,  # Width (in pixels)
            ee_trajectory[i],  # Start Point (3D)
            ee_trajectory[i + 1],  # End Point (3D)
        )

        # color (R, G, B, A)
        scene.geoms[scene.ngeom].rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        scene.ngeom += 1

    image = renderer.render()
    return image, info


def vis_agent_v2(env, agent, config, batch, task_id=1):
    model = env.unwrapped.model
    data = env.unwrapped.data
    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )

    width, height = 600, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    # sim
    observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get("goal")
    goal_frame = info.get("goal_rendered")
    done = False
    step = 0

    # trajectory
    ee_trajectory = []
    ee_id = model.body("ur5e/robotiq/base").id

    # renders = []
    while not done:
        step_start = time.time()
        action = actor_fn(observations=observation, goals=goal, temperature=0)
        action = np.array(action)
        if not config.get("discrete"):
            action = np.clip(action, -1, 1)

        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("************** Done **************")
        step += 1
        # time.sleep(0.05)

        # Record Trajectory
        ee_pos = data.body(ee_id).xpos.copy()
        ee_orn = data.body(ee_id).xmat.copy().reshape(3, 3)
        # offset point along approach vector
        ee_pos = ee_pos + 0.1 * ee_orn[:, 2]

        if (
            len(ee_trajectory) == 0
            or np.linalg.norm(ee_pos - ee_trajectory[-1]) > 0.005
        ):
            ee_trajectory.append(ee_pos)

        # frame = env.render()
        # renders.append(frame)
        observation = next_observation

    # Update the scene with the *current* physics state
    renderer.update_scene(data)
    scene = renderer.scene

    # Iterate through stored points and draw lines between them
    for i in range(len(ee_trajectory) - 1):
        # Check if we have space in the geom buffer
        if scene.ngeom >= scene.maxgeom:
            print("Scene geom buffer full!")
            break

        # Clean/Initialize the geom slot with defaults
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),  # size placeholder
            np.zeros(3),  # pos placeholder
            np.zeros(9),  # mat placeholder
            np.array([1.0, 0.0, 0.0, 1.0]).astype(np.float32),  # RGBA
        )

        # Initialize a new line geom in the scene
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],  # The geom object to modify
            mujoco.mjtGeom.mjGEOM_LINE,  # Type
            2.0,  # Width (in pixels)
            ee_trajectory[i],  # Start Point (3D)
            ee_trajectory[i + 1],  # End Point (3D)
        )

        # color (R, G, B, A)
        scene.geoms[scene.ngeom].rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        scene.ngeom += 1

    image = renderer.render()
    return image


def vis_agent(env, agent, config, batch, task_id=1):
    model = env.unwrapped.model
    data = env.unwrapped.data
    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )

    # sim
    observation, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get("goal")
    goal_frame = info.get("goal_rendered")
    done = False
    step = 0

    # ghost data
    ghost_data = mujoco.MjData(model)
    init_qpos = batch["observations"][0][: model.nq]
    ghost_data.qpos[:] = init_qpos
    mujoco.mj_forward(model, ghost_data)  # Compute forward kinematics for the ghost

    # trajectory
    ee_trajectory = []  # List to store (x, y, z) positions
    ee_id = model.body("ur5e/robotiq/base").id
    # UR5e End-Effector Body ID (adjust name if needed)

    # 4. Simulation Loop with Custom Visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure visual flags (optional: make scene prettier)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = (
            0  # Turn off shadows for clearer lines
        )
        renders = []
        while not done and viewer.is_running():
            step_start = time.time()
            action = actor_fn(observations=observation, goals=goal, temperature=0)
            action = np.array(action)
            if not config.get("discrete"):
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print("************** Done **************")
            step += 1
            # time.sleep(0.05)

            # --- B. Record Trajectory ---
            # Get current EE position. Copy it so it doesn't change with simulation.
            ee_pos = data.body(ee_id).xpos.copy()
            ee_orn = data.body(ee_id).xmat.copy().reshape(3, 3)
            # offset point along approach vector
            ee_pos = ee_pos + 0.1 * ee_orn[:, 2]

            if (
                len(ee_trajectory) == 0
                or np.linalg.norm(ee_pos - ee_trajectory[-1]) > 0.005
            ):
                ee_trajectory.append(ee_pos)

            # --- C. Custom Rendering ---
            # 1. Reset user scene geometries for this frame
            viewer.user_scn.ngeom = 0

            # 2. Render Trajectory Lines
            # We assume we can draw N-1 segments for N points
            for i in range(len(ee_trajectory) - 1):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=[3, 0, 0],  # Width of the line
                    pos=np.zeros(
                        3
                    ),  # Lines are defined by p1/p2, so pos is ignored or 0
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 1, 1],  # Cyan color
                )
                # Connect the two points
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    width=3,
                    from_=ee_trajectory[i],
                    to=ee_trajectory[i + 1],
                )
                viewer.user_scn.ngeom += 1

            # 3. Render "Ghost" Robot (Final State)
            # We iterate over all geoms in the model and reproduce them at the ghost_data positions
            for i in range(model.ngeom):
                # Skip non-visual geoms if necessary
                if model.geom_group[i] > 2:
                    continue

                # Get geometry position/rotation from the GHOST data (not live data)
                # Note: We must use the ghost_data we computed earlier
                geom_pos = ghost_data.geom_xpos[i]
                geom_mat = ghost_data.geom_xmat[i]

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=model.geom_type[i],
                    size=model.geom_size[i],
                    pos=geom_pos,
                    mat=geom_mat,
                    rgba=[0, 1, 0, 0.3],  # Green, semi-transparent (Alpha = 0.3)
                )
                viewer.user_scn.ngeom += 1

            # --- D. Sync to Viewer ---
            viewer.sync()

            frame = env.render()
            renders.append(frame)

            # Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            observation = next_observation
    return renders


def vis(env, batch, task_id=1):
    # ----- setting up for traj display -----
    model = env.unwrapped.model
    data = env.unwrapped.data

    # ghost data
    ghost_data = mujoco.MjData(model)
    # final_qpos = model.key_qpos[0] if model.nkey > 0 else np.zeros(model.nq)
    # final_qpos = batch["observations"][-1][: model.nq]
    init_qpos = batch["observations"][0][: model.nq]
    # print(final_qpos)
    ghost_data.qpos[:] = init_qpos
    mujoco.mj_forward(model, ghost_data)  # Compute forward kinematics for the ghost

    # trajectory
    ee_trajectory = []  # List to store (x, y, z) positions
    ee_id = model.body("ur5e/robotiq/base").id
    # UR5e End-Effector Body ID (adjust name if needed)

    idx = 0
    # 4. Simulation Loop with Custom Visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure visual flags (optional: make scene prettier)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = (
            0  # Turn off shadows for clearer lines
        )

        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # --- A. Step Physics ---
            # Replace with your policy/agent logic
            # action = env.action_space.sample()
            action = batch["actions"][idx]  # Use the action from the batch
            idx += 1
            # Manual stepping (bypass env.step to keep viewer control or use env.step and sync)
            # print(data.ctrl.shape, action.shape)
            # exit()
            data.ctrl[: action.shape[0]] = action
            mujoco.mj_step(model, data)

            # # set robot state (non-physical stepping)
            # qpos_dim = model.nq
            # qvel_dim = model.nv
            # if idx < batch["actions"].shape[0]:
            #     qpos = batch["observations"][idx][:qpos_dim]
            #     # qvel = batch["observations"][idx][qpos_dim : qpos_dim + qvel_dim]
            #     data.qpos[:] = qpos
            #     # data.qvel[:] = qvel
            #     mujoco.mj_forward(model, data)  # Update the simulation state
            #     idx += 1
            # else:
            #     # reached the end of the batch
            #     pass
            # # add time delay
            # time.sleep(0.05)

            # --- B. Record Trajectory ---
            # Get current EE position. Copy it so it doesn't change with simulation.
            ee_pos = data.body(ee_id).xpos.copy()
            ee_orn = data.body(ee_id).xmat.copy().reshape(3, 3)
            # offset point along approach vector
            ee_pos = ee_pos + 0.1 * ee_orn[:, 2]

            if (
                len(ee_trajectory) == 0
                or np.linalg.norm(ee_pos - ee_trajectory[-1]) > 0.005
            ):
                ee_trajectory.append(ee_pos)

            # --- C. Custom Rendering ---
            # 1. Reset user scene geometries for this frame
            viewer.user_scn.ngeom = 0

            # 2. Render Trajectory Lines
            # We assume we can draw N-1 segments for N points
            for i in range(len(ee_trajectory) - 1):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=[3, 0, 0],  # Width of the line
                    pos=np.zeros(
                        3
                    ),  # Lines are defined by p1/p2, so pos is ignored or 0
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 1, 1],  # Cyan color
                )
                # Connect the two points
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    width=3,
                    from_=ee_trajectory[i],
                    to=ee_trajectory[i + 1],
                )
                viewer.user_scn.ngeom += 1

            # 3. Render "Ghost" Robot (Final State)
            # We iterate over all geoms in the model and reproduce them at the ghost_data positions
            for i in range(model.ngeom):
                # Skip non-visual geoms if necessary
                if model.geom_group[i] > 2:
                    continue

                # Get geometry position/rotation from the GHOST data (not live data)
                # Note: We must use the ghost_data we computed earlier
                geom_pos = ghost_data.geom_xpos[i]
                geom_mat = ghost_data.geom_xmat[i]

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=model.geom_type[i],
                    size=model.geom_size[i],
                    pos=geom_pos,
                    mat=geom_mat,
                    rgba=[0, 1, 0, 0.3],  # Green, semi-transparent (Alpha = 0.3)
                )
                viewer.user_scn.ngeom += 1

            # --- D. Sync to Viewer ---
            viewer.sync()

            # Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            if idx >= batch["actions"].shape[0]:
                print("************** Done **************")
                break
    # exit(0)
    return

    # ----- vis setup -----
    plt.ion()  # Turn on interactive mode
    fig, (ax_current, ax_goal) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("OGBench Environment Visualization", fontsize=14)
    ax_current.set_title("Current State")
    ax_goal.set_title("Goal State")
    ax_current.axis("off")
    ax_goal.axis("off")

    ob, info = env.reset(
        options=dict(
            task_id=task_id,  # Set the evaluation task. Each environment provides five
            # evaluation goals, and `task_id` must be in [1, 5].
            render_goal=True,  # Set to `True` to get a rendered goal image (optional).
        )
    )

    goal = info["goal"]  # Get the goal observation to pass to the agent.
    goal_rendered = info["goal_rendered"]  # Get the rendered goal image (optional).

    # Display the goal image
    if goal_rendered is not None:
        ax_goal.clear()
        ax_goal.imshow(goal_rendered)
        ax_goal.set_title(f"Goal State")
        ax_goal.axis("off")

    done = False
    step_count = 0
    while True:  # not done:
        # action = env.action_space.sample()  # Replace this with your agent's action.
        action = batch["actions"][step_count]  # Use the action from the batch
        ob, reward, terminated, truncated, info = env.step(
            action
        )  # Gymnasium-style step.
        # If the agent reaches the goal, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
        # `reward` is 1 if the agent reaches the goal and 0 otherwise.
        done = terminated or truncated
        frame = env.render()  # Render the current frame (optional).

        # Update the visualization
        ax_current.clear()
        ax_current.imshow(frame)
        ax_current.set_title(f"Current State (Step {step_count}, Reward: {reward})")
        ax_current.axis("off")

        plt.pause(0.01)  # Small pause to update the display

        step_count += 1
        if step_count % 10 == 0:
            print(
                f"- Step {step_count}: Action: {action.shape}, "
                f"Reward: {reward}, Done: {done}, Frame: {frame.shape}"
            )

        if done or step_count >= batch["actions"].shape[0]:
            break

    success = info["success"]  # Whether the agent reached the goal (0 or 1).
    # `terminated` also indicates this.
    print(f"Task completed. Success: {success}\n")
    env.close()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final window open


if __name__ == "__main__":
    app.run(main)
