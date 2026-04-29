import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ogbench
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset

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

    if "oraclerep" in FLAGS.env_name and config["oraclerep"] == False:
        raise ValueError(
            "Must enable oracle representation in config dictionary to use this environment!"
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

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    while True:
        batch = train_dataset.sample(config["batch_size"])
        print(batch.keys())
        print(batch["observations"].shape)
        print(np.unique(batch["terminals"]))
        # exit()
        is_term = (batch["terminals"] == 1).any()
        print(batch["actions"].shape)
        if not is_term:
            continue
        # feed the batch to the visualization function
        vis(env, batch, task_id=1)


def vis(env, batch, task_id=1):
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
