import json
import os
import os
# Disable aggressive Triton GEMM fusions to prevent XLA compiler crashes
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
import minari
import gymnasium as gym

from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Minari_GCRL', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'D4RL/kitchen/mixed-v2', 'Minari Dataset ID / Environment name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0.0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_float('eval_goal_gaussian', None, 'Goal Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 0, 'Whether to evaluate on CPU.')
flags.DEFINE_integer('verbose', 0, 'Verbosity level.')

config_flags.DEFINE_config_file('agent', 'agents/crl/id.py', lock_config=False)


class OGBenchGoalWrapper(gym.Wrapper):
    """
    Wraps a standard Gym/Minari environment to supply goals via info['goal']
    and compute evaluation success metrics for OGBench, while scrubbing strings.
    """
    def __init__(self, env, eval_goals, distance_threshold=0.5):
        super().__init__(env)
        self.eval_goals = eval_goals
        self.distance_threshold = distance_threshold
        self.current_goal = None

    def reset(self, *, seed=None, options=None):
        options = options or {}
        obs, orig_info = self.env.reset(seed=seed, options=options)

        # Handle Minari dictionary observations (flatten them SAFELY)
        if isinstance(obs, dict):
            if 'observation' in obs:
                obs = obs['observation']
            else:
                obs = np.concatenate([np.atleast_1d(v) for v in obs.values()], axis=-1)
        
        # Sample a goal from our evaluation goal pool
        idx = np.random.randint(len(self.eval_goals))
        self.current_goal = self.eval_goals[idx]

        # Scrub non-numeric data from info to prevent np.mean() string crashes
        info = {k: float(v) for k, v in orig_info.items() if isinstance(v, (int, float, bool, np.number))}
        
        # Inject the goal array (OGBench handles the 'goal' key separately before taking the mean)
        info['goal'] = self.current_goal
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, orig_info = self.env.step(action)

        # Flatten observation safely
        if isinstance(obs, dict):
            if 'observation' in obs:
                obs = obs['observation']
            else:
                obs = np.concatenate([np.atleast_1d(v) for v in obs.values()], axis=-1)

        # Compute success metric for OGBench stats logging
        dist = np.linalg.norm(obs - self.current_goal)
        is_success = float(dist < self.distance_threshold)
        
        # Scrub non-numeric data from info
        info = {k: float(v) for k, v in orig_info.items() if isinstance(v, (int, float, bool, np.number))}
        
        # Add our custom GCRL metrics
        info['success'] = is_success
        info['distance_to_goal'] = float(dist)

        return obs, reward, terminated, truncated, info


def convert_minari_to_ogbench(minari_ds):
    """Adapter to flatten Minari episodes into OGBench transition arrays."""
    obs_list, actions_list, terminals_list, rewards_list = [], [], [], []
    
    for episode in minari_ds.iterate_episodes():
        # Handle Minari dictionary observations
        if isinstance(episode.observations, dict):
            if 'observation' in episode.observations:
                obs_array = episode.observations['observation']
            else:
                # Fallback: flatten all keys into a single array
                obs_array = np.concatenate(list(episode.observations.values()), axis=-1)
        else:
            obs_array = episode.observations
            
        # Slice the N+1 numpy array to match the N actions
        obs_list.append(obs_array[:-1])
        actions_list.append(episode.actions)
        
        # OGBench GCDataset REQUIRES trajectory boundaries. Force last to 1.0.
        terms = np.zeros(len(episode.actions), dtype=np.float32)
        terms[-1] = 1.0 
        terminals_list.append(terms)
        
        rewards_list.append(episode.rewards)
        
    return {
        'observations': np.concatenate(obs_list, axis=0).astype(np.float32),
        'actions': np.concatenate(actions_list, axis=0).astype(np.float32),
        'terminals': np.concatenate(terminals_list, axis=0).astype(np.float32),
        'rewards': np.concatenate(rewards_list, axis=0).astype(np.float32)
    }


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setting = str(FLAGS.agent['agent_name']).split('/')[-1].split('.')[0]
    enable_fk = FLAGS.agent.get('enable_fk_regularization', False)
    fk_key = 'fk_' if enable_fk else 'no_fk_'
    
    use_viscous_metric = FLAGS.agent.get('enable_viscous_metric', False)
    num_walks = FLAGS.agent.get('num_walks', 10)
    viscous_scale = FLAGS.agent.get('viscous_scale', 0.001)
    
    if FLAGS.verbose == 1:
        fk_key += f'viscous_{use_viscous_metric}_nwalks_{num_walks}_nuscale_{viscous_scale}_'
        setting = 'verbose_' + setting
    
    setup_wandb(project='goal_representation', group=FLAGS.run_group, name=setting+'_'+fk_key+FLAGS.env_name + '_' + exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, setting+'_'+fk_key+FLAGS.env_name + '_' + exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    # --- MINARI DATASET & ENV SETUP ---
    print(f"Loading Minari dataset: {FLAGS.env_name}...")
    minari_dataset = minari.load_dataset(FLAGS.env_name, download=True)
    
    env = minari_dataset.recover_environment(render_mode="rgb_array")
    env.reset()
    
    # Monkey-patch task_infos to provide a default evaluation goal routing
    if not hasattr(env.unwrapped, 'task_infos'):
        env.unwrapped.task_infos = [{'task_name': FLAGS.env_name, 'goal': None}]
        
    print("Converting Minari dataset to OGBench format...")
    full_dataset_dict = convert_minari_to_ogbench(minari_dataset)
    
    # --- Trajectory-Aware 90/10 Train/Validation Split ---
    total_transitions = len(full_dataset_dict['observations'])
    target_split_idx = int(total_transitions * 0.9)
    terminal_indices = np.nonzero(full_dataset_dict['terminals'] > 0)[0]
    valid_split_idx = terminal_indices[terminal_indices < target_split_idx][-1] + 1

    train_data_dict = {k: v[:valid_split_idx] for k, v in full_dataset_dict.items()}
    val_data_dict = {k: v[valid_split_idx:] for k, v in full_dataset_dict.items()}
    
    # --- APPLY OGBENCH WRAPPER ---
    val_terminals = np.nonzero(val_data_dict['terminals'] > 0)[0]
    eval_goals_pool = val_data_dict['observations'][val_terminals]
    
    env = OGBenchGoalWrapper(env, eval_goals=eval_goals_pool)
    # ----------------------------------

    if 'oraclerep' in FLAGS.env_name and config.get('oraclerep', False) == False:
        raise ValueError('Must enable oracle representation in config dictionary to use this environment!')
    
    # --- VISCOUS REGULARIZATION SPEED PROFILE ---
    if 'speed_profile' in config:
        if config['speed_profile'] == 'constant':
            speed_train = np.ones((train_data_dict['observations'].shape[0],))
            speed_val = np.ones((val_data_dict['observations'].shape[0],))
        else:
            raise NotImplementedError("Only 'constant' speed profile is supported for Minari GCRL setups.")

        train_data_dict['speed'] = speed_train
        val_data_dict['speed'] = speed_val

    # Initialize Base Datasets
    train_base_dataset = Dataset.create(**train_data_dict)
    val_base_dataset = Dataset.create(**val_data_dict)

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'VIPDataset': VIPDataset,
    }[config['dataset_class']]
    
    # Wrap in GCDataset for HER
    train_dataset = dataset_class(Dataset.create(norm=config.get('norm', False), **train_data_dict), config)
    val_dataset = dataset_class(Dataset.create(norm=config.get('norm', False), **val_data_dict), config)
    
    diff = train_dataset.get_diff()

    # Initialize agent
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config.get('discrete', False):
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    ex_goals = example_batch['value_goals'] if config.get('oraclerep', False) else None
    
    print("Initializing Agent...")
    agent = agent_class.create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config, ex_goals=ex_goals
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    print("Starting Training Loop...")
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    
    for i in tqdm.tqdm(range(0, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        
        batch = train_dataset.sample(config['batch_size'])
        batch['global_step'] = i
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                val_batch['global_step'] = i
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    eval_goal_gaussian=FLAGS.eval_goal_gaussian,
                    diff=diff,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0 and len(renders) > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    app.run(main)