import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.env_utils import generate_obstacle_coordinates, compute_speed_profile, compute_exponential_speed_profile
import ogbench
FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
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


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setting = str(FLAGS.agent['agent_name']).split('/')[-1].split('.')[0]
    enable_fk = FLAGS.agent['enable_fk_regularization']
    if(enable_fk):
        fk_key = 'fk_'
    else:
        fk_key = 'no_fk_'
    use_viscous_metric = FLAGS.agent.get('enable_viscous_metric', False)
    num_walks = FLAGS.agent.get('num_walks', 10)
    viscous_scale = FLAGS.agent.get('viscous_scale', 0.001)
    if(FLAGS.verbose==1):
        fk_key += f'viscous_{use_viscous_metric}_nwalks_{num_walks}_nuscale_{viscous_scale}_'
        setting = 'verbose_'+setting
    
    #print(fk_key)
    
    setup_wandb(project='goal_representation', group=FLAGS.run_group, name=setting+'_'+fk_key+FLAGS.env_name + '_' + exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, setting+'_'+fk_key+FLAGS.env_name + '_' + exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'], dataset_dir='/scratch/gautschi/hviswan/ogbench/data/')
    env.reset()
    if 'oraclerep' in FLAGS.env_name and config['oraclerep'] == False:
        raise ValueError('Must enable oracle representation in config dictionary to use this environment!')
    
    if 'speed_profile' in config:
        # env_u = env.unwrapped
        # S = env_u._maze_unit
        # obstacle_coordinates = generate_obstacle_coordinates(env_u, S, resolution=0.1)

        # if config['speed_profile'] == 'linear':        
        #     speed_train, speed_min = compute_speed_profile(train_dataset['observations'], obstacle_coordinates)
        #     speed_val, _ = compute_speed_profile(val_dataset['observations'], obstacle_coordinates)

        # elif config['speed_profile'] == 'exponential':
        #     speed_train, speed_min = compute_exponential_speed_profile(train_dataset['observations'], obstacle_coordinates)
        #     speed_val, _ = compute_exponential_speed_profile(val_dataset['observations'], obstacle_coordinates)

        if config['speed_profile'] == 'constant':
            speed_train = np.ones((train_dataset['observations'].shape[0],))
            speed_val = np.ones((val_dataset['observations'].shape[0],))

            speed_min = 0.1

        else:
            NotImplementedError

        train_dataset['speed'] = speed_train
        val_dataset['speed'] = speed_val
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'VIPDataset': VIPDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(norm=config['norm'], **train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(norm=config['norm'], **val_dataset), config)
    # Need to pass into evaluation functions
    diff = train_dataset.get_diff()

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    ex_goals = example_batch['value_goals'] if config['oraclerep'] else None
    agent = agent_class.create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config, ex_goals=ex_goals
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        batch['global_step'] = i
        agent, update_info = agent.update(batch)

        # Log metrics.
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

        # Evaluate agent.
        if i > 1 and i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
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

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
