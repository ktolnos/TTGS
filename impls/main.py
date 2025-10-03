import json
import os
import traceback

os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ''
import random
import time
import jax
import numpy as np
import tqdm
import wandb
import matplotlib.pyplot as plt
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from ttgs import TTGS
from eval_utils import EvalParams, run_taskwise_evaluation, plot_success_comparison
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
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 0, 'Whether to evaluate on CPU.')
flags.DEFINE_boolean('eval_at_all', True, 'Whether to evaluate during training.')
flags.DEFINE_boolean('train_actor_only', False, 'Whether train only actor.')
flags.DEFINE_boolean('train_dynamics_obs_only', False, 'Whether train only dynamics model in observation space.')

flags.DEFINE_integer('te_horizon', 24, 'TE horizon.')
flags.DEFINE_integer('tau', 12, 'Max dist tau when building graph.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for ttgs evaluation.')
flags.DEFINE_float('error', 0.001, 'Error for ttgs evaluation.')
flags.DEFINE_integer('threshold', 24, 'Threshold for ttgs evaluation.')
flags.DEFINE_string('subsample_ablt', 'default', 'Ablation mode for ttgs evaluation.')
flags.DEFINE_string('subgoal_ablt', 'default', 'Ablation mode for ttgs evaluation.')
flags.DEFINE_boolean('add_dataset_trajectories', False, 'Whether to add dataset trajectories to the graph.')
flags.DEFINE_integer('random_size', 5000, 'Size of random points for random points subsampling.')
flags.DEFINE_string('dist_mode', 'value', 'Distance mode for ttgs evaluation.')
flags.DEFINE_string('penalty_mode', 'dynamic', 'Penalty mode for ttgs evaluation.')
flags.DEFINE_float('penalty_factor', 1000.0, 'Penalty factor for ttgs evaluation.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)

jax_device = jax.devices('gpu')[0]
jax.config.update("jax_default_device", jax_device)


def main(_):
    # Track overall experiment time
    exp_start_time = time.time()
    timing_log = {}
    
    # Set up logger.
    setup_start_time = time.time()
    exp_name = get_exp_name(FLAGS.seed)
    # Create TTGS-specific wandb name: ttgs_envname_agentname_seed
    if FLAGS.subsample_ablt == 'random_points':
        ttgs_name = f"ttgs_{FLAGS.agent.agent_name}_{FLAGS.env_name}_seed{FLAGS.seed:03d}_subsampablt{FLAGS.subsample_ablt}_randomsize{FLAGS.random_size}_subgoalablt{FLAGS.subgoal_ablt}_addtraj{FLAGS.add_dataset_trajectories}_penaltymode{FLAGS.penalty_mode}_penaltyfactor{FLAGS.penalty_factor}"
    else:
        ttgs_name = f"ttgs_{FLAGS.agent.agent_name}_{FLAGS.env_name}_seed{FLAGS.seed:03d}_subsampablt{FLAGS.subsample_ablt}_subgoalablt{FLAGS.subgoal_ablt}_addtraj{FLAGS.add_dataset_trajectories}_penaltymode{FLAGS.penalty_mode}_penaltyfactor{FLAGS.penalty_factor}"
    setup_wandb(project='TTGS', group=FLAGS.run_group, name=ttgs_name)
    timing_log['setup_time'] = time.time() - setup_start_time

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    # Log ttgs.py file to wandb
    ttgs_path = os.path.join(os.path.dirname(__file__), 'ttgs.py')
    if os.path.exists(ttgs_path):
        wandb.save(ttgs_path, base_path=os.path.dirname(ttgs_path))

    # Set up environment and dataset.
    data_start_time = time.time()
    config = FLAGS.agent
    env, train_data, val_data = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    timing_log['data_loading_time'] = time.time() - data_start_time

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_data), config)
    if val_data is not None:
        val_dataset = dataset_class(Dataset.create(**val_data), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_start_time = time.time()
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    key = jax.random.PRNGKey(FLAGS.seed)
    start_index = 1 if FLAGS.restore_epoch is None else FLAGS.restore_epoch
    skip_training =  FLAGS.restore_epoch is not None and FLAGS.restore_epoch >= FLAGS.train_steps
    for i in tqdm.tqdm(range(start_index, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        if not skip_training:
            if FLAGS.train_actor_only:
                agent, update_info = agent.actor_update(batch)
            elif FLAGS.train_dynamics_obs_only:
                agent, update_info = agent.dynamics_obs_space_update(batch)
            else:
                agent, update_info = agent.update(batch)

            if i == FLAGS.train_steps:
                timing_log['training_time'] = time.time() - train_start_time

            # Log metrics.
            if i % FLAGS.log_interval == 0:
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config['batch_size'])
                    if FLAGS.train_actor_only:
                        _, val_info = agent.actor_loss(val_batch, grad_params=None)
                    elif FLAGS.train_dynamics_obs_only:
                        _, val_info = agent.dynamics_obs_space_loss(val_batch, grad_params=None)
                    else:
                        _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                wandb.log(train_metrics, step=i)
                train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        ####### Evaluate #######
        if FLAGS.eval_at_all and i == FLAGS.train_steps:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent

            ####### Original Agent Evaluation #######
            eval_start_time = time.time()
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

            std_params = EvalParams(
                agent=eval_agent,
                env=env,
                seed=FLAGS.seed,
                num_tasks=num_tasks,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
                config=config,
                mode='standard',
            )
            eval_task_metrics, eval_overall, renders = run_taskwise_evaluation(std_params)

            eval_metrics = {}
            eval_metrics.update(eval_task_metrics)

            for k, v in eval_overall.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics)
            eval_logger.log(eval_metrics, step=i)
            
            timing_log['standard_eval_time'] = time.time() - eval_start_time
            
            ####### TTGS Evaluation - Run after checkpoint save #######
            if i == FLAGS.train_steps:
                try:
                    ttgs_start_time = time.time()
                    eval_ttgs = TTGS(eval_agent, train_dataset=train_data, dist_mode=FLAGS.dist_mode, tau=FLAGS.tau, te_horizon=FLAGS.te_horizon, batch_size=FLAGS.batch_size, error=FLAGS.error, threshold=FLAGS.threshold, subsample_ablt=FLAGS.subsample_ablt, random_size=FLAGS.random_size, subgoal_ablt=FLAGS.subgoal_ablt, ablt_seed=FLAGS.seed, penalty_mode=FLAGS.penalty_mode, penalty_factor=FLAGS.penalty_factor)
                    # Log ttgs parameters
                    ttgs_params = {
                        'ttgs/te_horizon': eval_ttgs.te_horizon,
                        'ttgs/tau': eval_ttgs.tau,
                        'ttgs/batch_size': eval_ttgs.batch_size,
                        'ttgs/error': eval_ttgs.error,
                        'ttgs/threshold': eval_ttgs.threshold,
                        'ttgs/subsample_ablt': eval_ttgs.subsample_ablt,
                        'ttgs/subgoal_ablt': eval_ttgs.subgoal_ablt,
                        'ttgs/random_size': eval_ttgs.random_size,
                        'ttgs/dist_mode': eval_ttgs.dist_mode,
                        'ttgs/penalty_mode': eval_ttgs.penalty_mode,
                        'ttgs/penalty_factor': eval_ttgs.penalty_factor,
                    }
                    wandb.log(ttgs_params)

                    # Run ttgs evaluation across tasks (graph is built inside)
                    ttgs_eval_start_time = time.time()
                    ttgs_params_eval = EvalParams(
                        agent=eval_ttgs,
                        env=env,
                        seed=FLAGS.seed,
                        train_data=train_data,
                        num_tasks=num_tasks,
                        num_eval_episodes=FLAGS.eval_episodes,
                        mode='ttgs',
                        add_dataset_trajectories=FLAGS.add_dataset_trajectories,
                    )
                    ttgs_task_metrics, ttgs_overall, _ = run_taskwise_evaluation(ttgs_params_eval)

                    # Create comparison plot if available
                    if 'evaluation/overall_success' in eval_metrics:
                        base_success = eval_metrics.get('evaluation/overall_success', 0.0)
                        ttgs_success = ttgs_overall.get('overall_success', 0.0)
                        fig = plot_success_comparison(
                            base_success,
                            ttgs_success,
                            title=f'Overall Task Success Rate Comparison\n{FLAGS.env_name} - Training Step {i} - {FLAGS.agent.agent_name}',
                        )
                        plot_path = os.path.join(FLAGS.save_dir, f'comparison_step{i}.png')
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        wandb.log({'success_rate_comparison': wandb.Image(fig)})
                        plt.close(fig)

                    # Record TTGS timing components
                    timing_log['ttgs_total_time'] = time.time() - ttgs_start_time

                    wandb.log(ttgs_task_metrics)
                    wandb.log({f'ttgs_evaluation/{k}': v for k, v in ttgs_overall.items()})
                    
                except Exception as e:
                    print(f"=======TTGS Evaluation Failed: {e}=======")
                    traceback.print_exc()
                    print(f"Checkpoint was already saved. You can run ttgs evaluation later.")
                    wandb.log({'ttgs_evaluation/failed': 1, 'ttgs_evaluation/error': str(e)})
                    raise e

        # Save agent
        if i % FLAGS.save_interval == 0:
            print(f"=======Saving checkpoint at step {i} - {FLAGS.agent.agent_name}=======")
            save_agent(agent, FLAGS.save_dir, i)

    csv_file_hparams = 'compare_results.csv'
    
    # Check if we have both base and TTGS success rates to compare
    base_success_rate = eval_metrics.get('evaluation/overall_success', 0.0) if 'eval_metrics' in locals() else 0.0
    ttgs_success_rate = ttgs_overall.get('overall_success', 0.0) if 'ttgs_overall' in locals() else 0.0
    
    # Extract environment name
    env_name_clean = FLAGS.env_name.replace('-v0', '')
    agent_name = FLAGS.agent.agent_name
    seed = FLAGS.seed
    
    # Extended line with hyperparameters
    csv_line_hparams = f"{env_name_clean},{agent_name},{base_success_rate:.3f},{ttgs_success_rate:.3f},{seed},{FLAGS.te_horizon},{FLAGS.tau},{FLAGS.threshold},{FLAGS.error},{FLAGS.subsample_ablt},{FLAGS.random_size},{FLAGS.subgoal_ablt},{FLAGS.add_dataset_trajectories},{FLAGS.dist_mode},{FLAGS.penalty_mode},{FLAGS.penalty_factor}"

    # Write extended CSV (separate file to keep original format stable)
    file_exists_h = os.path.exists(csv_file_hparams)
    with open(csv_file_hparams, 'a', newline='') as f:
        if not file_exists_h:
            f.write("Environment,Base Agent,Base Success Rate,TTGS Success Rate,Seed,TE_Horizon,Tau,Threshold,Error,Subsample_Ablation_Mode,Random_Size,Subgoal_Ablation_Mode,Add_Dataset_Trajectories,Dist_Mode,Penalty_Mode,Penalty_Factor\n")
        f.write(csv_line_hparams + "\n")
    
    timing_log['total_experiment_time'] = time.time() - exp_start_time
    
    wandb.log({
        'timing/setup_time_minutes': timing_log.get('setup_time', 0) / 60,
        'timing/data_loading_time_minutes': timing_log.get('data_loading_time', 0) / 60,
        'timing/training_time_hours': timing_log.get('training_time', 0) / 3600,
        'timing/standard_eval_time_minutes': timing_log.get('standard_eval_time', 0) / 60,
        'timing/ttgs_total_time_minutes': timing_log.get('ttgs_total_time', 0) / 60,
        'timing/total_experiment_time_hours': timing_log.get('total_experiment_time', 0) / 3600,
    })
    
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
