import importlib.util
import json
import os
import random
import time
import traceback
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from agents import agents
from eval_utils import EvalParams, plot_success_comparison, run_taskwise_evaluation
from ttgs import TTGS
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "antmaze-large-navigate-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_integer("task_id", None, "Task id.")

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")
flags.DEFINE_boolean("eval_at_all", True, "Whether to evaluate during training.")
flags.DEFINE_boolean("train_actor_only", False, "Whether train only actor.")
flags.DEFINE_boolean(
    "train_dynamics_obs_only",
    False,
    "Whether train only dynamics model in observation space.",
)

flags.DEFINE_integer("te_horizon", 24, "TE horizon.")
flags.DEFINE_float("cluster_dist", 12, "Cluster distance.")
flags.DEFINE_integer("tau", 24, "Max dist tau when building graph.")
flags.DEFINE_integer("batch_size", 4, "Batch size for ttgs evaluation.")
flags.DEFINE_float("error", 0.001, "Error for ttgs evaluation.")
flags.DEFINE_integer("threshold", 48, "Threshold for ttgs evaluation.")
flags.DEFINE_string(
    "subsample_ablt", "random_points", "Ablation mode for ttgs evaluation."
)
flags.DEFINE_string("subgoal_ablt", "default", "Ablation mode for ttgs evaluation.")
flags.DEFINE_boolean(
    "add_dataset_trajectories",
    False,
    "Whether to add dataset trajectories to the graph.",
)
flags.DEFINE_integer(
    "random_size", 4000, "Size of random points for random points subsampling."
)
flags.DEFINE_integer("cluster_size", 4000, "The number of picked cluster centers.")
flags.DEFINE_string("dist_mode", "value", "Distance mode for ttgs evaluation.")
flags.DEFINE_string(
    "graph_mode", "full", "Graph construction mode for TTGS: full or knn."
)
flags.DEFINE_integer("knn_k", None, "Number of nearest neighbors when graph_mode=knn.")
flags.DEFINE_string("penalty_mode", "dynamic", "Penalty mode for ttgs evaluation.")
flags.DEFINE_float("penalty_factor", 1000.0, "Penalty factor for ttgs evaluation.")
flags.DEFINE_float(
    "replan_factor",
    0,
    "Replan when the TTGS-path distance exceeds replan_factor * threshold; <=0 disables replanning.",
)
flags.DEFINE_integer(
    "replan_cooldown",
    50,
    "Minimum number of environment steps between replans; <=0 disables cooldown.",
)
config_flags.DEFINE_config_file("agent", "agents/gciql.py", lock_config=False)

jax_device = jax.devices("gpu")[0]
jax.config.update("jax_default_device", jax_device)


def main(_):
    # Track overall experiment time
    exp_start_time = time.time()
    timing_log = {}

    # Set up logger.
    setup_start_time = time.time()
    exp_name = get_exp_name(FLAGS.seed)
    # Create TTGS-specific wandb name: ttgs_envname_agentname_seed
    ttgs_name = f"{FLAGS.agent.agent_name}_{FLAGS.env_name}_seed{FLAGS.seed:03d}"
    setup_wandb(project="TTGS", group=FLAGS.run_group, name=ttgs_name)
    timing_log["setup_time"] = time.time() - setup_start_time

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Log ttgs.py file to wandb
    ttgs_path = os.path.join(os.path.dirname(__file__), "ttgs.py")
    if os.path.exists(ttgs_path):
        wandb.save(ttgs_path, base_path=os.path.dirname(ttgs_path))

    # Set up environment and dataset.
    data_start_time = time.time()
    config = FLAGS.agent
    env, train_data, val_data = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config["frame_stack"]
    )
    timing_log["data_loading_time"] = time.time() - data_start_time

    # Use OTA-v datasets for the OTA agent to keep sampling logic consistent.
    use_ota_vendor = config.get("agent_name") == "ota"
    dataset_module = None
    if use_ota_vendor:
        repo_root = Path(__file__).resolve().parents[1]
        ota_dataset_path = repo_root / "impls" / "ota-v" / "utils" / "datasets.py"
        spec = importlib.util.spec_from_file_location(
            "ota_v_datasets", ota_dataset_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load OTA datasets from {ota_dataset_path}")
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)

    dataset_class_map = {
        "GCDataset": (GCDataset if not use_ota_vendor else dataset_module.GCDataset),
        "HGCDataset": (HGCDataset if not use_ota_vendor else dataset_module.HGCDataset),
    }
    base_dataset_cls = Dataset if not use_ota_vendor else dataset_module.Dataset

    dataset_class = dataset_class_map[config["dataset_class"]]
    train_dataset = dataset_class(base_dataset_cls.create(**train_data), config)
    if val_data is not None:
        val_dataset = dataset_class(base_dataset_cls.create(**val_data), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    eval_ttgs_agent = None

    example_batch = train_dataset.sample(1)
    if config["discrete"]:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_start_time = time.time()
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    key = jax.random.PRNGKey(FLAGS.seed)
    start_index = 1 if FLAGS.restore_epoch is None else FLAGS.restore_epoch
    skip_training = (
        FLAGS.restore_epoch is not None and FLAGS.restore_epoch >= FLAGS.train_steps
    )
    for i in tqdm.tqdm(
        range(start_index, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        batch = train_dataset.sample(config["batch_size"])
        if not skip_training:
            if FLAGS.train_actor_only:
                agent, update_info = agent.actor_update(batch)
            elif FLAGS.train_dynamics_obs_only:
                agent, update_info = agent.dynamics_obs_space_update(batch)
            else:
                agent, update_info = agent.update(batch)

            if i == FLAGS.train_steps:
                timing_log["training_time"] = time.time() - train_start_time

            # Log metrics.
            if i % FLAGS.log_interval == 0:
                train_metrics = {f"training/{k}": v for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(config["batch_size"])
                    if FLAGS.train_actor_only:
                        _, val_info = agent.actor_loss(val_batch, grad_params=None)
                    elif FLAGS.train_dynamics_obs_only:
                        _, val_info = agent.dynamics_obs_space_loss(
                            val_batch, grad_params=None
                        )
                    else:
                        _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update(
                        {f"validation/{k}": v for k, v in val_info.items()}
                    )
                train_metrics["time/epoch_time"] = (
                    time.time() - last_time
                ) / FLAGS.log_interval
                train_metrics["time/total_time"] = time.time() - first_time
                last_time = time.time()
                wandb.log(train_metrics, step=i)
                train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        ####### Evaluate #######
        if FLAGS.eval_at_all and i == FLAGS.train_steps:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent

            ####### Original Agent Evaluation #######
            eval_start_time = time.time()
            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )
            num_tasks = (
                FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            )

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
                mode="standard",
                task_id=FLAGS.task_id if FLAGS.task_id is not None else None,
            )
            standard_eval_loop_start = time.time()
            eval_task_metrics, eval_overall, renders = run_taskwise_evaluation(
                std_params
            )
            timing_log["standard_eval_loop_time"] = (
                time.time() - standard_eval_loop_start
            )

            eval_metrics = {}
            video_logs = {}
            eval_metrics.update(eval_task_metrics)

            for k, v in eval_overall.items():
                eval_metrics[f"evaluation/{k}"] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(
                    renders=renders, n_cols=num_tasks, pre_encode=True
                )
                video_logs["standard_video"] = video

            wandb.log(eval_metrics)
            eval_logger.log(eval_metrics, step=i)

            timing_log["standard_eval_time"] = time.time() - eval_start_time

            ####### TTGS Evaluation - Run after checkpoint save #######
            if i == FLAGS.train_steps:
                try:
                    ttgs_start_time = time.time()
                    eval_ttgs = TTGS(
                        eval_agent,
                        train_dataset=train_data,
                        dist_mode=FLAGS.dist_mode,
                        graph_mode=FLAGS.graph_mode,
                        knn_k=FLAGS.knn_k,
                        tau=FLAGS.tau,
                        te_horizon=FLAGS.te_horizon,
                        batch_size=FLAGS.batch_size,
                        error=FLAGS.error,
                        threshold=FLAGS.threshold,
                        subsample_ablt=FLAGS.subsample_ablt,
                        random_size=FLAGS.random_size,
                        cluster_dist=FLAGS.cluster_dist,
                        cluster_size=FLAGS.cluster_size,
                        subgoal_ablt=FLAGS.subgoal_ablt,
                        ablt_seed=FLAGS.seed,
                        penalty_mode=FLAGS.penalty_mode,
                        penalty_factor=FLAGS.penalty_factor,
                        replan_factor=FLAGS.replan_factor,
                        replan_cooldown=FLAGS.replan_cooldown,
                    )
                    # Log ttgs parameters
                    ttgs_params = {
                        "ttgs/te_horizon": eval_ttgs.te_horizon,
                        "ttgs/tau": eval_ttgs.tau,
                        "ttgs/batch_size": eval_ttgs.batch_size,
                        "ttgs/error": eval_ttgs.error,
                        "ttgs/threshold": eval_ttgs.threshold,
                        "ttgs/subsample_ablt": eval_ttgs.subsample_ablt,
                        "ttgs/subgoal_ablt": eval_ttgs.subgoal_ablt,
                        "ttgs/random_size": eval_ttgs.random_size,
                        "ttgs/cluster_dist": eval_ttgs.cluster_dist,
                        "ttgs/dist_mode": eval_ttgs.dist_mode,
                        "ttgs/graph_mode": eval_ttgs.graph_mode,
                        "ttgs/knn_k": eval_ttgs.knn_k,
                        "ttgs/penalty_mode": eval_ttgs.penalty_mode,
                        "ttgs/penalty_factor": eval_ttgs.penalty_factor,
                        "ttgs/cluster_size": eval_ttgs.cluster_size,
                        "ttgs/replan_factor": eval_ttgs.replan_factor,
                        "ttgs/replan_cooldown": eval_ttgs.replan_cooldown,
                    }
                    wandb.log(ttgs_params)

                    # Run ttgs evaluation across tasks (graph is built inside)
                    ttgs_eval_loop_start = time.time()
                    ttgs_params_eval = EvalParams(
                        agent=eval_ttgs,
                        env=env,
                        seed=FLAGS.seed,
                        train_data=train_data,
                        num_tasks=num_tasks,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                        mode="ttgs",
                        add_dataset_trajectories=FLAGS.add_dataset_trajectories,
                        task_id=FLAGS.task_id if FLAGS.task_id is not None else None,
                    )
                    ttgs_task_metrics, ttgs_overall, ttgs_renders = (
                        run_taskwise_evaluation(ttgs_params_eval)
                    )
                    timing_log["ttgs_eval_loop_time"] = (
                        time.time() - ttgs_eval_loop_start
                    )

                    # Create comparison plot if available
                    if "evaluation/overall_success" in eval_metrics:
                        base_success = eval_metrics.get(
                            "evaluation/overall_success", 0.0
                        )
                        ttgs_success = ttgs_overall.get("overall_success", 0.0)
                        fig = plot_success_comparison(
                            base_success,
                            ttgs_success,
                            title=f"Overall Task Success Rate Comparison\n{FLAGS.env_name} - Training Step {i} - {FLAGS.agent.agent_name}",
                        )
                        plot_path = os.path.join(
                            FLAGS.save_dir, f"comparison_step{i}.png"
                        )
                        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                        wandb.log({"success_rate_comparison": wandb.Image(fig)})
                        plt.close(fig)

                    # Record TTGS timing components
                    timing_log["ttgs_total_time"] = time.time() - ttgs_start_time

                    ttgs_success_metrics = {
                        k: v for k, v in ttgs_task_metrics.items() if "success" in k
                    }
                    if ttgs_success_metrics:
                        wandb.log(ttgs_success_metrics)
                    ttgs_overall_log = {}
                    if "overall_success" in ttgs_overall:
                        ttgs_overall_log["ttgs_evaluation/overall_success"] = (
                            ttgs_overall["overall_success"]
                        )
                    timing_aliases = {
                        "overall_timing_step_per_step": "time_per_step",
                        "overall_timing_graph_build": "time_graph_build",
                        "overall_timing_shortest_path": "time_shortest_path",
                        "overall_timing_subgoal_selection": "subgoal_time",
                        "overall_ttgs_replan_count": "replan_count",
                        "overall_ttgs_replan_episode_ratio": "replan_episode_ratio",
                        "overall_ttgs_inf_path_ratio": "inf_path_ratio",
                        "overall_ttgs_inf_path_count": "inf_path_count",
                        "overall_ttgs_clustering_time": "clustering_time",
                        "overall_ttgs_cluster_centers": "cluster_centers",
                    }
                    for key, alias in timing_aliases.items():
                        if key in ttgs_overall:
                            ttgs_overall_log[f"ttgs_evaluation/{alias}"] = ttgs_overall[
                                key
                            ]
                    if FLAGS.video_episodes > 0 and ttgs_renders:
                        ttgs_video = get_wandb_video(
                            renders=ttgs_renders,
                            n_cols=num_tasks,
                            pre_encode=True,
                        )
                        video_logs["ttgs_video"] = ttgs_video
                    if ttgs_overall_log:
                        wandb.log(ttgs_overall_log)
                    if video_logs:
                        wandb.log(video_logs)
                    eval_ttgs_agent = eval_ttgs

                except Exception as e:
                    print(f"=======TTGS Evaluation Failed: {e}=======")
                    traceback.print_exc()
                    print(
                        f"Checkpoint was already saved. You can run ttgs evaluation later."
                    )
                    wandb.log(
                        {"ttgs_evaluation/failed": 1, "ttgs_evaluation/error": str(e)}
                    )
                    raise e

        # Save agent
        if i % FLAGS.save_interval == 0 and not skip_training:
            print(
                f"=======Saving checkpoint at step {i} - {FLAGS.agent.agent_name}======="
            )
            save_agent(agent, FLAGS.save_dir, i)
            checkpoint_path = os.path.join(FLAGS.save_dir, f"params_{i}.pkl")
            restoring_final_only = (
                FLAGS.restore_path is not None
                and FLAGS.restore_epoch is not None
                and FLAGS.restore_epoch >= FLAGS.train_steps
            )
            if i == FLAGS.train_steps and not restoring_final_only:
                ckp_save_path = "paths.txt"
                key = f"{FLAGS.agent.agent_name}_{FLAGS.env_name}_{FLAGS.seed}"
                with open(ckp_save_path, "a") as f:
                    f.write(f"{key}: {checkpoint_path}\n")

    csv_base_name = (
        "restore_results.csv" if FLAGS.restore_path is not None else "new_results.csv"
    )
    csv_file_hparams = f"{FLAGS.agent.agent_name}_{csv_base_name}"

    # Check if we have both base and TTGS success rates to compare
    base_success_rate = (
        eval_metrics.get("evaluation/overall_success", 0.0)
        if "eval_metrics" in locals()
        else 0.0
    )
    ttgs_success_rate = (
        ttgs_overall.get("overall_success", 0.0) if "ttgs_overall" in locals() else 0.0
    )

    # Extract environment name
    env_name_clean = FLAGS.env_name.replace("-v0", "")
    seed = FLAGS.seed

    # Extended line with hyperparameters and key timing metrics
    eval_overall_data = eval_overall if "eval_overall" in locals() else {}
    ttgs_overall_data = ttgs_overall if "ttgs_overall" in locals() else {}
    base_eval_sec_per_step = eval_overall_data.get("overall_timing_step_per_step", 0.0)
    ttgs_eval_sec_per_step = ttgs_overall_data.get("overall_timing_step_per_step", 0.0)
    ttgs_graph_build_sec = ttgs_overall_data.get("overall_timing_graph_build", 0.0)
    ttgs_shortest_path_sec = ttgs_overall_data.get("overall_timing_shortest_path", 0.0)
    ttgs_subgoal_select_sec = ttgs_overall_data.get(
        "overall_timing_subgoal_selection", 0.0
    )
    ttgs_avg_replan = ttgs_overall_data.get("overall_ttgs_replan_count", 0.0)
    ttgs_replan_episode_ratio = ttgs_overall_data.get(
        "overall_ttgs_replan_episode_ratio", 0.0
    )
    ttgs_inf_path_ratio = ttgs_overall_data.get("overall_ttgs_inf_path_ratio", 0.0)
    ttgs_inf_path_count = ttgs_overall_data.get("overall_ttgs_inf_path_count", 0.0)
    ttgs_clustering_time = ttgs_overall_data.get("overall_ttgs_clustering_time", 0.0)
    ttgs_cluster_centers = ttgs_overall_data.get("overall_ttgs_cluster_centers", 0.0)
    csv_line_hparams = (
        f"{env_name_clean},{FLAGS.agent.agent_name},{base_success_rate:.3f},{ttgs_success_rate:.3f},"
        f"{base_eval_sec_per_step:.6f},{ttgs_eval_sec_per_step:.6f},{ttgs_graph_build_sec:.6f},"
        f"{ttgs_shortest_path_sec:.6f},{ttgs_subgoal_select_sec:.6f},{ttgs_avg_replan:.6f},"
        f"{ttgs_replan_episode_ratio:.6f},{ttgs_inf_path_ratio:.6f},{ttgs_inf_path_count:.6f},"
        f"{ttgs_clustering_time:.6f},{ttgs_cluster_centers:.6f},"
        f"{seed},"
        f"{FLAGS.cluster_dist},{FLAGS.tau},{FLAGS.threshold},"
        f"{FLAGS.subsample_ablt},{FLAGS.random_size},{FLAGS.cluster_size},"
        f"{FLAGS.subgoal_ablt},{FLAGS.dist_mode},{FLAGS.penalty_mode},"
        f"{FLAGS.penalty_factor},{FLAGS.replan_factor},{FLAGS.replan_cooldown},{FLAGS.graph_mode},{FLAGS.knn_k}"
    )

    # Write extended CSV (separate file to keep original format stable)
    file_exists_h = os.path.exists(csv_file_hparams)
    with open(csv_file_hparams, "a", newline="") as f:
        if not file_exists_h:
            f.write(
                "Environment,Base Agent,Base Success Rate,TTGS Success Rate,"
                "Base Eval SecPerStep,TTGS Eval SecPerStep,TTGS Graph Build Sec,"
                "TTGS ShortestPath Sec,TTGS Pick Subgoal Sec,TTGS Replan Count,TTGS Replan Episode Ratio,TTGS Inf Path Ratio,TTGS Inf Path Count,TTGS Clustering Time,TTGS Cluster Centers,Seed,Cluster_Dist,"
                "Tau,Threshold,Subsample_Ablation_Mode,Random_Size,Cluster_Size,"
                "Subgoal_Ablation_Mode,Dist_Mode,Penalty_Mode,Penalty_Factor,"
                "Replan_Factor,Replan_Cooldown,Graph_Mode,KNN_K\n"
            )
        f.write(csv_line_hparams + "\n")

    timing_log["total_experiment_time"] = time.time() - exp_start_time

    timing_payload = {
        "timing/setup_time_minutes": timing_log.get("setup_time", 0) / 60,
        "timing/data_loading_time_minutes": timing_log.get("data_loading_time", 0) / 60,
        "timing/training_time_hours": timing_log.get("training_time", 0) / 3600,
        "timing/standard_eval_time_minutes": timing_log.get("standard_eval_time", 0)
        / 60,
        "timing/standard_eval_loop_minutes": timing_log.get(
            "standard_eval_loop_time", 0
        )
        / 60,
        "timing/ttgs_total_time_minutes": timing_log.get("ttgs_total_time", 0) / 60,
        "timing/ttgs_eval_loop_minutes": timing_log.get("ttgs_eval_loop_time", 0) / 60,
        "timing/total_experiment_time_hours": timing_log.get("total_experiment_time", 0)
        / 3600,
    }
    wandb.log(timing_payload)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
