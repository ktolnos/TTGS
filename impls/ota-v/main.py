import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate_success_metrics, evaluate_value_metrics
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "antmaze-large-navigate-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 10000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 500000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

config_flags.DEFINE_config_file("wandb", "utils/wandb.py", lock_config=False)
config_flags.DEFINE_config_file("agent", "agents/gciql.py", lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        project=FLAGS.wandb.project, group=FLAGS.wandb.group, mode=FLAGS.wandb.mode
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.wandb.project, FLAGS.wandb.group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config["frame_stack"]
    )

    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

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
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(
        range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        # Update agent.
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config["batch_size"])
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
        if i == 1 or i % FLAGS.eval_interval == 0:
            num_eval_episodes = 1 if i == 1 else FLAGS.eval_episodes
            eval_agent = (
                jax.device_put(agent, device=jax.devices("cpu")[0])
                if FLAGS.eval_on_cpu
                else agent
            )

            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )
            num_tasks = (
                FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            )

            # Success metrics
            success_metrics = evaluate_success_metrics(
                eval_agent, env, config, FLAGS, num_eval_episodes, num_tasks
            )

            # Value metrics (Maze/Cube environments)
            if (
                "maze" in FLAGS.env_name or "cube" in FLAGS.env_name
            ) and "visual" not in FLAGS.env_name:
                value_metrics = evaluate_value_metrics(
                    eval_agent, env, FLAGS.env_name, config, num_tasks
                )

            eval_metrics = {**success_metrics, **value_metrics}

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
