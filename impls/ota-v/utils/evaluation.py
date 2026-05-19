import os
from collections import defaultdict

import jax
import numpy as np
from tqdm import trange

from utils.log_utils import get_value_plot, get_wandb_video


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(
            options=dict(task_id=task_id, render_goal=should_render)
        )
        goal = info.get("goal")
        goal_frame = info.get("goal_rendered")
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(
                observations=observation, goals=goal, temperature=eval_temperature
            )
            action = np.array(action)
            if not config.get("discrete"):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_success_metrics(agent, env, config, FLAGS, num_eval_episodes, num_tasks):
    """evaluate success rates and render videos."""
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    renders = []

    for task_id in trange(1, num_tasks + 1, desc="Evaluating Success Metrics"):
        task_name = (
            env.unwrapped.task_infos[task_id - 1]["task_name"]
            if hasattr(env.unwrapped, "task_infos")
            else env.task_infos[task_id - 1]["task_name"]
        )

        eval_info, trajs, cur_renders = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
            eval_gaussian=FLAGS.eval_gaussian,
        )

        renders.extend(cur_renders)

        metric_names = ["success"]
        eval_metrics.update(
            {
                f"evaluation/{task_name}_{k}": v
                for k, v in eval_info.items()
                if k in metric_names
            }
        )
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)

    for k, v in overall_metrics.items():
        eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)

    if FLAGS.video_episodes > 0:
        eval_metrics["video"] = get_wandb_video(renders=renders, n_cols=num_tasks)

    return eval_metrics


def evaluate_value(
    agent,
    env,
    env_name,
    task_id,
    config,
):
    """Evaluate the value function in the environment."""
    splits = env_name.split("-")
    agent_type = splits[0]
    difficulty = splits[1]

    trajs_path = os.path.join(
        os.getcwd(), "optimal_trajs/data", agent_type, difficulty, f"task{task_id}.npz"
    )
    if not os.path.exists(trajs_path):
        return None

    try:
        trajs = np.load(trajs_path, allow_pickle=True)
    except Exception as e:
        print(f"[Warning] Failed to load trajectory for task {task_id}: {e}")
        return None

    observations = np.stack(trajs["observations"])
    goals = np.repeat(
        trajs["observations"][-1].reshape(1, -1), len(observations), axis=0
    )

    value_info = dict()
    if config["agent_name"] == "hiql":
        v1, v2 = agent.network.select("value")(observations, goals)
        value_info["low_value"] = (v1 + v2) / 2
    elif config["agent_name"] == "ota":
        lv1, lv2 = agent.network.select("low_value")(observations, goals)
        value_info["low_value"] = (lv1 + lv2) / 2
        hv1, hv2 = agent.network.select("high_value")(observations, goals)
        value_info["high_value"] = (hv1 + hv2) / 2
    else:
        # NOTE: Implement if needed for other algorithms.
        return None

    return value_info


def moving_average(data, window_size=5):
    """Calculate moving average."""
    data = np.asarray(data, dtype=float)
    n = len(data)

    cumsum = np.cumsum(data)
    cumsum = np.insert(cumsum, 0, 0)

    window_lengths = np.minimum(np.arange(1, n + 1), window_size)

    start_indices = np.arange(n) - window_lengths + 1
    start_indices[start_indices < 0] = 0

    sums = cumsum[np.arange(1, n + 1)] - cumsum[start_indices]
    averages = sums / window_lengths

    return averages


def measure_order_consistency(env_name, k, value_list):
    """Measure order consistency."""
    window_size = 30  # default window size
    window_size_map = {"humanoidmaze": 150, "antmaze": 30, "cube": 10}
    for key, value in window_size_map.items():
        if key in env_name:
            window_size = value
            break

    smoothed_value_list = moving_average(value_list, window_size=window_size)
    consistency = np.mean(smoothed_value_list[k:] > smoothed_value_list[:-k])
    return consistency


def evaluate_value_metrics(agent, env, env_name, config, num_tasks):
    """Visualize value plots and measure order consistency."""
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    subgoal_steps = config.get("subgoal_steps", 1)

    for task_id in trange(1, num_tasks + 1, desc="Evaluating Value Metrics"):
        value_info = evaluate_value(
            agent=agent,
            env=env,
            env_name=env_name,
            task_id=task_id,
            config=config,
        )
        if not value_info:
            continue

        low_values = value_info.get("low_value")
        high_values = value_info.get("high_value")
        value_plot = get_value_plot(low_values, high_values)
        eval_metrics[f"value_plot/task_{task_id}"] = value_plot

        def compute_consistency(tag, values):
            """Helper function to compute order consistency."""
            consistency = measure_order_consistency(env_name, subgoal_steps, values)
            eval_metrics[f"order_consistency_ratio/{tag}/task_{task_id}"] = consistency
            overall_metrics[tag].append(consistency)

        if high_values is None:
            compute_consistency("value", low_values)
        else:
            compute_consistency("low_value", low_values)
            compute_consistency("high_value", high_values)

    for k, v in overall_metrics.items():
        eval_metrics[f"order_consistency_ratio/overall_{k}"] = np.mean(v)

    return eval_metrics
