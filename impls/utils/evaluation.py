import time
from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


# first call → split(rng0) → (rng1, key1) → use key1, store rng1
# second call → split(rng1) → (rng2, key2) → use key2, store rng2
# … and so on.


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
    seed,
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
        seed: Seed for the random number generator.
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
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        eval_seed = seed + i
        observation, info = env.reset(
            seed=eval_seed, options=dict(task_id=task_id, render_goal=should_render)
        )
        goal = info.get("goal")
        goal_frame = info.get("goal_rendered")
        done = False
        step = 0
        render = []
        step_loop_times = []
        while not done:
            loop_start_time = time.time()
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
            loop_end_time = time.time()
            if not should_render:
                step_loop_times.append(loop_end_time - loop_start_time)
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
            if step_loop_times:
                stats["timing.step_per_step"].append(float(np.mean(step_loop_times)))
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_maze(
    agent,
    env,
    env_name,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    init_goal_rep=True,
    use_subgoals=True,
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
            action, high_goals = actor_fn(
                observations=observation,
                goals=goal,
                temperature=eval_temperature,
                init_goal_rep=init_goal_rep,
                use_subgoals=use_subgoals,
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
                if (
                    ("point" in env_name or "ant" in env_name)
                    and "visual" not in env_name
                    and ("large" in env_name or "giant" in env_name)
                ):
                    size = 200

                    def xy_to_pixxy(x, y):
                        if "large" in env_name:
                            pixx = (x / 36) * (0.93 - 0.07) + 0.07
                            pixy = (y / 24) * (0.21 - 0.79) + 0.79
                        elif "giant" in env_name:
                            pixx = (x / 52) * (0.955 - 0.05) + 0.05
                            pixy = (y / 36) * (0.19 - 0.81) + 0.81
                        return pixx, pixy

                    x, y = high_goals[:2]
                    pixx, pixy = xy_to_pixxy(x, y)
                    frame[
                        int((pixy - 0.02) * size) : int((pixy + 0.02) * size),
                        int((pixx - 0.02) * size) : int((pixx + 0.02) * size),
                        0,
                    ] = 255
                    frame[
                        int((pixy - 0.02) * size) : int((pixy + 0.02) * size),
                        int((pixx - 0.02) * size) : int((pixx + 0.02) * size),
                        1:3,
                    ] = 0

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
