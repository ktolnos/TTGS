from collections import defaultdict
import time

import jax
import numpy as np
from tqdm import trange

from utils.evaluation import supply_rng

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
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
    num_eval_episodes=50,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    def _policy_action(obs, goal, prev_info=None, seed=None):
        return agent.get_action_dijkstra_precompute(obs, goal, key=seed, prev_info=prev_info)

    policy_action = supply_rng(_policy_action, rng=jax.random.PRNGKey(seed))

    trajs = []
    stats = defaultdict(list)
    shortest_path_times: List[float] = []
    episode_subgoal_times: List[float] = []
    episode_action_times: List[float] = []
    for i in trange(num_eval_episodes):
        traj = defaultdict(list)
        eval_seed = seed + i
        observation, info = env.reset(options=dict(task_id=task_id, seed=eval_seed))
        goal = info.get('goal')
        done = False
        step = 0
        path_start_time = time.time()
        agent.dijkstra_precompute_init(observation, goal)  # get self.dijstra_shortest_path
        path_end_time = time.time()
        path_time = path_end_time - path_start_time
        shortest_path_times.append(path_time)

        prev_info = None  # Initialize prev_info
        step_subgoal_times: List[float] = []
        step_action_times: List[float] = []
        while not done:
            if step == 0:
                action, act_info = policy_action(observation, goal, prev_info=None)
            else:
                action, act_info = policy_action(observation, goal, prev_info=prev_info)

            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            timings = act_info.get('timings', {}) if act_info else {}
            step_subgoal_times.append(float(timings.get('subgoal_selection', 0.0)))
            step_action_times.append(float(timings.get('action_sampling', 0.0)))

            observation = next_observation

            # Update prev_info for next iteration
            prev_info = act_info

        add_to(stats, flatten(info))
        trajs.append(traj)

        if step_subgoal_times:
            episode_subgoal_times.append(float(np.mean(step_subgoal_times)))
        if step_action_times:
            episode_action_times.append(float(np.mean(step_action_times)))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if shortest_path_times:
        stats['timing.shortest_path'] = float(np.mean(shortest_path_times))
    if episode_subgoal_times:
        stats['timing.subgoal_selection'] = float(np.mean(episode_subgoal_times))
    if episode_action_times:
        stats['timing.action_sampling'] = float(np.mean(episode_action_times))

    return stats, trajs
