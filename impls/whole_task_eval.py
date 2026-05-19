import pdb
import time
from collections import defaultdict
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from utils.evaluation import supply_rng


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
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render in addition to evaluation episodes.
        video_frame_skip: Number of environment steps to skip between rendered frames.
    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """

    def _policy_action(obs, goal, prev_info=None, seed=None):
        return agent.get_action_dijkstra_precompute(
            obs, goal, key=seed, prev_info=prev_info
        )

    policy_action = supply_rng(_policy_action, rng=jax.random.PRNGKey(seed))

    trajs = []
    stats = defaultdict(list)
    shortest_path_times: List[float] = []
    episode_subgoal_times: List[float] = []
    episode_step_times: List[float] = []
    episode_replan_counts: List[int] = []
    episode_replan_occurrences: List[float] = []
    inf_path_count = 0
    total_paths = 0
    renders: List[np.ndarray] = []

    replan_factor_value = float(getattr(agent, "replan_factor", 0.0))
    agent_threshold_value = float(getattr(agent, "threshold", 0.0))
    replan_cooldown_steps = int(getattr(agent, "replan_cooldown", 0) or 0)
    replanning_active = replan_factor_value > 0 and agent_threshold_value > 0

    total_episodes = num_eval_episodes + num_video_episodes
    for i in trange(total_episodes):
        start = time.time()
        traj = defaultdict(list)
        eval_seed = seed + i
        should_render = i >= num_eval_episodes
        reset_options = dict(task_id=task_id)
        if should_render:
            reset_options["render_goal"] = True
        observation, info = env.reset(seed=eval_seed, options=reset_options)
        goal = info.get("goal")
        goal_frame = info.get("goal_rendered")
        done = False
        step = 0
        path_start_time = time.time()
        agent.dijkstra_precompute_init(
            observation, goal
        )  # get self.dijstra_shortest_path
        if not should_render:
            total_paths += 1
            if bool(getattr(agent, "last_path_has_inf_edge", False)):
                inf_path_count += 1
        path_end_time = time.time()
        path_time = path_end_time - path_start_time
        # print(f"episode {i} path time: {path_time:.2f}s")
        if not should_render:
            shortest_path_times.append(path_time)

        prev_info = None  # Initialize prev_info
        step_subgoal_times: List[float] = []
        step_loop_times: List[float] = []
        render_frames: List[np.ndarray] = []
        replan_trigger_count = 0
        last_replan_step = -replan_cooldown_steps
        while not done:
            loop_start_time = time.time()
            if step == 0:
                action, act_info = policy_action(observation, goal, prev_info=None)
            else:
                action, act_info = policy_action(observation, goal, prev_info=prev_info)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            # print(f"episode {i} env step time: {step_time:.2f}s")
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
            # #print info to see what is the subgoal type and best obs ind and closest ind
            # print(f"subgoal type: {act_info.get('subgoal_type', 'None')}")
            # print(f"best obs ind: {act_info.get('best_obs_ind', 'None')}")
            # print(f"closest ind: {act_info.get('closest_ind', 'None')}")
            add_to(traj, transition)
            timings = act_info.get("timings", {}) if act_info else {}
            step_subgoal_times.append(float(timings.get("subgoal_selection", 0.0)))

            observation = next_observation

            # Replan logic: recompute the shortest path when the agent drifts too far from it.
            replan_start_time = time.time()
            if replanning_active and act_info is not None:
                best_obs = act_info.get("best_obs")
                subgoal_distance = None
                if best_obs is not None:
                    obs_jnp = jnp.asarray(observation)
                    best_obs_jnp = jnp.asarray(best_obs)
                    subgoal_distance = float(
                        jax.device_get(agent.get_distances(obs_jnp, best_obs_jnp))
                    )
                if subgoal_distance is not None:
                    distance_limit = replan_factor_value * agent_threshold_value
                    cooldown_ready = (
                        replan_cooldown_steps <= 0
                        or (step - last_replan_step) >= replan_cooldown_steps
                    )
                    if subgoal_distance > distance_limit and cooldown_ready:
                        agent.dijkstra_precompute_init(observation, goal)
                        if not should_render:
                            total_paths += 1
                            if bool(getattr(agent, "last_path_has_inf_edge", False)):
                                inf_path_count += 1
                        replan_trigger_count += 1
                        last_replan_step = step
                        print(
                            f"At episode {i}, replanning triggered at step {step} "
                            f"(distance {subgoal_distance:.2f} > {distance_limit:.2f})"
                        )
            replan_end_time = time.time()
            replan_time = replan_end_time - replan_start_time
            # print(f"episode {i} replan time: {replan_time:.2f}s")

            # Update prev_info for next iteration
            prev_info = act_info

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    frame = np.concatenate([goal_frame, frame], axis=0)
                render_frames.append(np.asarray(frame))

            loop_end_time = time.time()
            if not should_render:
                step_loop_times.append(loop_end_time - loop_start_time)

        if should_render:
            if render_frames:
                renders.append(np.asarray(render_frames))
        else:
            add_to(stats, flatten(info))
            trajs.append(traj)
        end = time.time()
        # print(f"Episode {i} time: {end - start:.2f}s")
        if not should_render and step_subgoal_times:
            episode_subgoal_times.append(float(np.mean(step_subgoal_times)))
        if not should_render and step_loop_times:
            episode_step_times.append(float(np.mean(step_loop_times)))
        if not should_render and replanning_active:
            episode_replan_counts.append(replan_trigger_count)
            episode_replan_occurrences.append(1.0 if replan_trigger_count > 0 else 0.0)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if shortest_path_times:
        stats["timing.shortest_path"] = float(np.mean(shortest_path_times))
    if episode_subgoal_times:
        stats["timing.subgoal_selection"] = float(np.mean(episode_subgoal_times))
    if episode_step_times:
        stats["timing.step_per_step"] = float(np.mean(episode_step_times))
    if replanning_active and episode_replan_counts:
        stats["ttgs.replan_count"] = float(np.mean(episode_replan_counts))
    if replanning_active and episode_replan_occurrences:
        stats["ttgs.replan_episode_ratio"] = float(np.mean(episode_replan_occurrences))
    ratio = float(inf_path_count) / float(total_paths) if total_paths > 0 else 0.0
    stats["ttgs.inf_path_ratio"] = ratio
    stats["ttgs.inf_path_count"] = float(inf_path_count)

    return stats, trajs, renders
