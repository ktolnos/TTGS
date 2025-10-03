import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset


def load_validation_dataset_only(dataset_name, frame_stack=None):
    """Only load the validation dataset, not create the environment.
    
    Args:
        dataset_name: the dataset name.
        frame_stack: the number of stacked frames (note: this parameter has no effect on loading the dataset, but is retained for interface consistency).
        
    Returns:
        the validation dataset.
    """
    # parse the dataset name
    splits = dataset_name.split('-')
    dataset_add_info = False
    
    if 'singletask' in splits:
        # single-task environment
        pos = splits.index('singletask')
        env_name = '-'.join(splits[: pos - 1] + splits[pos:]) 
        dataset_name = '-'.join(splits[:pos] + splits[-1:])
        dataset_add_info = True
    elif 'oraclerep' in splits:
        # environment with oracle goal representations
        env_name = '-'.join(splits[:-3] + splits[-1:])
        dataset_name = '-'.join(splits[:-2] + splits[-1:])
        dataset_add_info = True
    else:
        # original goal-conditioned environment
        env_name = '-'.join(splits[:-2] + splits[-1:])
    
    # use the low-level functions in ogbench.utils
    from ogbench.utils import download_datasets, load_dataset, DEFAULT_DATASET_DIR
    
    # load the dataset
    dataset_dir = os.path.expanduser(DEFAULT_DATASET_DIR)
    download_datasets([dataset_name], dataset_dir)
    
    val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    
    # determine the data types of observations and actions
    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    
    # load the validation dataset
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=True,
        add_info=dataset_add_info,
    )
    
    # remove the information keys (if needed)
    if not dataset_add_info:
        for k in ['qpos', 'qvel', 'button_states']:
            if k in val_dataset:
                del val_dataset[k]
    
    return Dataset.create(**val_dataset)
