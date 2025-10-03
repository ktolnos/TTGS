import os
os.environ.setdefault("NX_CUGRAPH_AUTOCONFIG", "True")
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import time
from functools import partial

import wandb
from tqdm import trange, tqdm
import pdb

# -------------------- Function factories to create JIT-compiled functions --------------------
def make_distance_functions(agent, dist_mode: str = 'value', l2_norm_c: float = None):
    """Create a set of JIT-compiled distance calculation functions for a specific agent instance.

    Args:
        agent: The agent that provides value and representation networks.
        env: Optional environment for environment-specific distance calculations.
        use_env_distance: Whether to use environment-related distance instead of value distance.
    """
    l2_norm_c_jnp = None if l2_norm_c is None else jnp.asarray(l2_norm_c, dtype=jnp.float32)
    
    def _value_distance(query, goal):
        """Distance based on converting value estimates to expected steps."""
        values = agent.sample_values(query, goal)
        epsilon = 1e-3  # Numerical stability

        gamma = agent.config['discount']
        # Ensure values are in valid range
        values = jnp.clip(values, -1 / (1 - gamma) + epsilon, -epsilon)

        expected_steps = jnp.log((values * (1 - gamma) + 1)) / jnp.log(gamma)
        expected_steps = jnp.maximum(expected_steps, 0)

        # Replace NaNs with a large constant
        return jnp.where(jnp.isnan(expected_steps), 10000.0, expected_steps)

    def _l2_distance(query, goal):
        """L2 distance on XY coordinates, optionally normalized by l2_norm_c.

        Uses only the first two dims (XY) to be consistent with
        get_subseq_points_dist which computes normalization on XY steps.
        """
        # Handle potential higher-dimensional observations by slicing XY only
        q_xy = query[:2]
        g_xy = goal[:2]
        dist = jnp.linalg.norm(q_xy - g_xy)
        return dist / l2_norm_c_jnp

    def get_distances(query, goal):
        """
        Args:
            query: Query observation
            goal: Goal observation  
        """ 
        if dist_mode == 'l2':
            return _l2_distance(query, goal)
        # Default to value distance unless a custom sampler exists
        if hasattr(agent, 'sample_distance'):
            return agent.sample_distance(query, goal)
        else:
            return _value_distance(query, goal) 
    
    # Vectorized distance calculation function
    def get_dist_vmapped(queries, goal):
        return jax.vmap(get_distances, in_axes=(0, None))(queries, goal)
    
    def get_dist_from_vmapped(query, goals):
        return jax.vmap(get_distances, in_axes=(None, 0))(query, goals)
    
    @jax.jit
    def get_dists_each_with_each(queries):
        """
        queries: (n, d)
        returns:
            dists: (n, n) # distance of queries[i] to queries[j] for all i, j
        """
        dists = jax.vmap(get_dist_vmapped, in_axes=(None, 0))(queries, queries)
        return dists

    @partial(jax.jit, static_argnames=("batch_size",))
    def get_dists_each_with_each_batched(queries: jnp.ndarray,
                                        batch_size: int = 4) -> jnp.ndarray:
        """(n,d) -> (n,n) distances; i-th row = d(queries[i], queries[:])."""
        def row_kernel(q):                 # q: (d,)
            return get_dist_from_vmapped(q, queries)  # (n,)
        return jax.lax.map(row_kernel, queries, batch_size=batch_size)  # (n,n)
    
    @jax.jit
    def get_dists_from_to(queries, goals):
        """
        queries: (n, d)
        goals:  (m, d)
        returns:
            dists: (n, m) # distance of query[i] to goals[j] for all i, j
        """
        return jax.vmap(get_dist_from_vmapped, in_axes=(0, None))(queries, goals)
    
    @jax.jit
    def get_dists_start_goal_trimmed(obs, goal, subsampled_obs):
        dists_to_start = get_dist_from_vmapped(obs, subsampled_obs)
        dists_to_goal = get_dist_vmapped(subsampled_obs, goal)
        
        dists_to_start = jnp.maximum(dists_to_start, 0)
        dists_to_goal = jnp.maximum(dists_to_goal, 0)
        
        closest_to_start_idx = jnp.argmin(dists_to_start)
        closest_to_goal_idx = jnp.argmin(dists_to_goal)
        
        return dists_to_start, dists_to_goal, closest_to_start_idx, closest_to_goal_idx

    @jax.jit
    def get_diagonal_dists(states, goals):
        """
        states: (n, d)
        goals:  (n, d)
        returns:
            dists: (n,)  # distance of states[i] to goals[i] for all i
        """
        return jax.vmap(get_distances, in_axes=(0, 0))(states, goals)
    
    # Create path planning function
    @jax.jit
    def compute_distances_jit(obs, goal, path_points):
        shortest_path_dists = get_dist_from_vmapped(obs, path_points)
        dist_start_goal = get_distances(obs, goal)
        closest_ind = jnp.argmin(shortest_path_dists)
        path_end_to_goal_dist = get_distances(path_points[-1], goal)
        return shortest_path_dists, dist_start_goal, closest_ind, path_end_to_goal_dist
    
    # Create action prediction function
    @jax.jit
    def predict_action_jit(obs, best_obs, key):
        return agent.sample_actions(obs, best_obs, seed=key, temperature=0)
     
    # Return all JIT-compiled functions
    return {
        'get_distances': get_distances,
        'get_dist_vmapped': get_dist_vmapped,
        'get_dist_from_vmapped': get_dist_from_vmapped,
        'get_dists_each_with_each': get_dists_each_with_each,
        'get_dists_from_to': get_dists_from_to,
        'get_dists_start_goal_trimmed': get_dists_start_goal_trimmed,
        'compute_distances_jit': compute_distances_jit,
        'predict_action_jit': predict_action_jit,
        'get_diagonal_dists': get_diagonal_dists,
        'get_dists_each_with_each_batched': get_dists_each_with_each_batched,
    }

# for inference only
class TTGS:
    def __init__(self, agent, train_dataset, tau=12, te_horizon=24, batch_size=4, error=0.001, threshold=24, dist_mode='value',
                 subsample_ablt: str = 'default', random_size: int = 5000, subgoal_ablt: str = 'default', ablt_seed: int = 0, penalty_mode: str = 'dynamic', penalty_factor: float = 1000.0):
        self.agent = agent
        
        # attributes for subgoal navigation
        self.graph = None
        self.subsampled_obs = None
        self.dijstra_shortest_path = None
        self.l2_norm_c = None

        # distance function settings
        self.dist_mode = dist_mode
        assert self.dist_mode in ['value', 'l2']

        if self.dist_mode == 'l2':
            self.l2_norm_c = self.get_subseq_points_dist(train_dataset)

        # jit functions
        self._init_jit_functions()

        # hyperparameters
        self.tau = tau
        self.te_horizon = te_horizon
        self.batch_size = batch_size
        self.error = error
        self.threshold = threshold
        
        # ablation settings
        self.subsample_ablt = subsample_ablt
        self.random_size = random_size
        assert self.subsample_ablt in ['default', 'random_points', 'cluster_only']

        self.subgoal_ablt = subgoal_ablt
        assert self.subgoal_ablt in ['default', 'next']

        self.penalty_factor = penalty_factor
        self.penalty_mode = penalty_mode
        assert self.penalty_mode in ['dynamic', 'static', 'none']

        self.ablt_seed = ablt_seed
    def _init_jit_functions(self):
        dist_funcs = make_distance_functions(self.agent, dist_mode=self.dist_mode, l2_norm_c=self.l2_norm_c)
        for k, v in dist_funcs.items():
            setattr(self, k, v)

    # -------------------- Subgoal Navigation Utilities --------------------
    
    def _td_aware_clustering(
        self,
        high_te_observations,
        shuffle_points=True,
    ):
        H_np = np.asarray(high_te_observations) # CPU
        if shuffle_points:
            rng = np.random.default_rng(0)
            rng.shuffle(H_np, axis=0)
        
        H = jax.device_put(H_np, jax.devices('gpu')[0]) # CPU -> GPU

        n = len(H) # number of high-TE observations
        HTD = self.te_horizon
        ct = float(HTD / 2)
        # One-time precompute, batched and jit.
        D = self.get_dists_each_with_each_batched(H, batch_size=self.batch_size)  # (n, n) on GPU
        D = D.at[jnp.arange(n), jnp.arange(n)].set(0) # (n, n) on GPU
        D_np = np.asarray(jax.device_get(D), dtype=np.float32) # (n, n) GPU -> CPU

        V_indices = [0] # on CPU
        C = [[0]] # on CPU

        pbar = tqdm(range(1, n), desc="Clustering, looping through points")
        for i in pbar:
            centers_idx = np.asarray(V_indices, dtype=np.int32) # (K,) on CPU
            d2c = D_np[i, centers_idx] # (K,) on CPU
            k_local = int(d2c.argmin()) # (K,) on CPU
            if float(d2c[k_local]) > ct:
                V_indices.append(i)
                C.append([i])
            else:
                k = k_local
                C[k].append(i)
                # Recompute
                mem = np.asarray(C[k], dtype=np.int32) # make it a numpy array so that np.ix_ can be used
                sums = D_np[np.ix_(mem, mem)].sum(axis=1) # (m,) on CPU, np.ix_(mem, mem) is (m, m) on CPU
                V_indices[k] = int(mem[int(sums.argmin())])

            if (i & 1024) == 0 or i == n - 1:
                pbar.set_postfix(K=len(V_indices))
        centers_ix = jnp.asarray(V_indices, dtype=np.int32) # CPU -> GPU
        print(f"Found {len(centers_ix)} clusters from {n} high-TE points.")
        return H[centers_ix] # on GPU
        
    def _compute_te_scores(self, trajectory_obs):
        """Compute trajectory efficiency scores.
        Efficiency = distance / te_horizon
        """
        T = len(trajectory_obs)
        H = int(self.te_horizon)
        
        # Convert trajectory to JAX array once for efficiency
        traj_jax = jnp.asarray(trajectory_obs) # CPU -> GPU
        
        current_states = traj_jax[:T - H] # (T-H, d) on GPU
        future_states = traj_jax[H:T] # (T-H, d) on GPU
        dists = self.get_diagonal_dists(current_states, future_states)   # (T-H,) on GPU

        te_scores = dists / float(H)                                     # (T-H,) on GPU
        valid_indices = jnp.arange(T - H, dtype=jnp.int32)               # (T-H,) on GPU

        # Return NumPy arrays if downstream code expects CPU arrays
        return np.asarray(te_scores, dtype=np.float32), np.asarray(valid_indices) # (T-H,) GPU -> CPU

    def get_trajectory_boundaries(self, train_dataset):
        """Find trajectory boundaries using terminals."""
        terminals = train_dataset['terminals']
        trajectory_boundaries = []
        start_idx = 0
        for i, terminal in enumerate(terminals):
            if terminal:
                trajectory_boundaries.append((start_idx, i + 1))
                start_idx = i + 1
        if start_idx < len(terminals):
            trajectory_boundaries.append((start_idx, len(terminals)))
        return trajectory_boundaries

    def get_subseq_points_dist(self, train_dataset):
        """Compute average L2 distance between consecutive points (XY only). """
        observations = np.asarray(train_dataset['observations'])
        traj_bounds = self.get_trajectory_boundaries(train_dataset)

        total_sum = 0.0
        total_count = 0.0

        for start, end in traj_bounds:
            seg = observations[start:end]
            if seg.shape[0] <= 1:
                continue
            # Use only XY coordinates
            seg_xy = seg[:, :2].astype(np.float32, copy=False) # (T, 2)
            diffs = seg_xy[1:] - seg_xy[:-1] # (T-1, 2)
            d = np.linalg.norm(diffs, axis=1) # (T-1,)
            total_sum += float(d.sum())
            total_count += float(d.shape[0])

        avg = total_sum / total_count
        return np.float32(avg)

    def get_subsampled_observations(self, train_dataset):
        """Select top percentage of points with highest TE scores from all training data."""
        if self.subsample_ablt == 'random_points':
            # Determine sample size K
            rng = np.random.default_rng(self.ablt_seed)
            pool = np.asarray(train_dataset['observations'])
            n_pick = min(self.random_size, len(pool))
            idxs = rng.choice(len(pool), size=n_pick, replace=False)
            picked = pool[idxs]
            return jax.device_put(picked, jax.devices('gpu')[0])

        if self.subsample_ablt == 'cluster_only':
            # Cluster-only ablation: replace TE-filtered points with a random
            # sample of the same size from the entire dataset, then cluster.
            rng = np.random.default_rng(self.ablt_seed)
            pool = np.asarray(train_dataset['observations'])
            n_pick = min(self.random_size, len(pool))
            idxs = rng.choice(len(pool), size=n_pick, replace=False)
            picked = pool[idxs]
            cluster_centers = self._td_aware_clustering(picked)
            return cluster_centers

        # Find trajectory boundaries using terminals
        observations = train_dataset['observations'] # (T, D)
        trajectory_boundaries = self.get_trajectory_boundaries(train_dataset)
        # Collect ALL observations with their TE scores
        all_obs_with_scores = []  # List of (observation, te_score) tuples
        
        # Process trajectories to compute TE scores for all points
        pbar = tqdm(trajectory_boundaries, desc="Computing TE scores")
        for traj_idx, (start, end) in enumerate(pbar):
            trajectory_obs = observations[start:end]
            
            if len(trajectory_obs) <= self.te_horizon:
                continue
            
            te_scores, valid_indices = self._compute_te_scores(trajectory_obs)
            
            for te_score, idx in zip(te_scores, valid_indices):
                all_obs_with_scores.append((trajectory_obs[idx], float(te_score)))
            
            # Update progress bar
            pbar.set_postfix({'total_points': len(all_obs_with_scores)})
        
        top_observations = [obs for obs, score in all_obs_with_scores if score >= 1.0 - self.error and score <= 1.0 + self.error] # CPU

        # Apply TD-aware clustering to the selected high-TE observations (to determine K by default)
        cluster_centers = self._td_aware_clustering(top_observations) # (K, d) on GPU

        # If not ablation, return default centers
        return cluster_centers

    def build_graph(self, subsampled_obs, train_dataset=None):
        self.subsampled_obs = subsampled_obs
        N = int(subsampled_obs.shape[0])

        max_dist = float(self.tau)
        subsampled_obs = jnp.asarray(subsampled_obs)  # (n, d)
        subsampled_dists = self.get_dists_each_with_each_batched(subsampled_obs, batch_size=self.batch_size)
        subsampled_dists = subsampled_dists.at[jnp.arange(N), jnp.arange(N)].set(jnp.inf)
        if self.penalty_mode == 'dynamic':
            penalty_factor = jnp.power(self.penalty_factor, subsampled_dists / max_dist)
        elif self.penalty_mode == 'static':
            penalty_factor = self.penalty_factor
        elif self.penalty_mode == 'none':
            penalty_factor = 1.0
        else:
            raise ValueError(f"Invalid penalty mode: {self.penalty_mode}")
        subsampled_dists = jnp.where(
            subsampled_dists > max_dist,
            subsampled_dists * penalty_factor,
            subsampled_dists,
        )
        # Ensure minimum edge weight of 1
        subsampled_dists = jnp.maximum(subsampled_dists, 1)
        G = nx.DiGraph(np.array(subsampled_dists))
        if train_dataset is not None:
            print('=== Adding dataset trajectories to the graph...')
            # Add edges from actual transitions in the dataset
            trajectory_boundaries = self.get_trajectory_boundaries(train_dataset)
            observations = train_dataset['observations']
            for start, end in trajectory_boundaries:
                traj_obs = list(map(tuple, observations[start:end].tolist()))
                distances = np.ones(len(traj_obs) - 1)  # actual transitions have distance 1
                G.add_weighted_edges_from(
                    list(zip(traj_obs[:-1], traj_obs[1:], distances))
                )
            print('=== Added dataset trajectories to the graph.')
        self.graph = G
        return

    def dijkstra_precompute_init(self, obs, goal):
        dists_to_start, dists_to_goal, closest_to_start_idx, closest_to_goal_idx = self.get_dists_start_goal_trimmed(obs, goal, self.subsampled_obs)
        start_idx = closest_to_start_idx.item()
        goal_idx = closest_to_goal_idx.item()

        try:
            path_indices = nx.shortest_path(self.graph, start_idx, goal_idx, weight='weight')
        except nx.NetworkXNoPath as exc:
            try:
                path_indices = nx.shortest_path(
                    self.graph,
                    start_idx,
                    goal_idx,
                    weight='weight',
                    backend='networkx',
                )
            except Exception:
                raise

        self.dijstra_shortest_path = self.subsampled_obs[jnp.array(path_indices)]
        wandb.log({"ttgs/shortest_path_points": len(self.dijstra_shortest_path)})
        wandb.log({"ttgs/dist_to_goal": float(self.get_distances(obs, goal))})
        return

    def get_action_dijkstra_precompute(self, obs, goal, prev_info=None, key=None):
        distance_start = time.time()
        shortest_path_dists_array, dist_start_goal_scalar, closest_ind_scalar, path_end_to_goal_dist_scalar = self.compute_distances_jit(
            obs, goal, self.dijstra_shortest_path)
        distance_time = time.time() - distance_start
        
        shortest_path_dists = np.array(shortest_path_dists_array)
        dist_start_goal = float(dist_start_goal_scalar)
        closest_ind = int(closest_ind_scalar)
        path_end_to_goal_dist = float(path_end_to_goal_dist_scalar)

        subgoal_start = time.time()
        if prev_info is not None and 'closest_ind' in prev_info:
            closest_ind = min(max(closest_ind, prev_info['closest_ind']), len(self.dijstra_shortest_path) - 1)

        path_len = len(self.dijstra_shortest_path)
        if self.subgoal_ablt == 'next':
            best_obs_ind = int(min(closest_ind + 1, path_len - 1))
            best_obs = self.dijstra_shortest_path[best_obs_ind]
            best_obs_d = float(shortest_path_dists[best_obs_ind])
            subgoal_type = 'next'

            if dist_start_goal <= self.threshold or (best_obs_ind == path_len - 1 and path_end_to_goal_dist > self.threshold):
                best_obs_ind = -1
                best_obs = goal
                best_obs_d = dist_start_goal
                subgoal_type = 'goal'
        else:
            threshold = float(self.threshold)
            best_obs = goal
            best_obs_ind = -1
            best_obs_d = dist_start_goal
            subgoal_type = 'goal'
            if dist_start_goal > threshold:
                idxs = np.arange(path_len, dtype=int)
                ahead_mask = idxs > closest_ind
                mask = (shortest_path_dists < threshold) & ahead_mask
                cand = np.flatnonzero(mask)
                if cand.size > 0:
                    best_obs_ind = int(cand[-1])  # furthest along path among reachable under constant threshold
                    best_obs = self.dijstra_shortest_path[best_obs_ind]
                    best_obs_d = float(shortest_path_dists[best_obs_ind])
                    subgoal_type = 'furthest_close'
                else:
                    # Fallback: stay on the closest point (no look-ahead)
                    best_obs_ind = int(min(closest_ind + 1, path_len - 1))
                    best_obs = self.dijstra_shortest_path[best_obs_ind]
                    best_obs_d = float(shortest_path_dists[best_obs_ind])
                    subgoal_type = f'fallback_closest+1' if best_obs_ind < path_len - 1 else 'fallback_goal'

                    if best_obs_ind == path_len - 1 and path_end_to_goal_dist > threshold:
                        # case when closest_ind is the last point on the path and its distance to the goal is still too far
                        best_obs = goal
                        best_obs_ind = -1
                        best_obs_d = dist_start_goal
                        subgoal_type = 'fallback_goal'
        subgoal_time = time.time() - subgoal_start

        action_start = time.time()
        best_action = self.predict_action_jit(obs, best_obs, key)
        action_time = time.time() - action_start
        
        current_min_distance = float(shortest_path_dists[closest_ind])
        
        info = {
            "best_obs": best_obs,
            'subgoal_type': subgoal_type,
            'best_obs_d': best_obs_d,
            'closest_obs': self.dijstra_shortest_path[closest_ind],
            'closest_ind': closest_ind,
            'best_obs_ind': best_obs_ind,
            'min_path_distance': current_min_distance,  # Minimum distance to any point on the path
            'timings': {
                'distance_computation': distance_time,
                'subgoal_selection': subgoal_time,
                'action_sampling': action_time,
            },
        }
        
        return best_action, info
