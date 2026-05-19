import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    MLP,
    GCActor,
    GCBilinearValue,
    GCDiscreteActor,
    GCIQEValue,
    GCMRNValue,
    GCValue,
    Identity,
    LengthNormalize,
    LogParam,
)


class CGIVLAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning (HIQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def goaldistance_loss(self, batch, grad_params):
        """
        利用QRL算法计算goal之间的distance
        """
        d_neg = self.network.select("goal_distance")(
            batch["observations"], batch["gd_random_observations"], params=grad_params
        )
        d_pos = self.network.select("goal_distance")(
            batch["observations"], batch["value_gd_goals"], params=grad_params
        )

        v1_t, v2_t = self.network.select("target_value")(
            batch["observations"], batch["value_gd_goals"]
        )
        v_t = -jnp.minimum(v1_t, v2_t)
        d_pos_targets = jnp.clip(v_t, min=0)
        # Apply loss shaping following the original implementation.
        d_neg_loss = (100 * jax.nn.softplus(5 - d_neg / 100)).mean()
        diff = d_pos_targets - d_pos
        mask = jnp.where((diff >= 0) & (diff <= 1), 0, 1)
        d_pos_loss = (jax.lax.stop_gradient(mask) * (d_pos_targets - d_pos) ** 2).mean()
        value_loss = d_neg_loss * self.config["neg_eps"] + d_pos_loss
        total_loss = value_loss

        return total_loss, {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "d_neg_loss": d_neg_loss,
            "d_neg_mean": d_neg.mean(),
            "d_neg_max": d_neg.max(),
            "d_neg_min": d_neg.min(),
            "d_pos_loss": d_pos_loss,
            "d_pos_mean": d_pos.mean(),
            "mask": mask.mean(),
        }

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        next_v1_t, next_v2_t = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)

        # 蒸馏损失
        d_neg = self.network.select("goal_distance")(
            batch["observations"], batch["value_random_observations"]
        )
        v_neg_1, v_neg_2 = -self.network.select("value")(
            batch["observations"],
            batch["value_random_observations"],
            params=grad_params,
        )

        v_distill_loss_1 = (
            self.config["v_distill_eps"] * ((v_neg_1 - d_neg) ** 2).mean()
        )
        v_distill_loss_2 = (
            self.config["v_distill_eps"] * ((v_neg_2 - d_neg) ** 2).mean()
        )
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        v1_t, v2_t = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2 + v_distill_loss_1 + v_distill_loss_2

        return value_loss, {
            "value_loss": value_loss1 + value_loss2,
            "v_distill_loss": v_distill_loss_1 + v_distill_loss_2,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_actor_loss(self, batch, grad_params, init_goal_rep=True):
        """Compute the low-level actor loss."""
        v1, v2 = self.network.select("target_value")(
            batch["observations"], batch["low_actor_goals"]
        )
        nv1, nv2 = self.network.select("target_value")(
            batch["next_observations"], batch["low_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        # v = self.network.select('goal_distance')(batch['observations'], batch['low_actor_goals'])
        # nv = self.network.select('goal_distance')(batch['next_observations'], batch['low_actor_goals'])
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        if init_goal_rep:
            goal_reps = batch["low_actor_goals"] - batch["observations"]
        else:
            goal_reps = self.network.select("goal_rep")(
                jnp.concatenate(
                    [batch["observations"], batch["low_actor_goals"]], axis=-1
                ),
                params=grad_params,
            )
        if not self.config["low_actor_rep_grad"]:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select("low_actor")(
            batch["observations"], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])

        # if init_goal_rep:
        #     actor_loss = -(exp_a * log_prob).mean()
        # else:
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params, init_goal_rep=True):
        """Compute the high-level actor loss."""
        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        # sample_goals = dist.mode()
        #
        # distance = self.network.select('goal_distance')(sample_goals, batch['high_actor_goals'])
        #
        # distance_loss = distance.mean()/jax.lax.stop_gradient(jnp.abs(distance).mean() + 1e-6)
        #
        # d_pos = self.network.select('discriminator')(batch['observations'], sample_goals)
        # discriminator_loss = jnp.clip(d_pos - 1 + self.config['discriminator_eps'], 0)
        # lam = self.network.select('lam_ds')(params=grad_params)
        # lam_loss = lam * (self.config['eps'] - jax.lax.stop_gradient(discriminator_loss.mean()))
        # log_prob = dist.log_prob(batch['high_actor_targets'])
        # bc_loss = -(self.config['alpha'] * log_prob).mean()
        # actor_loss = self.config['ds_loss_coef'] * distance_loss.mean() + self.config['dc_loss_coef'] * (discriminator_loss.mean()) + bc_loss + lam_loss
        v1, v2 = self.network.select("target_value")(
            batch["goals"], batch["high_actor_goals"]
        )
        nv1, nv2 = self.network.select("target_value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        exp_a = jnp.exp(adv * self.config["high_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)
        if init_goal_rep:
            target = batch["high_actor_targets"] - batch["observations"]
        else:
            target = self.network.select("goal_rep")(
                jnp.concatenate(
                    [batch["observations"], batch["high_actor_targets"]], axis=-1
                )
            )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            # 'sample_goals': sample_goals.mean(),
            # 'actor_loss': actor_loss,
            # 'distance_loss': distance_loss,
            # 'distance': distance.mean(),
            # 'discriminator_loss': discriminator_loss.mean(),
            # 'lam_loss': lam_loss,
            # 'lam': lam,
            # 'bc_loss': bc_loss
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - target) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    @partial(
        jax.jit,
        static_argnames=[
            "high_actor_update",
            "low_actor_update",
            "value_update",
            "goaldistance_update",
            "init_goal_rep",
        ],
    )
    def total_loss(
        self,
        batch,
        grad_params,
        rng=None,
        low_actor_update=False,
        high_actor_update=False,
        value_update=False,
        goaldistance_update=False,
        init_goal_rep=False,
    ):
        """Compute the total loss."""
        info = {}
        if goaldistance_update:
            goaldistance_loss, goaldistance_info = self.goaldistance_loss(
                batch, grad_params
            )
            for k, v in goaldistance_info.items():
                info[f"goaldistance/{k}"] = v
        else:
            goaldistance_loss = 0

        if value_update:
            value_loss, value_info = self.value_loss(batch, grad_params)
            for k, v in value_info.items():
                info[f"value/{k}"] = v
        else:
            value_loss = 0

        if low_actor_update:
            low_actor_loss, low_actor_info = self.low_actor_loss(
                batch, grad_params, init_goal_rep
            )
            for k, v in low_actor_info.items():
                info[f"low_actor/{k}"] = v
        else:
            low_actor_loss = 0

        if high_actor_update:
            high_actor_loss, high_actor_info = self.high_actor_loss(
                batch, grad_params, init_goal_rep
            )
            for k, v in high_actor_info.items():
                info[f"high_actor/{k}"] = v
        else:
            high_actor_loss = 0

        loss = value_loss + low_actor_loss + high_actor_loss + goaldistance_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @partial(
        jax.jit,
        static_argnames=[
            "low_actor_update",
            "high_actor_update",
            "value_update",
            "goaldistance_update",
            "init_goal_rep",
        ],
    )
    def update(
        self,
        batch,
        low_actor_update=False,
        high_actor_update=False,
        value_update=False,
        goaldistance_update=False,
        init_goal_rep=False,
    ):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=rng,
                low_actor_update=low_actor_update,
                high_actor_update=high_actor_update,
                value_update=value_update,
                goaldistance_update=goaldistance_update,
                init_goal_rep=init_goal_rep,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if value_update:
            self.target_update(new_network, "value")
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=["init_goal_rep", "use_subgoals"])
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        init_goal_rep=True,
        use_subgoals=True,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )
        if use_subgoals:
            low_goals = high_dist.sample(seed=high_seed)
            if init_goal_rep:
                low_goals = low_goals + observations
                low_dist = self.network.select("low_actor")(
                    observations,
                    low_goals - observations,
                    goal_encoded=False,
                    temperature=temperature,
                )
            else:
                low_goals = (
                    low_goals
                    / jnp.linalg.norm(low_goals, axis=-1, keepdims=True)
                    * jnp.sqrt(low_goals.shape[-1])
                )
                low_dist = self.network.select("low_actor")(
                    observations, low_goals, goal_encoded=True, temperature=temperature
                )
        else:
            low_goals = goals
            if init_goal_rep:
                low_goals = low_goals + observations
                low_dist = self.network.select("low_actor")(
                    observations,
                    low_goals - observations,
                    goal_encoded=False,
                    temperature=temperature,
                )
            else:
                # low_goals = low_goals / jnp.linalg.norm(low_goals, axis=-1, keepdims=True) * jnp.sqrt(low_goals.shape[-1])
                low_dist = self.network.select("low_actor")(
                    observations, low_goals, goal_encoded=False, temperature=temperature
                )

        # if init_goal_rep:
        #     low_goals = low_goals + observations
        #     low_dist = self.network.select('low_actor')(observations, low_goals - observations, goal_encoded=False,
        #                                                 temperature=temperature)
        # else:
        #     low_goals = low_goals / jnp.linalg.norm(low_goals, axis=-1, keepdims=True) * jnp.sqrt(low_goals.shape[-1])
        #     low_dist = self.network.select('low_actor')(observations, low_goals, goal_encoded=True, temperature=temperature)

        actions = low_dist.sample(seed=low_seed)

        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions, low_goals

    def sample_values(self, observations, goals):
        """Sample values from the value function."""
        v1, v2 = self.network.select("value")(observations, goals)
        v = (v1 + v2) / 2
        return v

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_goals,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
            ex_goals: Example goals
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config["value_hidden_dims"], config["rep_dim"]),
                activate_final=False,
                layer_norm=config["layer_norm"],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define encoders.
        encoders = dict()
        if config["gd_encoder"] is not None:
            encoder_module = encoder_modules[config["gd_encoder"]]
            encoders["goal_distance"] = encoder_module()
        # Define the encoders that handle the inputs to the value and actor networks.
        # The subgoal representation phi([s; g]) is trained by the parameterized value function V(s, phi([s; g])).
        # The high-level actor predicts the subgoal representation phi([s; w]) for subgoal w given s and g.
        # The low-level actor predicts actions given the current state s and the subgoal representation phi([s; w]).
        if config["encoder"] is not None:
            # Pixel-based environments require visual encoders for state inputs, in addition to the pre-defined shared
            # encoder for subgoal representations.

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )

            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            high_actor_encoder_def = None

        if config["init_goal_rep"] is True:
            low_actor_encoder_def = None
            high_actor_encoder_def = None
            value_encoder_def = None
            target_value_encoder_def = None

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )

        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        goaldistance_def = GCIQEValue(
            hidden_dims=config["goaldistance_hidden_dims"],
            latent_dim=config["goaldistance_latent_dim"],
            layer_norm=config["goaldistance_layer_norm"],
            dim_per_component=8,
            encoder=encoders.get("goal_distance"),
        )

        if config["discrete"]:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["rep_dim"],
            state_dependent_std=False,
            const_std=config["const_std"],
            gc_encoder=high_actor_encoder_def,
        )
        lam_def = LogParam()
        lam_dc_def = LogParam()
        lam_ds_def = LogParam()
        network_info = dict(
            goal_rep=(
                goal_rep_def,
                (jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            ),
            goal_distance=(goaldistance_def, (ex_observations, ex_goals)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            lam=(lam_def, ()),
            lam_dc=(lam_dc_def, ()),
            lam_ds=(lam_ds_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="cgivl",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_log_q=True,
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            alpha=0.003,
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            gd_encoder=ml_collections.config_dict.placeholder(str),
            # Dataset hyperparameters.
            dataset_class="SHGCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            gd_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            gd_p_trajgoal=0.8,  # Probability of using a future state in the same trajectory as the value goal.
            gd_p_randomgoal=0.2,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            gd_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
            goaldistance_hidden_dims=(512, 512, 512),
            goaldistance_latent_dim=512,
            goaldistance_layer_norm=True,
            goal_dim=2,
            eps=0.05,  # Margin for the dual lambda loss.
            neg_eps=0.001,
            init_goal_rep=False,
            min_d=0.0,
            v_distill_eps=0.01,
            use_subgoals=True,
        )
    )
    return config
