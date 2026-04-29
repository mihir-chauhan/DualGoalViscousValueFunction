import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCValue, MLP
from utils.vib import VIB
from jax.scipy.special import logsumexp

class GCIVLVIBAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.
    Uses a VIB (Variational Information Bottleneck) goal representation.

    This is a variant of GCIQL that only uses a V function, without Q functions.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def stochastic_fk_loss(self, batch, grad_params, key):
        """
        Calculates the Viscous Geometric Regularization via Fixed-Point Iteration.
        Implements the update: V(s) <- lambda * log( E[ exp(V(s')/lambda) ] )
        
        This minimizes || V(s) - T_nu V(s) ||^2, ensuring V(s) evolves according 
        to the viscous HJB operator without 'collusion' from neighbors.
        """
        # 1. Hyperparameters
        nu = self.config.get('viscous_scale', 0.001)
        # To satisfy the Taylor expansion V + nu*Laplacian + 0.5*|grad|^2:
        # We need sigma^2 = 2*nu and lambda = sigma^2 (so lambda = 2*nu).
        # Adjust per your specific PDE derivation if different.
        lambda_temp = self.config.get('temperature', 2.0 * nu) 
        sigma = jnp.sqrt(2.0 * nu)
        K = 10
        kappa = 1.0
        
        # 2. Setup Inputs
        obs = batch['observations']
        local_speed = batch['speed']
        if local_speed.ndim == 1: local_speed = local_speed[:, None]
        goal_reps, kl_loss, kl_info = self.network.select('vib')(
            batch['value_goals'], key, encoded=False, params=grad_params
        )
        B, D = obs.shape
        G = goal_reps.shape[-1]

        # 3. Get V(s) [Anchor]
        v_out = self.network.select('value')(obs, goal_reps, params=grad_params)
        
        # Ensemble handling
        if isinstance(v_out, tuple):
            v_s = (v_out[0] + v_out[1]) / 2.0
        elif hasattr(v_out, 'shape') and v_out.shape[0] == 2 and v_out.ndim > 1:
            v_s = jnp.mean(v_out, axis=0)
        else:
            v_s = v_out
            
        if v_s.ndim == 1: v_s = v_s[:, None] # (B, 1)

        # 4. Generate Neighbors (s + epsilon)
        obs_expanded = jnp.repeat(obs[:, None, :], K, axis=1)
        goal_expanded = jnp.repeat(goal_reps[:, None, :], K, axis=1)
        
        noise = jax.random.normal(key, shape=obs_expanded.shape) * sigma
        noisy_obs = obs_expanded + noise
        
        flat_noisy_obs = noisy_obs.reshape(B * K, D)
        flat_goal = goal_expanded.reshape(B * K, G)
        
        # 5. Get V(s + epsilon) using Current Params
        # We use the same params, but strictly for target calculation
        v_neighbors_out = self.network.select('value')(flat_noisy_obs, flat_goal, params=grad_params)
        
        if isinstance(v_neighbors_out, tuple):
            v_neighbors_flat = (v_neighbors_out[0] + v_neighbors_out[1]) / 2.0
        elif hasattr(v_neighbors_out, 'shape') and v_neighbors_out.shape[0] == 2 and v_neighbors_out.ndim > 1:
            v_neighbors_flat = jnp.mean(v_neighbors_out, axis=0)
        else:
            v_neighbors_flat = v_neighbors_out
            
        v_neighbors = v_neighbors_flat.reshape(B, K) # (B, K)

        # 6. Compute Diffusion Target (The Semigroup Operator)
        # Operator: T[V](s) = lambda * ( LogSumExp(V(s')/lambda) - log(K) )
        # To maintain maximum precision (avoiding large exponents), we can
        # center the LSE operation using the anchor v_s (constant shift identity).
        # T[V](s) = v_s + lambda * ( LogSumExp((V(s') - v_s)/lambda) - log(K) )
        
        # Note: v_s here is treated as a constant value for the shift, not a gradient path.
        # This form is algebraically identical to the standard LSE but numerically safer
        # if V is large (e.g. -1000) but local differences are small.
        
        diff_v = (v_neighbors - v_s) / lambda_temp
        lse_diff = logsumexp(diff_v, axis=1) # Shape (B,)
        
        # Reconstruct the full target value
        target_update = v_s.squeeze() + lambda_temp * (lse_diff - jnp.log(K))
        target_update = target_update[:, None] # (B, 1)

        # 7. Loss Calculation
        # STOP GRADIENT on the target. This enforces the fixed-point iteration.
        # The anchor V(s) must move to match the diffused surface defined by neighbors.
        # Gradients only flow through v_s on the LHS.
        
        fixed_point_target = jax.lax.stop_gradient(target_update)
        loss = jnp.square(jnp.maximum(0.0, v_s - fixed_point_target)).mean()

        # --- LOSS COMPONENT 2: Riemannian Metric (Slope) ---
        # "Don't be steeper than the geometry allows."
        # Enforces ||grad V|| <= kappa / speed
        
        # 1. Compute Finite Difference Gradient Magnitude
        # slope ~ |dV| / |dx|
        # dist_to_neighbors is (B, K)
        dist_to_neighbors = jnp.linalg.norm(noise, axis=-1)
        dist_to_neighbors = jax.lax.stop_gradient(dist_to_neighbors) # Treat as fixed constant
        
        # Absolute difference in value
        abs_diff = jnp.abs(v_neighbors - v_s)
        
        # Estimate Gradient Norm (Value change per unit distance)
        grad_estimate = abs_diff / (dist_to_neighbors + 1e-6)
        
        # 2. Compute Allowed Slope (Riemannian Speed Limit)
        # Near walls (speed ~ 0.1) -> Limit ~ 10.0 (Steep Cliff Allowed)
        # Open space (speed ~ 1.0) -> Limit ~ 1.0 (Smooth Slope enforced)
        max_slope = kappa / (local_speed + 1e-6)
        
        # 3. Penalize Slope Violations
        # Only penalize if gradient is STEEPER than allowed.
        # (It's okay to be flatter than the limit).
        metric_violation = jnp.maximum(0.0, grad_estimate - max_slope)
        loss_metric = jnp.square(metric_violation).mean()

        # --- Combine ---
        total_loss = loss + loss_metric
        
        return total_loss, {
            'viscous_loss': loss,
            'v_mean': v_s.mean(),
            'boundary_loss': loss_metric.mean(),
            'avg_speed': local_speed.mean(),
            'target_mean': fixed_point_target.mean(),
            'diffusion_drift': (fixed_point_target - v_s).mean(), # Represents nu * Laplacian
            'avg_max_slope': max_slope.mean(),
            'avg_grad_est': grad_estimate.mean()
        }

    def value_loss(self, batch, grad_params, rng):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        goal_reps, kl_loss, kl_info = self.network.select('vib')(
            batch['value_goals'], rng, encoded=False, params=grad_params
        )

        (next_v1_t, next_v2_t) = self.network.select('target_value')(
            batch['next_observations'], jax.lax.stop_gradient(goal_reps)
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], jax.lax.stop_gradient(goal_reps))
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], goal_reps, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        total_loss = value_loss + kl_loss
        logs = {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

        if self.config.get('enable_fk_regularization', True):
            v_loss, v_loss_dict = self.stochastic_fk_loss(batch, grad_params, self.rng)
            total_loss += v_loss
            logs.update(**v_loss_dict)
            return total_loss, logs | kl_info

        return value_loss + kl_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'kl_loss': kl_loss,
        } | kl_info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the AWR actor loss."""
        goal_reps, _, _ = self.network.select('vib')(batch['actor_goals'], rng, encoded=False)

        v1, v2 = self.network.select('value')(batch['observations'], jax.lax.stop_gradient(goal_reps))
        nv1, nv2 = self.network.select('value')(batch['next_observations'], jax.lax.stop_gradient(goal_reps))
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, value_rng = jax.random.split(rng)
        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        vib_seed, seed = jax.random.split(seed)
        goal_reps, _, _ = self.network.select('vib')(goals, vib_seed, encoded=False)

        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        sample_seed, seed = jax.random.split(seed)
        actions = dist.sample(seed=sample_seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_goals=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # NOTE: this version does not support pixel-based observations; please refer to the visual-dedicated file.
        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )

        vib_def = VIB(
            encoder=MLP(
                hidden_dims=config['vib_hidden_dims'],
                layer_norm=config['layer_norm'],
            ),
            beta=config['beta'],
            rep_dim=config['goalrep_dim'],
        )

        network_info = dict(
            vib=(vib_def, (ex_observations, jax.random.PRNGKey(0), False)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcivl_vib',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            vib_hidden_dims=(512, 512, 512),  # VIB network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            goalrep_dim=64,  # Dimensionality of the VIB goal representation.
            beta=0.003,  # VIB strength hyperparameter.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            enable_fk_regularization = True,
            speed_profile = 'constant', # the speed profile used in the Eikonal loss
            oraclerep=False,  # always False; dummy option for compatibility.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
