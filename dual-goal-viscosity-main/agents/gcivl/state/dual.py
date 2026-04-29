import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue
from jax.scipy.special import logsumexp
from utils.encoders import GCEncoder, encoder_modules


class GCIVLDualAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.
    Uses a dual goal representation.

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
    
    def huber_loss(self, x, delta=1.0):
        abs_x = jnp.abs(x)
        quadratic = jnp.minimum(abs_x, delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic**2 + delta * linear

    def compute_grad_norm(self, network_key, s, g, params):
        """
        Compute ||grad V|| w.r.t state for either 'value' or 'rep_value'.
        """
        def value_sum(obs, goal, p):
            # Select the correct network head
            if network_key == 'rep_value':
                # Returns single value v
                v = self.network.select(network_key)(obs, goal, params=p)
                return (-v).mean() # Distance = -V
            else:
                # Returns ensemble (v1, v2) - we just use v1 for grad norm
                v1, _ = self.network.select(network_key)(obs, goal, params=p)
                return (-v1).mean()

        # Compute gradient w.r.t observations (arg 0)
        grad_fn = jax.vmap(jax.grad(lambda _s, _g, _p: value_sum(_s[None], _g[None], _p).squeeze()), in_axes=(0, 0, None))
        grads = grad_fn(s, g, params)
        norm = jnp.linalg.norm(grads + 1e-8, axis=-1)
        return norm

    @jax.jit
    def distance_grad_s(self, obs, goals, grad_params):
        """
        Baseline Helper: Compute the exact gradient norm of the value function.
        Used for verification only.
        """
        def value_sum(s, g, params):
            # We want grad(-V). Since V is negative distance, -V is positive distance.
            v1, v2 = self.network.select('value')(s, g, params=params)
            return (-v1).mean(), (-v2).mean()

        # Compute gradients w.r.t state (argnums=0)
        # We use vmap to handle the batch dimension correctly
        grad_fn1 = jax.vmap(jax.grad(lambda s, g, p: -self.network.select('value')(s[None], g[None], params=p)[0].squeeze()), in_axes=(0, 0, None))
        grad_fn2 = jax.vmap(jax.grad(lambda s, g, p: -self.network.select('value')(s[None], g[None], params=p)[1].squeeze()), in_axes=(0, 0, None))
        
        g1 = grad_fn1(obs, goals, grad_params)
        g2 = grad_fn2(obs, goals, grad_params)
        return g1, g2
    
    @jax.jit
    def rep_distance_grad_s(self, obs, goals, grad_params):
        """
        Compute gradient of Rep Value w.r.t State.
        Fixes the '(2,)' error by indexing the first head of the ensemble.
        """
        def get_v1_value(s, g, p):
            # 1. Forward Pass
            # rep_value returns (v1, v2) or stacked [v1, v2]
            v_out = self.network.select('rep_value')(s, g, params=p)
            
            # 2. Extract Head 1
            # If it's a tuple/list (v1, v2)
            if isinstance(v_out, (tuple, list)):
                v1 = v_out[0]
            # If it's a stacked array (2, ...)
            elif hasattr(v_out, 'shape') and v_out.shape[0] == 2:
                v1 = v_out[0]
            else:
                v1 = v_out # Already scalar
            
            # 3. Return Scalar (Distance = -Value)
            return (-v1).squeeze()

        # Compute gradients w.r.t state (argnums=0)
        # We use vmap to handle the batch dimension correctly
        grad_fn1 = jax.vmap(jax.grad(lambda s, g, p: get_v1_value(s[None], g[None], p)), in_axes=(0, 0, None))
        
        g1 = grad_fn1(obs, goals, grad_params)
        return g1

    def rep_loss(self, batch, grad_params):
        """Compute the IQL loss for the representation value function.

        The value function is parameterized by the representation, while the critic is a standard, unrestricted (MLP) critic function.
        """

        # Rep value loss.
        q1, q2 = self.network.select('target_rep_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss.
        next_v = self.network.select('rep_value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        total_loss = value_loss + critic_loss
        logs = {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
        return total_loss, logs

    def stochastic_fk_loss(self, batch, grad_params, key):
        """
        Calculates the Viscous Geometric Regularization via Fixed-Point Iteration.
        Implements the update: V(s) <- lambda * log( E[ exp(V(s')/lambda) ] )
        
        This minimizes || V(s) - T_nu V(s) ||^2, ensuring V(s) evolves according 
        to the viscous HJB operator without 'collusion' from neighbors.
        """
        # 1. Hyperparameters
        nu = self.config.get('viscous_scale', 0.01)
        # To satisfy the Taylor expansion V + nu*Laplacian + 0.5*|grad|^2:
        # We need sigma^2 = 2*nu and lambda = sigma^2 (so lambda = 2*nu).
        # Adjust per your specific PDE derivation if different.
        #lambda_temp = self.config.get('temperature', 2.0 * nu)
        lambda_temp = 1.0
        sigma = jnp.sqrt(2.0 * nu)
        K = self.config.get('num_walks', 10)
        kappa = 0.1
        
        # 2. Setup Inputs
        obs = batch['observations']
        local_speed = batch['speed']
        if local_speed.ndim == 1: local_speed = local_speed[:, None]
        goal_reps = self.network.select('rep_value')(batch['value_goals'])
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

        # 4. Generate Neighbors (s + evlon)
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


        dist_to_neighbors = jnp.linalg.norm(noise, axis=-1)
        dist_to_neighbors = jax.lax.stop_gradient(dist_to_neighbors) # Treat as fixed constant
        greens_contribution = 1 / (dist_to_neighbors + 1e-6)
        
        # Absolute difference in value
        abs_diff = jnp.abs(v_neighbors - v_s)#/jnp.linalg.norm(v_s + 1e-6)
        const_v = jax.lax.stop_gradient(v_s)
        
        # Estimate the denominator using Triangle Inequality \v
        cost_to_neighbor = abs_diff * greens_contribution
        
        #cost_to_neighbor = abs_diff
        
        # 2. Compute Allowed Slope (Riemannian Speed Limit)
        # Near walls (speed ~ 0.1) -> Limit ~ 10.0 (Steep Cliff Allowed)
        # Open space (speed ~ 1.0) -> Limit ~ 1.0 (Smooth Slope enforced)
        q_s = kappa / (local_speed + 1e-6)
        
        delta_t = 1.0

        metric_residual = jnp.maximum(0.0, cost_to_neighbor - q_s * delta_t)
        loss_metric = jnp.square(metric_residual).mean()
        use_metric = self.config.get('enable_viscous_metric', True)
        use_metric_only = self.config.get('use_metric_only', False)

        # --- Combine ---
        total_loss = loss_metric
        
        return total_loss, {
            'fk_loss': loss_metric,
            'v_mean': v_s.mean(),
            'boundary_loss': loss_metric.mean(),
            'avg_speed': local_speed.mean(),
            'avg_grad_est': cost_to_neighbor.mean()
        }

    def value_loss(self, batch, grad_params):
        """Viscous IQL Value Loss (MLP Driver)."""
        
        # Pre-compute Goal Reps (Fixed)
        goal_reps = self.network.select('rep_value')(batch['value_goals'])
        
        # --- 1. STANDARD IQL ---
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], goal_reps)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], goal_reps)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        
        (v1, v2) = self.network.select('value')(batch['observations'], goal_reps, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        total_loss = value_loss1 + value_loss2
        
        logs = {'value_loss': total_loss, 'v_mean': v.mean(), 'resistance': jnp.abs(v1 - v2).mean()}

        # --- 2. PHYSICS REGULARIZATION ---
        # --- 2. PHYSICS REGULARIZATION (Lipschitz / Viscous HJB Surrogate) ---
        if self.config.get('enable_fk_regularization', True):
            v_loss, v_loss_dict = self.stochastic_fk_loss(batch, grad_params, self.rng)
            total_loss += v_loss
            logs.update(**v_loss_dict)
            return total_loss, logs
        return total_loss, logs
            
    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss."""
        goal_reps = self.network.select('rep_value')(batch['actor_goals'])

        v1, v2 = self.network.select('value')(batch['observations'], goal_reps)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goal_reps)
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

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        loss = value_loss + actor_loss + rep_loss

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
        self.target_update(new_network, 'rep_critic')

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
        goal_reps = self.network.select('rep_value')(goals)
        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        actions = dist.sample(seed=seed)
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
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )

        rep_value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )

        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            rep_critic=(rep_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_observations, ex_actions)),
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
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcivl_dual',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            rep_expectile=0.7,  # IQL expectile for learning the representation value function.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type='bilinear',  # Parameterization of the dual goal representation (see utils/dual.py).
            # Dataset hyperparameters.
            encoder=ml_collections.config_dict.placeholder(str),
            enable_fk_regularization = True,
            enable_viscous_metric = True,
            use_metric_only = False,
            num_walks = 10,  # Number of stochastic walks for viscous regularization.
            viscous_scale = 0.001,  # Viscous scale (nu)
            dataset_class='GCDataset',  # Dataset class name.
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
