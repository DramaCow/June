import chex
import jax
from flax import struct
from june import algos
from june.algos import Algorithm, AlgorithmState, AlgorithmParams
from june.utils.param_space import ParamSpace
from copy import deepcopy
import jax.numpy as jnp
import numpy as np

from typing import Any, Tuple

import jax
from flax import struct
from jax import numpy as jnp

@struct.dataclass
class PBTParams(AlgorithmParams):
    search_space: ParamSpace
    truncation_selection: float = 0.25 # proportion of population to exploit
    resample_prob: float = 0. # probability of resampling hyperparameters on explore

@struct.dataclass
class PBTState(AlgorithmState):
    states: chex.ArrayTree
    params: chex.ArrayTree
    fitness: chex.Array
    gen_ids: chex.Array
    gen_count: int
    timestep_count: int = 0
    
    @property
    def best_fitness(self):
        best_id = self.fitness.argmax()
        return jax.tree.map(lambda x: x[best_id], self.fitness)
    
    @property
    def best_train_state(self):
        best_id = self.fitness.argmax()
        return jax.tree.map(lambda x: x[best_id], self.train_states)
    
    @property
    def best_hyperparams(self):
        best_id = self.fitness.argmax()
        return jax.tree.map(lambda x: x[best_id], self.hyperparams)
    

class PBT(Algorithm):
    algo: Algorithm
    pop_size: int = 10
    num_steps: int = 1000
    
    def __init__(
        self,
        algorithm,
        pop_size,
        num_steps,
    ):
        if not isinstance(algorithm, Algorithm):
            if isinstance(algorithm, dict):
                algorithm = algos.make(**algorithm)
            else:
                raise ValueError("algorithm must be an instance of Algorithm or a dictionary")
        
        self.algo = algorithm
        self.pop_size = pop_size
        self.num_steps = num_steps
        
    @property
    def default_params(self):
        # NOTE: by default, explore will not perform any mutations
        return PBTParams(search_space=self.algo.default_params)
    
    def init_state_impl(self, rng: chex.PRNGKey, params: PBTParams) -> PBTState:
        # jax.debug.print("{}", params)
        rng, rng_params, rng_init = jax.random.split(rng, 3)
    
        rngs = jax.random.split(rng_params, self.pop_size)
        pop_params = jax.vmap(params.search_space.sample)(rngs)
        
        rngs = jax.random.split(rng_init, self.pop_size)
        states = jax.vmap(self.algo.init_state)(rngs, pop_params.value)
        
        return PBTState(
            rng=rng,
            params=pop_params,
            states=states,
            # ===
            fitness=jnp.full(self.pop_size, -jnp.inf),
            gen_ids=jnp.zeros(self.pop_size, dtype=jnp.int32),
            gen_count=0,
        )
    
    def train_impl(self, algo_state: PBTState, params: PBTParams) -> Tuple[PBTState, Any]:
        return jax.lax.scan(lambda state, _: self.train_generation(state, params), algo_state, None, length=self.num_steps)
    
    # === EVERYTHING BELOW IS INTERNAL BUT CAN BE OVERLOADED ===
    
    def train_generation(self, algo_state, params):
        # jax.debug.print("Before: {}", algo_state.params.value)
        
        # exploit
        algo_state, (copy_ids, exploit_mask) = self.exploit(algo_state, params)
        cand_states, cand_params = jax.tree.map(lambda x: x[copy_ids], (algo_state.states, algo_state.params))
        
        # explore
        algo_state, cand_params = self.explore(algo_state, cand_params, exploit_mask, params)
        
        # we need to reset the rngs for each algorithm to avoid deterministic behaviour
        rng, algo_state = algo_state.get_rng()
        cand_states = cand_states.replace(rng=jax.random.split(rng, self.pop_size))
        
        # train
        cand_states, cand_evaluations = jax.vmap(self.algo.train)(cand_states, cand_params.value)
        cand_fitness = cand_evaluations.reshape(self.pop_size, -1).mean(axis=-1)
        
        # jax.debug.print("Intermediate: {}", cand_params.value)
        
        algo_state = self.update_archive(algo_state, cand_states, cand_params, cand_fitness, params)
        
        # jax.debug.print("After: {}", algo_state.params.value)
        
        return algo_state, cand_evaluations
    
    def exploit(self, algo_state, params):
        rng, algo_state = algo_state.get_rng()
        
        k = (self.pop_size * params.truncation_selection).astype(jnp.int32).clip(1, self.pop_size)
        kth_best_fitness = algo_state.fitness.sort(descending=True)[k-1]
        
        exploit_mask = algo_state.fitness < kth_best_fitness
        copy_ids = jax.random.choice(rng, jnp.arange(self.pop_size), p=~exploit_mask, shape=exploit_mask.shape)
        copy_ids = jnp.where(exploit_mask, copy_ids, jnp.arange(self.pop_size))
        
        return algo_state, (copy_ids, exploit_mask)
        
    def explore(self, algo_state, ind_params, exploit_mask, params):
        def explore_single(rng, ind_params, exploit_flag):
            mut_params = self.mutate_params(rng, ind_params, params)
            return jax.tree.map(lambda x, y: jax.lax.select(exploit_flag, x, y), mut_params, ind_params)
        
        rng, algo_state = algo_state.get_rng()
        rngs = jax.random.split(rng, self.pop_size)
        cand_params = jax.vmap(explore_single)(rngs, ind_params, exploit_mask)
        
        return algo_state, cand_params
        
    def mutate_params(self, rng, ind_params, params):
        # NOTE: it could be reasonable to move the responsibility
        #       of resampling into the per-parameter mutation function,
        #       which would allow customization of the per-parameter
        #       resampling probabilities. Counter point is this would
        #       introduce extra complexity for the user. For now,
        #       we will keep the resampling probability as a "global".
    
        def resample(domain):
            def on_resample(rng, x):
                rng1, rng2 = jax.random.split(rng)
                y = domain.sample(rng2)
                return jax.tree.map(lambda a, b: jax.lax.select(jax.random.uniform(rng1) < params.resample_prob, a, b), y, x)
            return on_resample

        rng_mutate, rng_resample = jax.random.split(rng)
        mut_params = params.search_space.mutate(rng_mutate, ind_params)
        
        return params.search_space.domain_apply(
            resample,
            rng_resample,
            mut_params.params,
            cond_fn=lambda domain: domain.is_mutable
        )
        
    def update_archive(self, algo_state, cand_states, cand_params, cand_fitness, params=None):
        # take the new states and params, concatenate them with the old ones, and pick the best
        # TODO: verify if this is correct. Maybe we should just completely replace the old
        #       individuals with the new candidates?
        
        cand_gen_ids = jnp.ones_like(algo_state.gen_ids) * (algo_state.gen_count + 1)
        cand_states, cand_params, cand_fitness, cand_gen_ids = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            (algo_state.states, algo_state.params, algo_state.fitness, algo_state.gen_ids),
            (cand_states, cand_params, cand_fitness, cand_gen_ids),
        )
        
        elite_ids = jnp.argsort(cand_fitness, descending=True)[:self.pop_size]
        
        ind_states, ind_params, ind_fitness, ind_gen_ids = jax.tree.map(
            lambda x: x[elite_ids],
            (cand_states, cand_params, cand_fitness, cand_gen_ids),
        )
        
        return algo_state.replace(
            states=ind_states,
            params=ind_params,
            fitness=ind_fitness,
            gen_ids=ind_gen_ids,
            gen_count=algo_state.gen_count + 1
        )

# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='pbt', entry_point=module_path + ':PBT')
# ==================