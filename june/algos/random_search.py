import chex
import jax
from flax import struct
from june import algos
from june.algos import Algorithm, AlgorithmState, AlgorithmParams
from june.utils.param_space import ParamSpace
from copy import deepcopy
from typing import Any, Dict, Union, Callable
import jax.numpy as jnp

@struct.dataclass
class RandomSearchParams(AlgorithmParams):
    search_space: ParamSpace

@struct.dataclass
class RandomSearchState(AlgorithmState):
    params: chex.ArrayTree
    states: AlgorithmState

class RandomSearch(Algorithm):
    algo: Algorithm
    num_trials: int
    num_steps: int
    eval_callback: Callable
    
    def __init__(
        self,
        algorithm: Union[Algorithm | str],
        num_trials: int,
        num_steps: int = 1,
    ):
        if not isinstance(algorithm, Algorithm):
            if isinstance(algorithm, dict):
                algorithm = algos.make(**algorithm)
            else:
                raise ValueError("algorithm must be an instance of Algorithm or a dictionary")
        
        # if eval_callback is None:
        if True:
            def eval_callback(evaluations):
                pass # by default, don't do anything with the evaluations
            self.eval_callback = eval_callback
                
        self.algo = algorithm
        self.num_trials = num_trials
        self.num_steps = num_steps
        
    @property
    def default_params(self):
        return RandomSearchParams(search_space=ParamSpace(self.algo.default_params))
    
    def init_state_impl(self, rng: chex.PRNGKey, params: AlgorithmParams) -> chex.ArrayTree:
        rng, rng_params, rng_init = jax.random.split(rng, 3)
        
        rngs = jax.random.split(rng_params, self.num_trials)
        pop_params = jax.vmap(params.search_space.sample)(rngs)
        
        rngs = jax.random.split(rng_init, self.num_trials)
        states = jax.vmap(self.algo.init_state)(rngs, pop_params.value)
        
        return RandomSearchState(
            rng=rng,
            params=pop_params,
            states=states,
        )
    
    def train_impl(self, algo_state: AlgorithmState, params: AlgorithmParams) -> chex.ArrayTree:
        def step(algo_state, _):
            states, evaluations = jax.vmap(self.algo.train)(algo_state.states, algo_state.params.value)
            self.eval_callback(evaluations)
            return algo_state.replace(states=states), evaluations
        return jax.lax.scan(step, algo_state, None, length=self.num_steps)
    
# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='random_search', entry_point=module_path + ':RandomSearch')
# ==================