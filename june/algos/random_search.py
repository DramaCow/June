import chex
import jax
from flax import struct
from june import algos
from june.algos import Algorithm, AlgorithmState, AlgorithmParams
from june.utils.param_space import ParamSpace
from copy import deepcopy
from typing import Any, Dict, Union, Callable
import jax.numpy as jnp
from june.utils import Storage

@struct.dataclass
class RandomSearchParams(AlgorithmParams):
    search_space: ParamSpace

@struct.dataclass
class RandomSearchState(AlgorithmState):
    params: chex.ArrayTree
    states: AlgorithmState
    storage: Storage

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

        storage = Storage.create({
            "state": jax.tree.map(lambda x: x[0], states),
            "params": jax.tree.map(lambda x: x[0], pop_params.value),
            "fitness": 0., 
        }, self.num_trials * self.num_steps)
        
        return RandomSearchState(
            rng=rng,
            params=pop_params,
            states=states,
            storage=storage,
        )
    
    def train_impl(self, algo_state: AlgorithmState, params: AlgorithmParams) -> chex.ArrayTree:
        def step(algo_state, _):
            states, evaluations = jax.vmap(self.algo.train)(algo_state.states, algo_state.params.value)
            fitness = jax.vmap(self.algo.get_fitness)(algo_state.states, algo_state.params.value, evaluations)
            storage = algo_state.storage.extend({"state": algo_state.states, "params": algo_state.params.value, "fitness": fitness})
            self.eval_callback(evaluations)
            return algo_state.replace(states=states, storage=storage), evaluations
        return jax.lax.scan(step, algo_state, None, length=self.num_steps)

    def get_best(self, algo_state, algo_params, evaluations):
        algo = self.algo

        evals = []
        for agent_id, algo_idx in enumerate(algo_state.params.value.index):
            evaluation = evaluations[algo_idx]
            evals.append(evaluation[-1, agent_id].mean())
            
        index = jnp.array(evals).argmax()
        state, params = jax.tree.map(lambda x: x[index], (algo_state.states, algo_state.params.value))
        return algo, state, params
    
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