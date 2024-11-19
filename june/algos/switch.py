from rejax import get_algo
from copy import deepcopy
from typing import List, Callable
import chex
import jax
from flax import struct
from typing import Any
from june.algos.registration import make
from june.algos import Algorithm, AlgorithmState, AlgorithmParams
import jax.numpy as jnp

# Wrapper for selecting from different algorithms at runtime.
# NOTE: this is a bit of a mess, and I'm not sure how to handle this.
# - we are going to have to basically run all algorithms in parallel and mask out which one we are using?
# - are we going to run into compute constraints doing this?

@struct.dataclass
class SwitchParams(AlgorithmParams):
    params: List[AlgorithmParams]
    index: int

@struct.dataclass
class SwitchState(AlgorithmState):
    states: List[chex.ArrayTree]

class Switch(Algorithm):
    algos: List[Algorithm]
    
    def __init__(self, algorithms: List[Algorithm]):
        algos = []
        for algorithm in algorithms:
            if not isinstance(algorithm, Algorithm):
                if isinstance(algorithm, dict):
                    algorithm = make(**algorithm)
                else:
                    raise ValueError("algorithm must be an instance of Algorithm or a dictionary")
            algos.append(algorithm)
        self.algos = algos
        
    @property
    def default_params(self):
        return SwitchParams(params=[algo.default_params for algo in self.algos], index=0)
        
    def init_state_impl(self, rng: chex.PRNGKey, params: SwitchParams):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, len(self.algos))
        states = list(map(lambda algo, rng, params: algo.init_state(rng, params), self.algos, rngs, params.params))
        return SwitchState(rng=rng, states=states)
    
    def train_impl(self, algo_state: SwitchState, params: SwitchParams):
        states, evaluation = jax.lax.switch(params.index, [self.make_train_branch(i) for i in range(len(self.algos))], algo_state, params)
        return algo_state.replace(states=states), evaluation
    
    def make_train_branch(self, index: int):        
        def train_fn(algo_state, params):
            state, evaluation = self.algos[index].train(algo_state.states[index], params.params[index])
            
            states = algo_state.states
            states[index] = state
            
            evaluations = self._pholder_evaluations(algo_state.states, params)
            evaluations[index] = evaluation
            
            return states, evaluations
        return train_fn
    
    def _pholder_evaluations(self, states, params):
        # does this get jitted out properly?
        evaluations = []
        for ph_state, ph_params, algo in zip(states, params.params, self.algos):
            _, eval_shape = jax.eval_shape(algo.train, ph_state, ph_params)
            ph_evaluation = jax.tree.map(lambda x: jnp.empty(x.shape, dtype=x.dtype), eval_shape)
            evaluations.append(ph_evaluation)
        return evaluations
        
# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='switch', entry_point=module_path + ':Switch')
# ==================