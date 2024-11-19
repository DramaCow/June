from flax import struct
from gymnax.environments.environment import Environment
from typing import Any, Callable
import chex
from typing import Optional
from functools import partial
import jax
from gymnax.environments import environment
from copy import deepcopy
from typing import Union

from rejax.evaluate import evaluate

# Algo should have clear separation of:
# - env config (env, env_params, total_timesteps etc.)
# - algo config (hyperparameters, network architecture etc.)
# - eval config (eval_callback, eval_freq etc.)
# Should be able to swap these components at runtime (subscriber pattern?)
# Definitely shouild be compositional, not inheritable
# NOTE: wouldn't eval depend on the other 2? How does that work? 

@struct.dataclass
class AlgorithmParams:
    """i.e. hyperparams"""
    pass

@struct.dataclass
class AlgorithmState:
    """An all inclusive encapsulation of the current state of training (including rngs)"""
    rng: chex.PRNGKey
    
    def get_rng(self):
        rng = self.rng
        rng, _rng = jax.random.split(rng)
        return rng, self.replace(rng=_rng) # TODO: prefer the output order to be the other way around for some reason

class Algorithm:
    def set_resource(self, resource: Union[float, int]):
        # for wrappers controlling resource allocation
        raise NotImplementedError
    
    @property
    def default_params(self):
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, rng: chex.PRNGKey, params: Optional[AlgorithmParams]) -> chex.ArrayTree:
        if params is None:
            params = self.default_params
        return self.init_state_impl(rng, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def train(self, algo_state: AlgorithmState, params: Optional[AlgorithmParams]):
        if params is None:
            params = self.default_params
        return self.train_impl(algo_state, params)
    
    def init_state_impl(self, rng: chex.PRNGKey, params: AlgorithmParams) -> chex.ArrayTree:
        raise NotImplementedError
    
    def train_impl(self, algo_state: AlgorithmState, params: AlgorithmParams) -> chex.ArrayTree:
        raise NotImplementedError
    
# @struct.dataclass
# class Algorithm:
    
    