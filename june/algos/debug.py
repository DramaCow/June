from june.algos import Algorithm, AlgorithmState, AlgorithmParams
from flax import struct
import jax
import chex

@struct.dataclass
class DebugAlgorithmState(AlgorithmState):
    value: float
    
@struct.dataclass
class DebugAlgorithmParams(AlgorithmParams):
    increment: float

# Dummy algorithm that increments a value by a fixed amount
# used for debugging
class DebugAlgorithm(Algorithm):        
    @property
    def default_params(self):
        return DebugAlgorithmParams(increment=0.0)
    
    def init_state_impl(self, rng: chex.PRNGKey, params: AlgorithmParams) -> chex.ArrayTree:
        return DebugAlgorithmState(rng=rng, value=params.increment)
    
    def train_impl(self, algo_state: AlgorithmState, params: AlgorithmParams) -> chex.ArrayTree:
        value = algo_state.value + params.increment
        return algo_state.replace(value=value), value
    
# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='debug', entry_point=module_path + ':DebugAlgorithm')
# ==================