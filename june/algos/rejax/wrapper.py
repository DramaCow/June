from june.algos import Algorithm, AlgorithmParams, AlgorithmState
from rejax.algos import Algorithm as RejaxAlgorithm
from rejax import get_algo
from gymnax.environments import environment
import chex
from flax import struct

# ==================
# let's import all the rejax algorithm wrappers
from .algos import RejaxPPOWrapper, RejaxDQNWrapper, RejaxSACWrapper, RejaxTD3Wrapper, RejaxPQNWrapper, RejaxIQNWrapper

_REJAX_ALGOS = {
    "ppo": RejaxPPOWrapper,
    "dqn": RejaxDQNWrapper,
    "sac": RejaxSACWrapper,
    "td3": RejaxTD3Wrapper,
    "pqn": RejaxPQNWrapper,
    "iqn": RejaxIQNWrapper,
}
# ==================

# Single wrapper to rule them all
class RejaxWrapper(Algorithm):
    algo_cls: RejaxAlgorithm
    param_kwargs: dict # kwargs for creating default params
    
    def __init__(self, algorithm, param_kwargs):
        if isinstance(algorithm, str):
            # algorithm = get_algo(algorithm)
            algorithm = _REJAX_ALGOS[algorithm]
        self.algo_cls = algorithm
        self.param_kwargs = param_kwargs

    @property
    def default_params(self) -> AlgorithmParams:
        return self.algo_cls.create(**self.param_kwargs)
    
    def init_state_impl(self, rng: chex.PRNGKey, params: AlgorithmParams) -> chex.ArrayTree:
        return params.init_state(rng)
    
    def train_impl(self, algo_state: AlgorithmState, params: AlgorithmParams) -> chex.ArrayTree:
        return params.train(train_state=algo_state)

# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='rejax', entry_point=module_path + ':RejaxWrapper')
# ==================