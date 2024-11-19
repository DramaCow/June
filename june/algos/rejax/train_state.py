from typing import Any, Callable, Union
from flax.training.train_state import TrainState as BaseTrainState
import jax
from flax import core, struct
import optax

class TrainStateWithOptArgs(BaseTrainState):
    """
    Like regular Flax TrainState, but also accepts additional optimizer params.
    """
    opt_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    
    def apply_gradients(self, *, grads):
        grads_with_opt = grads
        params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, **self.opt_params,
        )
        
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)
        new_params = new_params_with_opt
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )