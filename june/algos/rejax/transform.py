from typing import Any, Optional
import jax
import optax

def scale_by_dynamic_learning_rate(*, flip_sign: bool = True) -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        del params # why?
        return optax.EmptyState()
    
    m = -1 if flip_sign else 1
    def update_fn(updates, state, params=None, **extra_args):
        del params # why?
        updates = jax.tree.map(lambda g: g * extra_args["learning_rate"] * m, updates)
        return updates, state
    
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def adam_with_dynamic_learning_rate(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False
) -> optax.GradientTransformationExtraArgs:
    return optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        scale_by_dynamic_learning_rate(),
    )