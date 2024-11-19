from __future__ import annotations

from typing import Callable

import jax
from flax import struct
from jax import numpy as jnp
import chex

from june.utils.domain import Domain
from june.utils.param_state import ParamTreeState

@struct.dataclass
class ParamSpace:
    params: chex.ArrayTree
    
    def domain_apply(
        self,
        f: Callable[[Domain], Callable[[chex.PRNGKey, ...], ParamSpace]],
        rng: chex.PRNGKey,
        *args: chex.ArrayTree,
        cond_fn: Optional[Callable[[Domain], bool]] = None,
    ):
        return make_domain_apply(self.params, f, cond_fn)(rng, *args)

    def sample(self, rng: chex.PRNGKey) -> ParamTreeState:
        return self.domain_apply(lambda domain: domain.sample, rng)
    
    def mutate(self, rng: chex.PRNGKey, params: ParamTreeState) -> ParamTreeState:
        return self.domain_apply(lambda domain: domain.mutate, rng, params.params, cond_fn=lambda domain: domain.is_mutable)


def make_domain_apply(
    params: ParamSpace,
    f: Callable[[Domain], Callable[[chex.PRNGKey, ...], ParamSpace]],
    cond_fn: Optional[Callable[[Domain], bool]] = None,
):
    """
    Constructs a function that given an rng and arbitrary pytrees
    each of the same shape as the ParamSpace, applies a function 
    to the leaf nodes of the ParamSpace that is a leaf (AND satisfies
    the condition function cond_fn).
    """
    
    # In principle, this can be rolled into f as required,
    # but this adds a bit of convenience. Consider removing.
    if cond_fn is None:
        cond_fn = lambda _: True
    
    def op_leaf(leaf):
        if isinstance(leaf, Domain) and cond_fn(leaf):
            return f(leaf)
        return lambda rg, *_: leaf
    
    def is_leaf(x):
        return isinstance(x, Domain) or isinstance(x, ParamSpace)
        
    tree_op = jax.tree.map(op_leaf, params, is_leaf=is_leaf)
    
    def mutate(rng, *args):
        leaves, treedef = jax.tree_util.tree_flatten(params, is_leaf=is_leaf)
        rng_keys = jax.random.split(rng, len(leaves))
        rng_pytree = jax.tree_util.tree_unflatten(treedef, rng_keys)
        return ParamTreeState(jax.tree.map(
            lambda op, rng, *xs: op(rng, *xs),
            tree_op,
            rng_pytree,
            *args,
            is_leaf=is_leaf
        ))
    
    return mutate