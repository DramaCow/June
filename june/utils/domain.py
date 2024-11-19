from __future__ import annotations # for forward referencing (Sampler/Domain)

import jax
import jax.numpy as jnp
import chex
from typing import List, Sequence, Optional, Type, Any
from copy import copy
from flax import struct
from dataclasses import dataclass

from june.utils.param_state import ParamTreeState, ParamState, CategoricalState, IntegerState, FloatState

class Sampler:
    def sample(self, rng: chex.PRNGKey, domain: Domain):
        raise NotImplementedError
    
# TODO: I'm not so fond of this... It feels like search space responsibilities
#       will get bloated. On the other hand, specifying mutations outside the 
#       search space raises questions about how to handle the case where the
#       mutations are not applied. More thought required.
class Mutator:
    def mutate(self, rng: chex.PRNGKey, x: Any, domain: Domain):
        raise NotImplementedError

class Domain:
    _default_sampler_cls = None
    _default_mutator_cls = None
    
    sampler: Optional[Sampler] = None
    mutator: Optional[Sampler] = None
    
    def sample(self, rng: chex.PRNGKey) -> ParamTree:
        return self.sampler.sample(rng, self)
    
    def mutate(self, rng: chex.PRNGKey, x: ParamState) -> ParamState:
        return self.mutator.mutate(rng, x, self)
    
    @property
    def is_mutable(self):
        return self.mutator is not None
    
    # Builder pattern nonsense
    
    def set_sampler(self, sampler):
        new = copy(self)
        new.sampler = sampler
        return new
    
    def set_mutator(self, mutator):
        new = copy(self)
        new.mutator = mutator
        return new
    
    def uniform(self):
        return self.set_sampler(self._default_sampler_cls())
    
    def mutable(self):
        return self.set_mutator(self._default_mutator_cls())
    
class Categorical(Domain):
    class _Uniform(Sampler):
        def sample(self, rng: chex.PRNGKey, domain: Categorical) -> CategoricalState:
            idx = jax.random.choice(rng, domain.num_categories)
            return CategoricalState(value=jax.tree.map(lambda x: x[idx], domain.categories), index=idx)
        
    class _Mutator(Mutator):
        def mutate(self, rng: chex.PRNGKey, x: CategoricalState, domain: Categorical) -> CategoricalState:
            shift = jax.random.choice(rng, jnp.array([-1, 1]))
            idx = (x.index + shift).clip(0, domain.num_categories - 1)
            value = jax.tree.map(lambda x: x[idx], domain.categories)
            return x.replace(value=value, index=idx)
            
    _default_sampler_cls = _Uniform
    _default_mutator_cls = _Mutator
    
    categories: chex.ArrayTree
    
    def __init__(self, categories: chex.ArrayTree):
        self.categories = categories
        
    @property
    def num_categories(self):
        return jax.tree_util.tree_leaves(self.categories)[0].shape[0]
    
    def __repr__(self):
        return f"{'Mutable' if self.is_mutable else ''}Categorical({self.categories})"
    
class Integer(Domain):
    class _Uniform(Sampler):
        def sample(self, rng: chex.PRNGKey, domain: Integer) -> IntegerState:
            return IntegerState(value=jax.random.randint(rng, (), domain.lower, domain.upper))
        
    class _Mutator(Mutator):
        def mutate(self, rng: chex.PRNGKey, x: IntegerState, domain: Integer) -> IntegerState:
            perturbation_factor = jax.random.choice(rng, jnp.array([0.8, 1.2]))
            return x.replace(value=jnp.int32(x.value * perturbation_factor))
        
    _default_sampler_cls = _Uniform
    _default_mutator_cls = _Mutator

    lower: int
    upper: int
    
    def __init__(self, lower: int, upper: int):
        self.lower = lower
        self.upper = upper
        
    def __repr__(self):
        return f"{'Mutable' if self.is_mutable else ''}Integer({self.lower}, {self.upper})"

class Float(Domain):
    class _Uniform(Sampler):
        def sample(self, rng: chex.PRNGKey, domain: Float) -> FloatState:
            return FloatState(value=jax.random.uniform(rng, minval=domain.lower, maxval=domain.upper))
        
    class _Mutator(Mutator):
        def mutate(self, rng: chex.PRNGKey, x: FloatState, domain: Float) -> FloatState:
            perturbation_factor = jax.random.choice(rng, jnp.array([0.8, 1.2]))
            return x.replace(value=x.value * perturbation_factor)
        
    _default_sampler_cls = _Uniform
    _default_mutator_cls = _Mutator
    
    lower: float = -jnp.inf
    upper: float = jnp.inf
    
    def __init__(self, lower: Optional[float], upper: Optional[float]):
        self.lower = -jnp.inf if lower is None else lower
        self.upper = -jnp.inf if upper is None else upper
        
    def __repr__(self):
        return f"{'Mutable' if self.is_mutable else ''}Float({self.lower}, {self.upper})"
    
# Register each domain as a leaf node in pytrees (is there a nicer way of doing this?)
    
jax.tree_util.register_pytree_node(
    Categorical,
    lambda f: ((), (f.categories, f.sampler, f.mutator)),
    lambda aux_data, _: Categorical(categories=aux_data[0]).set_sampler(aux_data[1]).set_mutator(aux_data[2]),
)

jax.tree_util.register_pytree_node(
    Float,
    lambda f: ((), (f.lower, f.upper, f.sampler, f.mutator)),
    lambda aux_data, _: Float(lower=aux_data[0], upper=aux_data[1]).set_sampler(aux_data[2]).set_mutator(aux_data[3]),
)

jax.tree_util.register_pytree_node(
    Integer,
    lambda f: ((), (f.lower, f.upper, f.sampler, f.mutator)),
    lambda aux_data, _: Integer(lower=aux_data[0], upper=aux_data[1]).set_sampler(aux_data[2]).set_mutator(aux_data[3]),
)

# Ultimately, the below is what should be used

def choice(categories: chex.ArrayTree):
    return Categorical(categories=categories).uniform()

def uniform(lower: float, upper: float):
    return Float(lower=lower, upper=upper).uniform()

def randint(lower: int, upper: int):
    return Integer(lower=lower, upper=upper).uniform()

if __name__=="__main__":
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 100)
    
    a = Categorical(jnp.array([1, 2, 3])).uniform()
    # a = Float(0, 1)
    # a = Integer(0, 10)
    print(jax.vmap(a.sample)(rngs))