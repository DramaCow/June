import jax
from flax import struct
import chex

"""
This indirection allows for things like stateful mutations.
"""

@struct.dataclass
class ParamState:
    pass
    # how to say attribute value is required (maybe as a property?)

@struct.dataclass
class CategoricalState(ParamState):
    value: chex.Array
    index: int
    
@struct.dataclass
class IntegerState(ParamState):
    value: int

@struct.dataclass
class FloatState(ParamState):
    value: float
    
@struct.dataclass
class ParamTreeState(ParamState):
    params: chex.ArrayTree
    
    @property
    def value(self):
        def get_value_at_leaf(leaf):
            if isinstance(leaf, ParamState):
                return leaf.value
            return leaf
    
        def is_leaf(x):
            return isinstance(x, ParamState)
        
        return jax.tree.map(lambda param: get_value_at_leaf(param), self.params, is_leaf=is_leaf)