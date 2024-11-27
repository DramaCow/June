import jax
import jax.numpy as jnp
import chex
from flax import struct

@struct.dataclass
class Storage:
    buffer: chex.ArrayTree
    size: int
    
    @classmethod
    def create(cls, pholder_item, capacity):
        buffer = jax.tree.map(lambda x: jnp.empty_like(x)[None, ...].repeat(capacity, axis=0), pholder_item)
        return cls(buffer=buffer, size=0)

    def insert(self, item):
        new_buffer = jax.tree.map(lambda x, y: x.at[self.size].set(y), self.buffer, item)
        return self.replace(buffer=new_buffer, size=self.size + 1)

    def extend(self, items):
        def insert_once(storage, item):
            return storage.insert(item), None
        return jax.lax.scan(insert_once, self, items)[0]

    def get(self, index):
        return jax.tree.map(lambda x: x[index], self.buffer)