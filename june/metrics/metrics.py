import jax.numpy as jnp
from flax import struct

@struct.dataclass
class EpisodeMetrics:
    sum_returns: float = 0.
    sum_lengths: int = 0
    episode_count: int = 0
    
    def update(self, returned_episode_returns, returned_episode_lengths, dones):
        # For now, we 
        sum_returns = self.sum_returns + jnp.where(dones, returned_episode_returns, 0.).sum()
        sum_lengths = self.sum_lengths + jnp.where(dones, returned_episode_lengths, 0).sum()
        episode_count = self.episode_count + dones.sum()
        return EpisodeMetrics(
            sum_returns=sum_returns,
            sum_lengths=sum_lengths,
            episode_count=episode_count,
        )
        
    def get_metrics(self):
        return {
            "mean_return": jnp.where(self.episode_count > 0, self.sum_returns / self.episode_count, 0),
            "mean_episode_length": jnp.where(self.episode_count > 0, self.sum_lengths / self.episode_count, 0),
            "episode_count": self.episode_count,
        }