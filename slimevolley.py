import jax
from evojax.task.slimevolley import SlimeVolley
from gymnax.environments import environment
from gymnax.environments import spaces
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 3000

class SlimeVolleyWrapper(environment.Environment):
    def __init__(self):
        self.task = SlimeVolley(test=False, max_steps=3000)

    @property
    def default_params(self):
        return EnvParams()

    def step_env(self, rng, state, action, params):
        state, action = jax.tree.map(lambda x: x[None, ...], (state, action))
        action = jax.nn.one_hot(action, num_classes=self.num_actions)
        state, reward, done = self.task.step(state, action)
        state, reward, done = jax.tree.map(lambda x: x[0], (state, reward, done))
        return state.obs, state, reward, done, {}
    
    def reset_env(self, rng, params):
        rng = jax.tree.map(lambda x: x[None, ...], rng)
        state = self.task.reset(rng)
        state, _, _ = self.task.step(state, jnp.array([[0., 0., 0.]])) # dirty hack because JAX can't infer types (basically, Evojax initializes some values weakly)
        state = jax.tree.map(lambda x: x[0], state)
        return state.obs, state

    @property
    def num_actions(self):
        return 3

    def action_space(self, params):
        return spaces.Discrete(3)

    def observation_space(self, params):
        return spaces.Box(-1., 1., (12,), dtype=jnp.float32)

if __name__=="__main__":
    import jax
    import jax.numpy as jnp

    env = SlimeVolleyWrapper()
    env_params = env.default_params

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    obs, state, reward, done, info = env.step(rng,state, 0)
    print(obs, state, reward, done)