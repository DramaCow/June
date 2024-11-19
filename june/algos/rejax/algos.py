import chex
from flax import struct, core
from rejax.algos import PPO, DQN, SAC, TD3, PQN, IQN

from .train_state import TrainStateWithOptArgs
from .transform import scale_by_dynamic_learning_rate

class RejaxPPOWrapper(PPO):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        actor_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.actor_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        critic_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.critic_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(actor_ts=actor_ts, critic_ts=critic_ts)
    
    def train(self, train_state):
        actor_ts = train_state.actor_ts.replace(opt_params={"learning_rate": self.learning_rate})
        critic_ts = train_state.critic_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(actor_ts=actor_ts, critic_ts=critic_ts)
        return super().train(train_state=train_state)
    
class RejaxDQNWrapper(DQN):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        q_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.q_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(q_ts=q_ts)
    
    def train(self, train_state):
        q_ts = train_state.q_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(q_ts=q_ts)
        return super().train(train_state=train_state)
    
class RejaxSACWrapper(SAC):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        actor_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.actor_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        critic_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.critic_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        alpha_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.alpha_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(actor_ts=actor_ts, critic_ts=critic_ts, alpha_ts=alpha_ts)
    
    def train(self, train_state):
        actor_ts = train_state.actor_ts.replace(opt_params={"learning_rate": self.learning_rate})
        critic_ts = train_state.critic_ts.replace(opt_params={"learning_rate": self.learning_rate})
        alpha_ts = train_state.alpha_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(actor_ts=actor_ts, critic_ts=critic_ts, alpha_ts=alpha_ts)
        return super().train(train_state=train_state)
    
class RejaxTD3Wrapper(TD3):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        actor_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.actor_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        critic_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.critic_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(actor_ts=actor_ts, critic_ts=critic_ts)
    
    def train(self, train_state):
        actor_ts = train_state.actor_ts.replace(opt_params={"learning_rate": self.learning_rate})
        critic_ts = train_state.critic_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(actor_ts=actor_ts, critic_ts=critic_ts)
        return super().train(train_state=train_state)
    
class RejaxPQNWrapper(PQN):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        q_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.q_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(q_ts=q_ts)
    
    def train(self, train_state):
        q_ts = train_state.q_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(q_ts=q_ts)
        return super().train(train_state=train_state)
    
class RejaxIQNWrapper(IQN):
    def init_state(self, rng: chex.PRNGKey):
        state = super().init_state(rng)
        q_ts = TrainStateWithOptArgs.create(apply_fn=(), params=state.q_ts.params, tx=scale_by_dynamic_learning_rate(), opt_params={"learning_rate": self.learning_rate})
        return state.replace(q_ts=q_ts)
    
    def train(self, train_state):
        q_ts = train_state.q_ts.replace(opt_params={"learning_rate": self.learning_rate})
        train_state.replace(q_ts=q_ts)
        return super().train(train_state=train_state)