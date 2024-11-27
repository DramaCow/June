import chex
import jax
from flax import struct
from june import algos
from june.algos import Algorithm, AlgorithmState, AlgorithmParams
from june.utils.param_space import ParamSpace
from copy import deepcopy
import jax.numpy as jnp
import numpy as np
from june.utils import Storage

@struct.dataclass
class ASHAParams(AlgorithmParams):
    search_space: ParamSpace

@struct.dataclass
class ASHAState(AlgorithmState):
    params: chex.ArrayTree
    states: AlgorithmState
    fitness: chex.Array         # fitness of each configuration at each rung
    rungs: chex.Array           # highest rung each trial has reached (initially -1)
    trial_steps: chex.Array     # how many steps has each trial taken
    worker_trial_id: chex.Array # index of which trial a worker is working on
    evaluations: chex.Array     # evaluations of each trial at each step
    # === PURELY FOR DEBUGGING ===
    step_count: int
    worker_history: chex.Array
    storage: Storage

class ASHA(Algorithm):
    algo: Algorithm
    num_workers: int
    num_steps: int
    reduction_factor: int # η
    debug: bool = True
    
    def __init__(
        self,
        algorithm: Algorithm,
        num_workers: int,
        num_steps: int,
        reduction_factor: int = 2,
        debug: bool = False,
    ):
        # TODO: warm start?
        if not isinstance(algorithm, Algorithm):
            if isinstance(algorithm, dict):
                algorithm = algos.make(**algorithm)
            else:
                raise ValueError("algorithm must be an instance of Algorithm or a dictionary")
        
        self.algo = algorithm
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.reduction_factor = reduction_factor
        self.debug = debug
        
        assert reduction_factor > 1, "reduction_factor must be greater than 1"
    
    # === inferred params ===    
    @property
    def num_rungs(self):
        return int(np.log(self.num_steps) / np.log(self.reduction_factor))
    
    @property
    def num_initial_trials(self):
        # number of trials in the first rung
        return 20
        return self.reduction_factor ** self.num_rungs
    # =======================
        
    @property
    def default_params(self):
        return ASHAParams(search_space=ParamSpace(self.algo.default_params))
    
    def init_state_impl(self, rng: chex.PRNGKey, params: ASHAParams) -> chex.ArrayTree:
        # We sample a bunch of configurations and cache them all
        rng, rng_params, rng_init = jax.random.split(rng, 3)
    
        rngs = jax.random.split(rng_params, self.num_initial_trials)
        pop_params = jax.vmap(params.search_space.sample)(rngs)
        
        rngs = jax.random.split(rng_init, self.num_initial_trials)
        states = jax.vmap(self.algo.init_state)(rngs, pop_params.value)
        
        fitness = jnp.full((self.num_rungs, self.num_initial_trials), -jnp.inf)
        rungs = jnp.full(self.num_initial_trials, -1, dtype=jnp.int32) # what rung each trial is on, -1 if not yet completed a rung
        trial_steps = jnp.zeros(self.num_initial_trials, dtype=jnp.uint32) # how many units has each trial has consumed
        
        worker_trial_id = -jnp.ones(self.num_workers, dtype=jnp.int32) # flag for which workers are currently training a configuration
        
        # make evaluations array that is ultimately returned
        evaluation_shape = jax.eval_shape(self.algo.train, *jax.tree.map(lambda x: x[0], (states, pop_params.value)))[1]
        evaluations = jax.tree.map(lambda x: jnp.empty((self.num_initial_trials, self.num_steps, *x.shape), dtype=x.dtype), evaluation_shape)

        storage = Storage.create({
            "state": jax.tree.map(lambda x: x[0], states),
            "params": jax.tree.map(lambda x: x[0], pop_params.value),
            "fitness": 0., 
        }, self.num_workers * self.num_steps)
        
        return ASHAState(
            rng=rng,
            params=pop_params,
            states=states,
            # ===
            fitness=fitness,
            rungs=rungs,
            trial_steps=trial_steps,
            worker_trial_id=worker_trial_id,
            evaluations=evaluations,
            # === PURELY FOR DEBUGGING ===
            step_count=0,
            worker_history=jnp.full((self.num_steps, self.num_workers), -1, dtype=jnp.int32),
            storage=storage,
        )

    def train_impl(self, algo_state: ASHAState, params: ASHAParams) -> chex.ArrayTree:
        def step(algo_state, _):
            inds, promoted = self.get_jobs(algo_state)
            
            if self.debug:
                algo_state = algo_state.replace(worker_history=algo_state.worker_history.at[algo_state.step_count].set(inds))
                jax.debug.callback(self.debug_print_workers, algo_state)
                
            # train step
            worker_states, worker_params = jax.tree.map(lambda x: x[inds], (algo_state.states, algo_state.params))
            worker_states, worker_evals = jax.vmap(self.algo.train)(worker_states, worker_params.value)
            states = jax.tree.map(lambda x, y: x.at[inds].set(y), algo_state.states, worker_states)
            
            # record evaluations
            evaluations = jax.tree.map(lambda x, y: x.at[inds, algo_state.trial_steps[inds]].set(y), algo_state.evaluations, worker_evals)
            
            # update rungs and trial steps
            rungs = algo_state.rungs[inds] + promoted
            trial_steps = algo_state.trial_steps[inds] + 1
            # rungs = algo_state.rungs[inds] + (trial_steps == self._steps_required(algo_state.rungs[inds]))

            worker_fitness = jax.vmap(self.algo.get_fitness)(worker_states, worker_params.value, worker_evals)
            storage = algo_state.storage.extend({"state": worker_states, "params": worker_params.value, "fitness": worker_fitness})
            
            # if trial has reached the required steps, then free the worker
            worker_free = trial_steps == self._steps_required(rungs)
            worker_fitness = jnp.where(worker_free, worker_fitness, -jnp.inf)
            worker_trial_id = jnp.where(worker_free, -1, inds)
            
            fitness = algo_state.fitness.at[rungs, inds].set(worker_fitness)
            rungs = algo_state.rungs.at[inds].set(rungs)
            trial_steps = algo_state.trial_steps.at[inds].set(trial_steps)
            
            algo_state = algo_state.replace(
                states=states,
                fitness=fitness,
                rungs=rungs,
                trial_steps=trial_steps,
                worker_trial_id=worker_trial_id,
                step_count=algo_state.step_count+1,
                evaluations=evaluations,
                storage=storage,
            )
            
            # self.eval_callback(evaluations)
            return algo_state, None
        algo_state, _ = jax.lax.scan(step, algo_state, None, length=self.num_steps)
        return algo_state, algo_state.evaluations

    def get_best(self, algo_state, algo_params):
        algo = self.algo
        index = algo_state.trial_steps.argmax()
        state, params = jax.tree.map(lambda x: x[index], (algo_state.states, algo_state.params.value))
        return algo, state, params
    
    def get_jobs(self, algo_state: ASHAState):
        trial_ready = (algo_state.trial_steps == self._steps_required(algo_state.rungs))
        
        if self.debug:
            jax.debug.print("rungs: {}", algo_state.rungs)
        
        def get_promotable(k):
            rung_mask = algo_state.rungs >= k # trials that have reached this rung
            rung_size = rung_mask.sum() # number of trials that have reached this rung
            
            # of the trials that have reached this rung, how many a trials move to next rung
            m = rung_size // self.reduction_factor
            
            promoted_mask = algo_state.rungs > k # trials that have already been promoted
            num_already_promoted = promoted_mask.sum() # number of trials that have already been promoted
            
            num_to_promote = m - num_already_promoted # of the remaining trials, how many should be promoted
            candidate_mask = (~promoted_mask) & rung_mask & trial_ready # consider only trials on this rung that are ready and have not been promoted
            
            # mask for where the top m trials are
            fitness = jnp.where(candidate_mask, algo_state.fitness[k], -jnp.inf) # consider the fitness of candidate trials only
            
            inds = jnp.argsort(fitness, descending=True)
            mask = jnp.arange(self.num_initial_trials) < num_to_promote #m
            top_m_mask = jnp.zeros_like(mask).at[inds].set(mask)
            
            if self.debug:
                jax.debug.print(
                    "k: {}\n rung_size: {}\n m = {}\n num_to_promote = {}\n num_already_promoted = {}",
                    k, rung_size, m, num_to_promote, num_already_promoted,
                )
            
            # trial is exactly on this rung, and is better than the mth best trial, and is ready
            promotable = (algo_state.rungs == k) & top_m_mask & trial_ready
            
            # if m == 0, then no trials are promotable
            promotable = jax.lax.select(
                num_to_promote > 0,
                promotable,
                jnp.zeros_like(promotable),
            )
            
            return promotable
        
        def get_job(priority, worker_trial_id):
            def select(priority):
                idx = priority.argmax()
                return priority.at[idx].set(-1), (idx, True)
                
            return jax.lax.cond(
                worker_trial_id == -1,
                select,                                         # pick highest priority first
                lambda _: (priority, (worker_trial_id, False)), # skip if worker busy
                priority,
            )
        
        promotable = jax.vmap(get_promotable)(jnp.arange(self.num_rungs))
        priority = jnp.where(
            promotable.any(axis=0),
            promotable.argmax(axis=0) + 1,
            jnp.where(algo_state.rungs == -1, 0, -1)
        )
        
        _, (inds, promoted) = jax.lax.scan(get_job, priority, algo_state.worker_trial_id)
        
        if self.debug:
            jax.debug.print("****Selected: {}, rungs: {}", inds, algo_state.rungs[inds])
            jax.debug.print("Priority: {}", priority)
        
        return inds, promoted
    
    def debug_print_workers(self, algo_state):
        cell_length = max(len(str(self.num_initial_trials)), 2)
        # history = "\n".join(["[" + "][".join([str(x).rjust(cell_length) for x in row]) + "]" for row in algo_state.worker_history][::-1])
        
        print("---")
        print('\n'.join([f"trial {str(i).rjust(cell_length)} : " + '#' * trial_steps + " (" + str(trial_steps) + ")" for i, trial_steps in enumerate(algo_state.trial_steps)]))
        print("---")
        
        # print(algo_state.trial_steps)
        # print(history)
        # print()
        
        # num_dangling_units = self.num_steps - self.cum_rung_units
        # units_per
    
    def _steps_required(self, rungs):
        return (self.reduction_factor ** (rungs + 1) - 1) // (self.reduction_factor - 1)
    
# ==================
# REGISTER ALGORITHM
# ==================
from june.algos.registration import register
if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname
register(algo_id='asha', entry_point=module_path + ':ASHA')
# ==================