import jax
import jax.numpy as jnp
import june
from june.utils.param_space import ParamSpace
from june.utils.domain import uniform, choice, randint

ENV = "CartPole-v1"

def eval_best(algo, storage):
    best_item = storage.get(storage.buffer["fitness"].argmax())
    algo_state, algo_params = best_item["state"], best_item["params"]
    algo, algo_state, algo_params = algo.unwrap(algo_state, algo_params)
    evaluations = algo.evaluate(algo_state, algo_params)
    return algo.algo_cls, algo_params, evaluations

def make_base_algo():
    env_config = {
        "env": ENV,
        "total_timesteps": 1048576,
    }

    ppo = june.algos.make("rejax", algorithm="ppo", param_kwargs=env_config)
    ppo_params = ppo.default_params.replace(
        learning_rate=uniform(1e-5, 1e-2).mutable(),
        gamma=uniform(0.7, 1.0).mutable(),
        vf_coef=uniform(0.1, 1.0).mutable(),
        ent_coef=uniform(0., 0.2).mutable(),
    )

    sac = june.algos.make("rejax", algorithm="sac", param_kwargs=env_config)
    sac_params = sac.default_params.replace(
        learning_rate=uniform(1e-5, 1e-2).mutable(),
        gamma=uniform(0.7, 1.0).mutable(),
        polyak=uniform(0.1, 1.0).mutable(),
        target_entropy_ratio=uniform(0.5, 1.0).mutable(),
    )

    dqn = june.algos.make("rejax", algorithm="dqn", param_kwargs=env_config)
    dqn_params = dqn.default_params.replace(
        learning_rate=uniform(1e-5, 1e-2).mutable(),
        gamma=uniform(0.7, 1.0).mutable(),
        polyak=uniform(0.1, 1.0).mutable(),
        eps_start=uniform(0.1, 1.0).mutable(),
        eps_end=uniform(0.0, 0.1).mutable(),
    )

    base_algo = june.algos.make("switch", algorithms=[ppo, sac, dqn])
    search_space = ParamSpace(base_algo.default_params.replace(
        index=choice(jnp.array([0, 1, 2])), # explicitly not mutable
        params=[ppo_params, sac_params, dqn_params],
    ))

    return base_algo, search_space

def run_random_search(base_algo, search_space, seed):
    rng = jax.random.PRNGKey(seed)
    algo = june.algos.make("random_search", algorithm=base_algo, num_trials=4, num_steps=8)
    algo_params = algo.default_params.replace(search_space=search_space)
    algo_state = algo.init_state(rng, algo_params)
    algo_state, evaluations = algo.train(algo_state, algo_params)
    algo, algo_params, final_eval = eval_best(algo.algo, algo_state.storage)
    return algo, final_eval.mean(), final_eval.std()

def run_pbt(base_algo, search_space, seed):
    rng = jax.random.PRNGKey(seed)
    algo = june.algos.make("pbt", algorithm=base_algo, pop_size=4, num_steps=8)
    algo_params = algo.default_params.replace(search_space=search_space)
    algo_state = algo.init_state(rng, algo_params)
    algo_state, evaluations = algo.train(algo_state, algo_params)
    algo, algo_params, final_eval = eval_best(algo.algo, algo_state.storage)
    return algo, final_eval.mean(), final_eval.std()

def run_asha(base_algo, search_space, seed):
    rng = jax.random.PRNGKey(seed)
    algo = june.algos.make("asha", algorithm=base_algo, num_workers=4, num_steps=8)
    algo_params = algo.default_params.replace(search_space=search_space)
    algo_state = algo.init_state(rng, algo_params)
    algo_state, evaluations = algo.train(algo_state, algo_params)
    algo, algo_params, final_eval = eval_best(algo.algo, algo_state.storage)
    return algo, final_eval.mean(), final_eval.std()

# number of tests
N = 5
base_algo, search_space = make_base_algo()

base_algo, search_space = make_base_algo()
for i in range(N):
    print(f"cartpole RANDOM_SEARCH {i}:", run_random_search(base_algo, search_space, i))
for i in range(N):
    print(f"cartpole PBT {i}:", run_pbt(base_algo, search_space, i))
for i in range(N):
    print(f"cartpole ASHA {i}:", run_asha(base_algo, search_space, i))