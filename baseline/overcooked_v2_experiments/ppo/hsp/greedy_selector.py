"""
HSP Greedy Diversity Selection.

원본 HSP (greedy_select.py) 직접 포팅.
학습된 N개 정책의 이벤트 특징을 수집하고,
L1 distance 기반 greedy selection으로 K개 다양한 정책을 선택한다.
"""

import numpy as np
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper, LogWrapper

from overcooked_v2_experiments.ppo.models.rnn import ActorCriticRNN


def greedy_select_policies(event_matrix: np.ndarray, k: int, seed: int = 0) -> list:
    """
    이벤트 특징 행렬에서 L1 distance 기반 greedy diversity selection.

    Args:
        event_matrix: (N, EVENT_DIM) — 각 policy의 정규화된 이벤트 빈도
        k: 선택할 policy 수
    Returns:
        sorted list of selected policy indices
    """
    np.random.seed(seed)
    n = event_matrix.shape[0]
    if k >= n:
        return list(range(n))

    # 정규화
    max_vals = event_matrix.max(axis=0) + 1e-8
    normalized = event_matrix / max_vals

    selected = [np.random.randint(0, n)]
    for _ in range(1, k):
        distances = np.full(n, -1e9)
        for i in range(n):
            if i not in selected:
                distances[i] = sum(
                    np.abs(normalized[i] - normalized[j]).sum()
                    for j in selected
                )
        selected.append(int(distances.argmax()))
    return sorted(selected)


def collect_event_features(
    pop_dir: str | Path,
    config: dict,
    n_eval_episodes: int = 50,
) -> np.ndarray:
    """
    N개 정책을 self-play로 평가하여 이벤트 빈도 행렬을 수집한다.

    Args:
        pop_dir: hsp_population_all/ 경로 (member_0/, member_1/, ...)
        config: 전체 config dict (env, model 포함)
        n_eval_episodes: 각 policy 당 평가 에피소드 수
    Returns:
        event_matrix: (N, EVENT_DIM) numpy array
    """
    pop_dir = Path(pop_dir)
    env_config = config["env"]
    model_config = config["model"]

    env_name = str(env_config.get("ENV_NAME", "overcooked_v2"))
    env_kwargs = dict(env_config.get("ENV_KWARGS", {}))
    env_raw = jaxmarl.make(env_name, **env_kwargs)
    ACTION_DIM = env_raw.action_space(env_raw.agents[0]).n

    if env_name == "overcooked_v2":
        env = OvercookedV2LogWrapper(env_raw, replace_info=False)
    else:
        env = LogWrapper(env_raw, replace_info=False)

    EVENT_DIM = config.get("HSP_EVENT_DIM", 5)
    GRU_HIDDEN_DIM = model_config["GRU_HIDDEN_DIM"]
    network = ActorCriticRNN(action_dim=ACTION_DIM, config=model_config)

    # member 디렉토리 찾기
    member_dirs = sorted(
        [d for d in pop_dir.iterdir() if d.is_dir() and d.name.startswith("member_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    N = len(member_dirs)
    print(f"[HSP Greedy] Collecting event features for {N} policies, {n_eval_episodes} episodes each")

    all_event_counts = []

    for i, md in enumerate(member_dirs):
        # ckpt_final_actor.pkl 로드
        ckpt_path = md / "ckpt_final_actor.pkl"
        if not ckpt_path.exists():
            raise ValueError(f"Missing {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)

        # Self-play 평가: n_eval_episodes 에피소드 실행
        event_counts = _evaluate_policy_events(
            params, network, env, env_raw, EVENT_DIM, GRU_HIDDEN_DIM,
            n_eval_episodes,
        )
        all_event_counts.append(event_counts)
        print(f"  Policy {i}: events = {event_counts}")

    event_matrix = np.array(all_event_counts)  # (N, EVENT_DIM)
    return event_matrix


def _evaluate_policy_events(
    params, network, env, env_raw, event_dim, gru_hidden_dim,
    n_episodes,
):
    """
    단일 policy로 self-play n_episodes 실행 → 에피소드별 이벤트 합산.
    반환: (EVENT_DIM,) 평균 이벤트 카운트
    """
    total_events = np.zeros(event_dim)

    for ep in range(n_episodes):
        key = jax.random.PRNGKey(ep)
        key, key_r = jax.random.split(key)
        obs, state = env.reset(key_r)

        h0 = ActorCriticRNN.initialize_carry(1, gru_hidden_dim)
        h1 = ActorCriticRNN.initialize_carry(1, gru_hidden_dim)
        done = jnp.zeros((1,), dtype=jnp.bool_)

        ep_events = np.zeros(event_dim)
        max_steps = getattr(env_raw, 'max_steps', 400)

        for step in range(max_steps):
            # obs shape: (H, W, C) → (T=1, B=1, H, W, C) for network
            obs_0 = obs[env.agents[0]][jnp.newaxis, jnp.newaxis]
            obs_1 = obs[env.agents[1]][jnp.newaxis, jnp.newaxis]

            key, k0, k1, k_env = jax.random.split(key, 4)

            h0, pi_0, _, _ = network.apply(
                params, h0, (obs_0, done[jnp.newaxis])
            )
            action_0 = pi_0.sample(seed=k0).squeeze(0)

            h1, pi_1, _, _ = network.apply(
                params, h1, (obs_1, done[jnp.newaxis])
            )
            action_1 = pi_1.sample(seed=k1).squeeze(0)

            env_act = {
                env.agents[0]: action_0.squeeze(0),
                env.agents[1]: action_1.squeeze(0),
            }

            obs, state, reward, done_dict, info = env.step(k_env, state, env_act)
            done = jnp.array([done_dict["__all__"]])

            # 이벤트 수집
            if "shaped_reward_events" in info:
                for agent_id in env.agents:
                    ev = info["shaped_reward_events"][agent_id]
                    ep_events += np.array(ev[:event_dim])

            if done_dict["__all__"]:
                break

        total_events += ep_events

    avg_events = total_events / max(n_episodes, 1)
    return avg_events
