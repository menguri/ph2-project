"""GridSpread environment.

N agents spawn at center of (2*radius+1) x (2*radius+1) grid.
N goals at equal angles, Chebyshev-approximate distance radius from center.
All agents must each occupy a distinct goal simultaneously.
Collision prevention: agents moving to the same cell are reverted.
"""

from enum import IntEnum
from functools import partial

import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Tuple, Dict

from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces


class Actions(IntEnum):
    right = 0
    up_right = 1
    up = 2
    up_left = 3
    left = 4
    down_left = 5
    down = 6
    down_right = 7
    stay = 8


@struct.dataclass
class State:
    agent_pos: chex.Array   # (n_agents, 2) — (x, y)
    goal_pos: chex.Array    # (n_agents, 2) — fixed
    time: int
    terminal: bool


class GridSpread(MultiAgentEnv):
    def __init__(
        self,
        n_agents: int = 4,
        radius: int = 3,
        max_steps: int = 100,
        random_reset: bool = False,
        step_cost: float = 1.0,
    ):
        super().__init__(num_agents=n_agents)
        self.n_agents = n_agents
        self.radius = radius
        self.width = 2 * radius + 1
        self.height = 2 * radius + 1
        self.max_steps = max_steps
        self.random_reset = random_reset
        self.step_cost = step_cost

        self.agents = [f"agent_{i}" for i in range(n_agents)]

        self.action_to_dir = jnp.array([
            [ 1,  0],   # right
            [ 1, -1],   # up_right
            [ 0, -1],   # up
            [-1, -1],   # up_left
            [-1,  0],   # left
            [-1,  1],   # down_left
            [ 0,  1],   # down
            [ 1,  1],   # down_right
            [ 0,  0],   # stay
        ])

        # Goal positions: 등각 배치 (Euclidean 원형 근사)
        center = radius
        angles = jnp.array([2.0 * jnp.pi * i / n_agents for i in range(n_agents)])
        goals_x = jnp.clip(
            center + jnp.round(radius * jnp.cos(angles)).astype(jnp.int32),
            0, self.width - 1,
        )
        goals_y = jnp.clip(
            center + jnp.round(radius * jnp.sin(angles)).astype(jnp.int32),
            0, self.height - 1,
        )
        self.fixed_goal_pos = jnp.stack([goals_x, goals_y], axis=-1)  # (n_agents, 2)
        self.spawn_pos = jnp.array([center, center], dtype=jnp.int32)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        agent_pos = jnp.tile(self.spawn_pos[None], (self.n_agents, 1))
        state = State(
            agent_pos=agent_pos,
            goal_pos=self.fixed_goal_pos,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        acts = jnp.array([actions[f"agent_{i}"] for i in range(self.n_agents)])

        # --- 1단계: 마스킹 — 다른 에이전트가 현재 있는 칸으로 이동 시 stay 처리
        candidate_pos = state.agent_pos + self.action_to_dir[acts]
        candidate_pos = jnp.clip(candidate_pos, 0, self.width - 1)

        def _is_blocked(i):
            """agent i의 목적지에 다른 ���이전트가 현재 있는지"""
            others_at_dest = jax.vmap(
                lambda j: jnp.all(candidate_pos[i] == state.agent_pos[j])
            )(jnp.arange(self.n_agents))
            return jnp.any(others_at_dest.at[i].set(False))

        blocked = jax.vmap(lambda i: _is_blocked(i))(jnp.arange(self.n_agents))
        safe_acts = jnp.where(blocked, 8, acts)  # 8 = stay

        # --- 2단계: 동시 충돌 — 두 에이전트가 같은 빈 칸에 동시 도달 시 되돌림
        next_pos = state.agent_pos + self.action_to_dir[safe_acts]
        next_pos = jnp.clip(next_pos, 0, self.width - 1)

        collision_mat = jax.vmap(
            lambda a: jax.vmap(lambda b: jnp.all(a == b))(next_pos)
        )(next_pos)
        collision_mat = collision_mat & ~jnp.eye(self.n_agents, dtype=bool)
        collides = jnp.any(collision_mat, axis=1)
        next_pos = jnp.where(collides[:, None], state.agent_pos, next_pos)

        # on_goal[i,j]: agent i가 goal j 위에 있는지
        on_goal = jax.vmap(
            lambda apos: jax.vmap(
                lambda gpos: jnp.all(apos == gpos)
            )(state.goal_pos)
        )(next_pos)  # (n_agents, n_agents)

        # 각 goal에 정확히 1명이어야 success
        per_goal_count = jnp.sum(on_goal, axis=0)  # (n_agents,)
        all_covered = jnp.all(per_goal_count == 1)

        # shaped reward: 협력 강제 — N-2개까지는 0, N-1개일 때 5, 전부(N) 점령 시 10
        #   목적: "혼자 goal 차지하고 stay" lazy local optimum 제거.
        #   N=4 기준: 0,1,2개 → 0,  3개 → 5,  4개(all_covered) → 10
        n_single_covered = jnp.sum(per_goal_count == 1).astype(jnp.int32)
        _reward_table = jnp.concatenate([
            jnp.zeros(self.n_agents - 1, dtype=jnp.float32),  # 0 ~ N-2 점령: 0
            jnp.array([5.0, 10.0], dtype=jnp.float32),         # N-1 점령: 5, N 점령: 10
        ])  # 길이 N+1
        shaped = _reward_table[n_single_covered]
        reward = shaped - self.step_cost

        next_state = state.replace(agent_pos=next_pos, time=state.time + 1)
        done = self.is_terminal(next_state)
        next_state = next_state.replace(terminal=done)

        obs = self.get_obs(next_state)
        rewards = {f"agent_{i}": reward for i in range(self.n_agents)}
        dones = {f"agent_{i}": done for i in range(self.n_agents)}
        dones["__all__"] = done

        coverage_ratio = n_single_covered / self.n_agents
        # eval용: step_cost 미포함 (순수 성과 지표)
        sparse_reward = jnp.where(all_covered, 10.0, 0.0)
        combined_reward = shaped  # = 0/1/3/5/10 (step_cost 미포함)
        shaped_reward_events = {
            f"agent_{i}": jnp.array([
                all_covered.astype(jnp.float32),
                coverage_ratio,
            ])
            for i in range(self.n_agents)
        }

        # GridSpread 전용 로깅: 매 step에 전체 점령된 goal 개수 (env당 스칼라, 0 ~ N)
        # 모든 에이전트를 통틀어 "현재 점령된 goal 수" — per-agent가 아니라 환경 전체 지표
        n_goals_reached_f = n_single_covered.astype(jnp.float32)
        # k-cover 비율 로깅: 각 step에서 정확히 k개 goal이 점령되었는지 0/1 indicator.
        # rollout 전체에 대해 mean을 취하면 "전체 step 중 k-cover가 발생한 비율"이 됨.
        # k = 2, 3, ..., N 에 대해 각각 cover_k 로 기록.
        info_dict = {
            "sparse_reward": {f"agent_{i}": sparse_reward for i in range(self.n_agents)},
            "combined_reward": {f"agent_{i}": combined_reward for i in range(self.n_agents)},
            "n_goals_reached": n_goals_reached_f,
            "shaped_reward_events": shaped_reward_events,
        }
        for k in range(2, self.n_agents + 1):
            info_dict[f"cover_{k}"] = (n_single_covered == k).astype(jnp.float32)
        return obs, next_state, rewards, dones, info_dict

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Ego-centric 좌표 벡터 observation: (4N,) 1D 벡터.

        [self_x, self_y, other1_x, other1_y, ..., otherN-1_x, otherN-1_y | goal0_x, goal0_y, ..., goalN-1_x, goalN-1_y]

        Agent position 블록은 agent별 cyclic shift로 self가 항상 slot 0에 위치.
        나머지 파트너는 원래 idx 순서를 유지 (temporal consistency 보존).
        Goal은 agent-agnostic 공유 타겟이므로 절대 순서로 고정.
        ego_onehot 불필요 (self 식별이 슬롯 위치에 암묵적으로 인코딩됨).
        좌표는 0~1 정규화.
        """
        n = self.n_agents

        # 에이전트 좌표 정규화: (N, 2)
        norm_ax = state.agent_pos[:, 0].astype(jnp.float32) / max(self.width - 1, 1)
        norm_ay = state.agent_pos[:, 1].astype(jnp.float32) / max(self.height - 1, 1)
        agent_pos_norm = jnp.stack([norm_ax, norm_ay], axis=-1)  # (N, 2)

        # goal 좌표 정규화: (2N,) 고정 순서
        norm_gx = state.goal_pos[:, 0].astype(jnp.float32) / max(self.width - 1, 1)
        norm_gy = state.goal_pos[:, 1].astype(jnp.float32) / max(self.height - 1, 1)
        goal_coords = jnp.stack([norm_gx, norm_gy], axis=-1).ravel()  # (2N,)

        obs = {}
        for i in range(n):
            # self가 slot 0에 오도록 cyclic shift: [i, i+1, ..., N-1, 0, ..., i-1]
            rolled = jnp.roll(agent_pos_norm, shift=-i, axis=0).ravel()  # (2N,)
            obs[f"agent_{i}"] = jnp.concatenate([rolled, goal_coords])   # (4N,)
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_obs_default(self, state: State) -> chex.Array:
        """Full global obs for PH1 pool / CT recon target.
        Returns: (n_agents, 4N)
        """
        obs_dict = self.get_obs(state)
        return jnp.stack(
            [obs_dict[f"agent_{i}"] for i in range(self.n_agents)], axis=0
        )

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        avail = jnp.ones(9, dtype=jnp.int32)
        return {f"agent_{i}": avail for i in range(self.n_agents)}

    @partial(jax.jit, static_argnums=[0])
    def is_terminal(self, state: State) -> bool:
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        return "GridSpread"

    @property
    def num_actions(self) -> int:
        return 9

    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        return spaces.Discrete(9)

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        return spaces.Box(0, 1, (4 * self.n_agents,))
