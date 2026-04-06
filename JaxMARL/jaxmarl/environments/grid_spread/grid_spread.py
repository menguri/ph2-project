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

        # shaped reward: 점령된 goal 수에 따른 단계별 보상
        #   1~3개 goal 점령: 1점씩 (충돌 방지로 겹침 불가 → single_covered만 유효)
        #   4개 전부 점령 (all_covered): 10점
        n_single_covered = jnp.sum(per_goal_count == 1).astype(jnp.float32)
        shaped = jnp.where(all_covered, 10.0, n_single_covered)
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
        combined_reward = shaped  # = n_single_covered or 10.0 (step_cost 미포함)
        shaped_reward_events = {
            f"agent_{i}": jnp.array([
                all_covered.astype(jnp.float32),
                coverage_ratio,
            ])
            for i in range(self.n_agents)
        }

        return obs, next_state, rewards, dones, {
            "sparse_reward": {f"agent_{i}": sparse_reward for i in range(self.n_agents)},
            "combined_reward": {f"agent_{i}": combined_reward for i in range(self.n_agents)},
            "shaped_reward_events": shaped_reward_events,
        }

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """좌표 벡터 observation: (5N,) 1D 벡터.

        [ego_onehot(N) | agent0_x, agent0_y, ..., agentN-1_x, agentN-1_y | goal0_x, goal0_y, ..., goalN-1_x, goalN-1_y]
        좌표는 0~1 정규화. agent slot 고정 (rotation 없음).
        """
        n = self.n_agents

        # 에이전트 좌표 정규화: (2N,)
        norm_ax = state.agent_pos[:, 0].astype(jnp.float32) / max(self.width - 1, 1)
        norm_ay = state.agent_pos[:, 1].astype(jnp.float32) / max(self.height - 1, 1)
        agent_coords = jnp.stack([norm_ax, norm_ay], axis=-1).ravel()  # (2N,)

        # goal 좌표 정규화: (2N,)
        norm_gx = state.goal_pos[:, 0].astype(jnp.float32) / max(self.width - 1, 1)
        norm_gy = state.goal_pos[:, 1].astype(jnp.float32) / max(self.height - 1, 1)
        goal_coords = jnp.stack([norm_gx, norm_gy], axis=-1).ravel()  # (2N,)

        shared = jnp.concatenate([agent_coords, goal_coords])  # (4N,)

        obs = {}
        for i in range(n):
            ego_onehot = jnp.zeros(n).at[i].set(1.0)           # (N,)
            obs[f"agent_{i}"] = jnp.concatenate([ego_onehot, shared])  # (5N,)
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_obs_default(self, state: State) -> chex.Array:
        """Full global obs for PH1 pool / CT recon target.
        Returns: (n_agents, 5N)
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
        return spaces.Box(0, 1, (5 * self.n_agents,))
