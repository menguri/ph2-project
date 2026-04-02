"""GridSpread environment.

N agents spawn at center of (2*radius+1) x (2*radius+1) grid.
N goals at equal angles, Chebyshev-approximate distance radius from center.
All agents must each occupy a distinct goal simultaneously for +1/step.
No collision: multiple agents may share a cell.
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
        radius: int = 2,
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

        # 이동 (충돌 없음)
        next_pos = state.agent_pos + self.action_to_dir[acts]
        next_pos = jnp.clip(next_pos, 0, self.width - 1)

        # on_goal[i,j]: agent i가 goal j 위에 있는지
        on_goal = jax.vmap(
            lambda apos: jax.vmap(
                lambda gpos: jnp.all(apos == gpos)
            )(state.goal_pos)
        )(next_pos)  # (n_agents, n_agents)

        # 각 goal에 정확히 1명이어야 success
        per_goal_count = jnp.sum(on_goal, axis=0)  # (n_agents,)
        all_covered = jnp.all(per_goal_count == 1)
        reward = jnp.where(all_covered, 5.0, -self.step_cost)

        next_state = state.replace(agent_pos=next_pos, time=state.time + 1)
        done = self.is_terminal(next_state)
        next_state = next_state.replace(terminal=done)

        obs = self.get_obs(next_state)
        rewards = {f"agent_{i}": reward for i in range(self.n_agents)}
        dones = {f"agent_{i}": done for i in range(self.n_agents)}
        dones["__all__"] = done

        coverage_ratio = jnp.sum(per_goal_count == 1).astype(jnp.float32) / self.n_agents
        shaped_reward = {f"agent_{i}": 0.0 for i in range(self.n_agents)}
        shaped_reward_events = {
            f"agent_{i}": jnp.array([
                all_covered.astype(jnp.float32),
                coverage_ratio,
            ])
            for i in range(self.n_agents)
        }

        return obs, next_state, rewards, dones, {
            "shaped_reward": shaped_reward,
            "shaped_reward_events": shaped_reward_events,
        }

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Ego-centric observation: (H, W, n_agents+1).

        Ch 0:   ego agent 위치
        Ch 1~n-1: 다른 에이전트 위치 (rotation으로 ego가 항상 ch 0)
        Ch n:   모든 goal 위치 (단일 채널)
        """
        H, W, n = self.height, self.width, self.n_agents

        # 에이전트별 binary 채널
        agent_channels = jnp.stack([
            jnp.zeros((H, W)).at[state.agent_pos[i, 1], state.agent_pos[i, 0]].set(1.0)
            for i in range(n)
        ], axis=0)  # (n, H, W)

        # Goal channel (단일): vectorized scatter
        goal_channel = jnp.zeros((H, W)).at[
            state.goal_pos[:, 1], state.goal_pos[:, 0]
        ].set(1.0)  # (H, W)

        def make_ego_obs(ego_i):
            rotated = jnp.roll(agent_channels, shift=-ego_i, axis=0)  # (n, H, W)
            all_ch = jnp.concatenate([rotated, goal_channel[None]], axis=0)  # (n+1, H, W)
            return jnp.transpose(all_ch, (1, 2, 0))  # (H, W, n+1)

        return {f"agent_{i}": make_ego_obs(i) for i in range(n)}

    @partial(jax.jit, static_argnums=[0])
    def get_obs_default(self, state: State) -> chex.Array:
        """Full global obs for PH1 pool / CT recon target.
        Returns: (n_agents, H, W, n_agents+1)
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
        return spaces.Box(0, 1, (self.height, self.width, self.n_agents + 1))
