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
        dist_shaping_coef: float = 0.0,  # === DIST_SHAPING_EXPERIMENT ===
        early_terminate: bool = False,
    ):
        super().__init__(num_agents=n_agents)
        self.n_agents = n_agents
        self.radius = radius
        self.width = 2 * radius + 1
        self.height = 2 * radius + 1
        self.max_steps = max_steps
        self.random_reset = random_reset
        self.step_cost = step_cost
        # === DIST_SHAPING_EXPERIMENT ===
        self.dist_shaping_coef = dist_shaping_coef
        # early_terminate=True면 all_covered 달성 즉시 episode 종료 (성공 +20 일회성).
        self.early_terminate = early_terminate

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

        # 맵 경계는 clip으로 유지.
        raw_next_pos = state.agent_pos + self.action_to_dir[acts]
        raw_next_pos = jnp.clip(raw_next_pos, 0, self.width - 1)

        # === 겹침 방지 (Overcooked-V2 패턴) ===
        # 같은 칸에 2명 이상 가려고 하면 해당 에이전트들은 이동 취소(제자리).
        # cascade 처리: A가 멈추면서 B의 이동이 새 충돌을 만들 수 있으므로 수렴까지 반복.
        def _masked_positions(mask):
            # mask[i]=True → 에이전트 i는 제자리, False → raw_next_pos 사용
            return jnp.where(mask[:, None], state.agent_pos, raw_next_pos)

        def _get_collisions(mask):
            positions = _masked_positions(mask)  # (N, 2)
            grid = jnp.zeros((self.height, self.width), dtype=jnp.int32)
            grid, _ = jax.lax.scan(
                lambda g, p: (g.at[p[1], p[0]].add(1), None),
                grid,
                positions,
            )
            collision_grid = grid > 1
            return jax.vmap(lambda p: collision_grid[p[1], p[0]])(positions)

        # 초기 스폰(전부 중앙)에서 시작하면 fixed point가 "전부 겹침"이라
        # "any collisions" 조건은 무한 루프. mask가 변하지 않을 때까지만 돈다.
        def _cond(carry):
            prev_mask, new_mask = carry
            return jnp.any(prev_mask != new_mask)

        def _body(carry):
            _, m = carry
            return m, m | _get_collisions(m)

        initial_mask = jnp.zeros((self.n_agents,), dtype=bool)
        first_mask = initial_mask | _get_collisions(initial_mask)
        _, mask = jax.lax.while_loop(_cond, _body, (initial_mask, first_mask))
        next_pos = _masked_positions(mask)

        # on_goal[i,j]: agent i가 goal j 위에 있는지
        on_goal = jax.vmap(
            lambda apos: jax.vmap(
                lambda gpos: jnp.all(apos == gpos)
            )(state.goal_pos)
        )(next_pos)  # (n_agents, n_agents)

        # 각 goal에 정확히 1명이어야 success
        per_goal_count = jnp.sum(on_goal, axis=0)  # (n_agents,)
        all_covered = jnp.all(per_goal_count == 1)

        # shaped reward: N개 전부 점령 시 +20, N-1개 점령 시 +0.1 중간 보상.
        n_single_covered = jnp.sum(per_goal_count == 1).astype(jnp.int32)
        shaped = jnp.where(
            all_covered,
            50.0,
            jnp.where(
                n_single_covered == self.n_agents - 1,
                5.0,
                jnp.where(
                    n_single_covered == self.n_agents - 2,
                    0.5,
                    jnp.where(
                        n_single_covered == self.n_agents - 3,
                        0.1,
                        0.0,
                    ),
                ),
            ),
        ).astype(jnp.float32)
        reward = shaped - self.step_cost

        # === DIST_SHAPING_EXPERIMENT START ===
        # 실험: 각 agent에 가장 가까운 goal까지의 Chebyshev 거리에 비례한 약한 음성 보상.
        # 효과 없으면 이 블록 전체(아래 continued 블록까지)를 삭제하고
        #   rewards = {f"agent_{i}": reward for i in range(self.n_agents)}
        # 한 줄로 되돌리면 원래 동작(shared scalar reward)으로 복원됨.
        _DIST_COEF = self.dist_shaping_coef
        _dist_mat = jnp.max(
            jnp.abs(next_pos[:, None, :] - state.goal_pos[None, :, :]), axis=-1
        )  # (N, N): Chebyshev distance
        _min_dist = jnp.min(_dist_mat, axis=1).astype(jnp.float32)  # (N,)
        _per_agent_reward = reward - _DIST_COEF * _min_dist  # (N,)
        # === DIST_SHAPING_EXPERIMENT END ===

        next_state = state.replace(agent_pos=next_pos, time=state.time + 1)
        time_up = next_state.time >= self.max_steps
        # early_terminate 모드: all_covered 달성 즉시 done. 기본은 time만 체크.
        if self.early_terminate:
            done = time_up | all_covered
        else:
            done = time_up
        next_state = next_state.replace(terminal=done)

        obs = self.get_obs(next_state)
        # === DIST_SHAPING_EXPERIMENT START (continued) ===
        rewards = {f"agent_{i}": _per_agent_reward[i] for i in range(self.n_agents)}
        # === DIST_SHAPING_EXPERIMENT END ===
        dones = {f"agent_{i}": done for i in range(self.n_agents)}
        dones["__all__"] = done

        coverage_ratio = n_single_covered / self.n_agents
        # eval용: step_cost 미포함 (순수 성과 지표)
        sparse_reward = jnp.where(all_covered, 20.0, 0.0)
        combined_reward = shaped  # = 0/5/20 (step_cost 미포함)
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
            # episode-level success rate 계산용: done step에서만 all_covered 여부를 기록.
            # rollout mean을 취한 뒤 success_rate = success_at_done / ep_done_flag (with eps).
            "success_at_done": (done & all_covered).astype(jnp.float32),
            "ep_done_flag": done.astype(jnp.float32),
        }
        for k in range(2, self.n_agents + 1):
            info_dict[f"cover_{k}"] = (n_single_covered == k).astype(jnp.float32)
        return obs, next_state, rewards, dones, info_dict

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Obs = [ego_onehot(N) | full_state(4N)] → dim = 5N.

        full_state = [agent0_x, agent0_y, ..., agentN-1_x, agentN-1_y,
                       goal0_x,  goal0_y,  ..., goalN-1_x,  goalN-1_y]
        좌표는 raw 정수(float32 캐스트)로 반환 — 정규화는 네트워크/호출부에서 담당.
        """
        n = self.n_agents

        agent_coords = state.agent_pos.astype(jnp.float32).ravel()  # (2N,)
        goal_coords = state.goal_pos.astype(jnp.float32).ravel()    # (2N,)
        full_state = jnp.concatenate([agent_coords, goal_coords])   # (4N,)

        obs = {}
        for i in range(n):
            ego_onehot = jnp.zeros(n, dtype=jnp.float32).at[i].set(1.0)  # (N,)
            obs[f"agent_{i}"] = jnp.concatenate([ego_onehot, full_state])  # (5N,)
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
        # 경계 밖으로 나가는 액션은 무효 → logit masking용 마스크 계산.
        # 9개 액션 각각을 현재 위치에 적용했을 때 맵 안에 들어오는지로 판정.
        # stay(idx=8)는 (0,0)이라 항상 valid.
        def _mask(pos):
            cand = pos[None, :] + self.action_to_dir  # (9, 2)
            in_x = (cand[:, 0] >= 0) & (cand[:, 0] < self.width)
            in_y = (cand[:, 1] >= 0) & (cand[:, 1] < self.height)
            return (in_x & in_y).astype(jnp.int32)

        masks = jax.vmap(_mask)(state.agent_pos)  # (n_agents, 9)
        return {f"agent_{i}": masks[i] for i in range(self.n_agents)}

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
        return spaces.Box(0, max(self.width, self.height) - 1, (5 * self.n_agents,))
