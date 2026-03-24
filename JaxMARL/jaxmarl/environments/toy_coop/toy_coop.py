"""Dual Destination (ToyCoop) environment.

A 5x5 cooperative gridworld where two agents must navigate to different goals
simultaneously. Ported from cec-zero-shot for use in ph2-project, with CEC-specific
features (held-out state tracking) removed.
"""

from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from functools import partial


class Actions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4


@struct.dataclass
class State:
    agent_pos: chex.Array       # (2, 2) - positions of both agents
    goal_pos: chex.Array        # (2, 2) - positions of "green" goals
    other_goal_pos: chex.Array  # (2, 2) - positions of "pink" goals
    time: int
    terminal: bool


class ToyCoop(MultiAgentEnv):
    """Simple 5x5 cooperative gridworld (Dual Destination).

    Two agents must simultaneously occupy *different* goal cells to receive reward.
    """

    def __init__(
        self,
        max_steps: int = 100,
        random_reset: bool = False,
        debug: bool = False,
        partial_obs: bool = False,
        incentivize_strat: int = 2,
        step_cost: float = 1.0,
    ):
        super().__init__(num_agents=2)
        self.width = 5
        self.height = 5
        self.max_steps = max_steps
        self.random_reset = random_reset
        self.debug = debug
        self.action_set = jnp.array([
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
        ])
        self.agents = ["agent_0", "agent_1"]

        # Movement vectors for each action
        self.action_to_dir = jnp.array([
            [1, 0],   # right
            [0, 1],   # down
            [-1, 0],  # left
            [0, -1],  # up
            [0, 0],   # stay
        ])

        self.all_pos = jnp.array(
            [[x, y] for x in range(self.width) for y in range(self.height)]
        )

        self.partial_obs = partial_obs
        self.incentivize_strat = incentivize_strat
        self.step_cost = step_cost

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state."""
        state = self.custom_reset_fn(key, random_reset=self.random_reset, debug=self.debug)
        obs = self.get_obs(state)
        return obs, state

    def custom_reset_fn(self, key, random_reset=False, debug=False):
        key1, key2 = jax.random.split(key)

        og_locations_2 = jnp.array(
            [[0, 2], [4, 2], [2, 0], [2, 4], [2, 1], [2, 3]]
        )
        og_locations_3 = jnp.array(
            [[0, 2], [4, 2], [1, 0], [3, 4], [1, 4], [3, 1]]
        )
        og_locations = jnp.where(
            self.incentivize_strat == 3, og_locations_3, og_locations_2
        )

        # Randomly place agents and goals
        indices = jax.random.permutation(key1, len(self.all_pos))[:6]
        rand_agent_pos = self.all_pos[indices[:2]]
        rand_goal_pos = self.all_pos[indices[2:4]]
        rand_other_goal_pos = self.all_pos[indices[4:]]

        agent_pos = jnp.where(random_reset, rand_agent_pos, og_locations[:2])
        green_goal_default = jnp.where(debug, og_locations[:2], og_locations[2:4])
        pink_goal_default = jnp.where(debug, og_locations[:2], og_locations[4:])
        goal_pos = jnp.where(random_reset, rand_goal_pos, green_goal_default)
        other_goal_pos = jnp.where(random_reset, rand_other_goal_pos, pink_goal_default)

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            other_goal_pos=other_goal_pos,
            time=0,
            terminal=False,
        )
        return state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""
        acts = jnp.array([actions["agent_0"], actions["agent_1"]])
        next_state, reward = self.step_agents(key, state, acts)

        next_state = next_state.replace(time=state.time + 1)
        done = self.is_terminal(next_state)
        next_state = next_state.replace(terminal=done)

        obs = self.get_obs(next_state)
        rewards = {"agent_0": reward, "agent_1": reward}
        shaped_reward = {"agent_0": 0, "agent_1": 0}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return obs, next_state, rewards, dones, {"shaped_reward": shaped_reward}

    @partial(jax.jit, static_argnums=[0])
    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float]:
        """Update agent positions and calculate rewards."""
        # Calculate next positions
        next_pos = state.agent_pos + self.action_to_dir[actions]

        # Bound positions to grid
        next_pos = jnp.clip(next_pos, 0, self.width - 1)

        # Check if positions would collide
        would_collide = jnp.all(next_pos[0] == next_pos[1])
        next_pos = jnp.where(would_collide, state.agent_pos, next_pos)

        # Modified reward calculation
        on_goal = lambda x, y: jnp.all(x == y)

        # Check which goal each agent is on (if any)
        agent0_green_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[0], state.goal_pos)
        agent1_green_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[1], state.goal_pos)
        agent0_pink_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[0], state.other_goal_pos)
        agent1_pink_goal = jax.vmap(on_goal, in_axes=(None, 0))(next_pos[1], state.other_goal_pos)

        # Only give reward if agents are on different goals
        both_on_green_goals = jnp.logical_and(
            jnp.any(agent0_green_goal), jnp.any(agent1_green_goal)
        )
        on_same_green_goal = jnp.any(jnp.logical_and(agent0_green_goal, agent1_green_goal))
        both_on_pink_goals = jnp.logical_and(
            jnp.any(agent0_pink_goal), jnp.any(agent1_pink_goal)
        )
        on_same_pink_goal = jnp.any(jnp.logical_and(agent0_pink_goal, agent1_pink_goal))

        incentivize_strat_cond = True  # True by default not incentivizing any strategy
        # agent 0 is incentivized to do top strategy
        incentivize_strat_cond = jnp.where(
            self.incentivize_strat == 0,
            jnp.all(next_pos[0] == state.goal_pos[0]),
            incentivize_strat_cond,
        )
        # agent 0 is incentivized to do bottom strategy
        incentivize_strat_cond = jnp.where(
            self.incentivize_strat == 1,
            jnp.all(next_pos[0] == state.goal_pos[1]),
            incentivize_strat_cond,
        )

        # reward is 3 if both on goals and not on same goal, and incentivize_strat_cond is True
        green_reward = (
            jnp.float32(
                jnp.logical_and(
                    incentivize_strat_cond,
                    jnp.logical_and(both_on_green_goals, ~on_same_green_goal),
                )
            )
            * 3
        )
        pink_reward = (
            jnp.float32(jnp.logical_and(both_on_pink_goals, ~on_same_pink_goal)) * 3
        )

        reward = jnp.where(
            self.incentivize_strat == 2,
            green_reward,
            pink_reward + green_reward,
        )

        return state.replace(agent_pos=next_pos), reward - self.step_cost

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Convert state into agent observations."""
        obs = jnp.zeros((self.height, self.width, 4))

        # Set agent positions
        obs = obs.at[state.agent_pos[0, 1], state.agent_pos[0, 0], 0].set(1)
        obs = obs.at[state.agent_pos[1, 1], state.agent_pos[1, 0], 1].set(1)

        # Set goal positions
        obs_0 = obs.at[state.goal_pos[:, 1], state.goal_pos[:, 0], 2].set(1)
        obs_0 = obs_0.at[state.other_goal_pos[:, 1], state.other_goal_pos[:, 0], 3].set(1)

        obs_1 = obs_0.at[:, :, 0].set(obs_0[:, :, 1])
        obs_1 = obs_1.at[:, :, 1].set(obs_0[:, :, 0])  # swap agent 0 and 1

        def make_partial_obs(obs):
            ego_pos = jnp.where(obs[:, :, 0] == 1, size=1)
            ego_y, ego_x = ego_pos[0][0], ego_pos[1][0]

            partial_obs = jnp.full_like(obs, -1)

            y_min = jnp.maximum(0, ego_y - 1)
            y_max = jnp.minimum(self.height - 1, ego_y + 1)
            x_min = jnp.maximum(0, ego_x - 1)
            x_max = jnp.minimum(self.width - 1, ego_x + 1)

            y_indices = jnp.arange(self.height)
            x_indices = jnp.arange(self.width)

            y_in_range = jnp.logical_and(y_indices >= y_min, y_indices <= y_max)
            x_in_range = jnp.logical_and(x_indices >= x_min, x_indices <= x_max)

            in_window = jnp.logical_and(
                y_in_range.reshape(-1, 1),
                x_in_range.reshape(1, -1),
            )

            partial_obs = jnp.where(
                in_window.reshape(self.height, self.width, 1),
                obs,
                partial_obs,
            )
            return partial_obs

        make_partial_fn = lambda o: jax.lax.cond(
            self.partial_obs,
            lambda x: make_partial_obs(x),
            lambda x: x,
            o,
        )
        stacked_obs = jnp.stack([obs_0, obs_1], axis=0)
        stacked_obs = jax.vmap(make_partial_fn)(stacked_obs)
        obs_0 = stacked_obs[0]
        obs_1 = stacked_obs[1]

        return {
            "agent_0": obs_0,
            "agent_1": obs_1,
        }

    @partial(jax.jit, static_argnums=[0])
    def get_obs_default(self, state: State) -> chex.Array:
        """partial_obs 설정과 무관하게 항상 full obs를 반환.
        PH1/CT recon target 용. 반환 shape: (num_agents, H, W, 4)"""
        obs = jnp.zeros((self.height, self.width, 4))
        obs = obs.at[state.agent_pos[0, 1], state.agent_pos[0, 0], 0].set(1)
        obs = obs.at[state.agent_pos[1, 1], state.agent_pos[1, 0], 1].set(1)
        obs_0 = obs.at[state.goal_pos[:, 1], state.goal_pos[:, 0], 2].set(1)
        obs_0 = obs_0.at[state.other_goal_pos[:, 1], state.other_goal_pos[:, 0], 3].set(1)
        obs_1 = obs_0.at[:, :, 0].set(obs_0[:, :, 1])
        obs_1 = obs_1.at[:, :, 1].set(obs_0[:, :, 0])
        return jnp.stack([obs_0, obs_1], axis=0)  # (2, H, W, 4)

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent. All actions always available."""
        avail = jnp.ones(len(self.action_set), dtype=jnp.int32)
        return {"agent_0": avail, "agent_1": avail}

    @partial(jax.jit, static_argnums=[0])
    def is_terminal(self, state: State) -> bool:
        """Check if episode is done."""
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "ToyCoop"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.height, self.width, 4))
