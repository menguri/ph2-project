"""Thin wrapper that replicates the CEC eval inference path
(test_general.py:get_rollouts) for use inside ph2-project.

Loads a CEC checkpoint, builds the same `ActorCriticRNN`, and exposes a
single `step(obs_per_agent, hidden, done, key)` call that mirrors the
exact tensor shapes used at training/eval time.

Obs adaptation from ph2's overcooked_v2 state to CEC's (9,9,26) v1 obs is
NOT done here — callers must hand in obs in CEC's native format. The
acceptance test (`scripts/cec_load_smoke.py`) uses random/zero obs to
verify the load + forward path; integration with ph2 envs is layered on
top later.
"""
from __future__ import annotations

from typing import Dict, Optional

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from .actor_networks import ActorCriticRNN, ScannedRNN
from .checkpoint_io import load_checkpoint

# Defaults derived from cec-zero-shot/baselines/CEC/config/ippo_final.yaml.
# Only keys actually consumed by ActorCriticRNN.__call__ are required, but
# we keep the eval-relevant ones too so call sites stay close to upstream.
DEFAULT_CEC_CONFIG: Dict = {
    "ENV_NAME": "overcooked",
    "GRAPH_NET": True,
    "LSTM": True,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "obs_dim": (9, 9, 26),
}

NUM_ACTIONS = 6  # right, down, left, up, stay, interact


class CECRuntime:
    def __init__(
        self,
        ckpt_path: str,
        *,
        config_overrides: Optional[Dict] = None,
        beta: float = 1.0,
        argmax: bool = False,
        prefer_format: str = "orbax",
    ):
        self.config = {**DEFAULT_CEC_CONFIG, **(config_overrides or {})}
        self.beta = float(beta)
        self.argmax = bool(argmax)

        self.network = ActorCriticRNN(action_dim=NUM_ACTIONS, config=self.config)
        ckpt, fmt = load_checkpoint(ckpt_path, prefer_format=prefer_format)
        self.ckpt_format = fmt
        if "params" not in ckpt:
            raise KeyError(
                f"Checkpoint at {ckpt_path} missing 'params' key. Got: {list(ckpt)[:5]}"
            )
        self.params = ckpt["params"]
        self._apply = jax.jit(self.network.apply)

    # ------------------------------------------------------------------
    def init_hidden(self, num_agents: int = 2):
        return ScannedRNN.initialize_carry(num_agents, self.config["GRU_HIDDEN_DIM"])

    # ------------------------------------------------------------------
    def step(
        self,
        obs_per_agent: jnp.ndarray,   # (num_agents, *obs_dim) or (num_agents, flat)
        hidden,
        done: jnp.ndarray,            # (num_agents,) bool
        key: jax.Array,
        agent_positions: Optional[jnp.ndarray] = None,  # (num_agents, 2,2) — see note
    ):
        """Run one inference step. Returns (actions:int[num_agents], new_hidden, probs).

        `obs_per_agent` may be passed in any shape compatible with
        `flatten()` to (num_agents, prod(obs_dim)). The CEC actor reshapes
        internally to obs_dim before convs.
        """
        obs_per_agent = jnp.asarray(obs_per_agent)
        num_agents = obs_per_agent.shape[0]
        obs_batch = obs_per_agent.reshape(num_agents, -1)

        if agent_positions is None:
            # Upstream eval passes the real env_state.agent_pos with shape
            # (num_agents, 2, 2). The GRAPH_NET=True path doesn't actually
            # consume this tensor (the GAT block is commented out), but we
            # still pass a tensor of the right rank so the JIT trace shape
            # matches what the model was trained against.
            agent_positions = jnp.zeros((num_agents, 2, 2), dtype=jnp.int32)
        agent_positions = jnp.asarray(agent_positions)

        ac_in = (
            obs_batch[np.newaxis, :],
            jnp.asarray(done)[np.newaxis, :],
            agent_positions[np.newaxis, :],
        )
        new_hidden, pi, _value = self._apply(self.params, hidden, ac_in)
        logits = pi.logits * self.beta
        scaled = distrax.Categorical(logits=logits)
        if self.argmax:
            actions = jnp.argmax(scaled.probs, axis=-1)[0]
        else:
            actions = scaled.sample(seed=key)[0]
        return actions, new_hidden, scaled.probs[0]
