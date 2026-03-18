import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import jax
import numpy as np
import jax.numpy as jnp
import jaxmarl
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def get_recipe_identifier(ingredients: List[int]) -> int:
    """
    Get the identifier for a recipe given the ingredients.
    """
    return f"{ingredients[0]}_{ingredients[1]}_{ingredients[2]}"


def resolve_old_overcooked_flags(config: Dict[str, Any]) -> Tuple[bool, bool]:
    old_overcooked = bool(config.get("OLD_OVERCOOKED", False))
    disable_auto = bool(config.get("DISABLE_OLD_OVERCOOKED_AUTO", False))
    if "alg" in config and isinstance(config["alg"], dict):
        old_overcooked = old_overcooked or bool(
            config["alg"].get("OLD_OVERCOOKED", False)
        )
        disable_auto = disable_auto or bool(
            config["alg"].get("DISABLE_OLD_OVERCOOKED_AUTO", False)
        )
    # If the checkpoint was trained with the overcooked_v2 engine, disable auto-detection
    # so that layouts like "cramped_room" (which also exist in overcooked v1) are not
    # incorrectly routed to the v1 engine during eval.
    env_cfg = config.get("env", {})
    if isinstance(env_cfg, dict):
        env_name = env_cfg.get("ENV_NAME", config.get("ENV_NAME", ""))
        if str(env_name) == "overcooked_v2":
            disable_auto = True
    return old_overcooked, disable_auto


def resolve_eval_engine(
    layout: Any,
    env_kwargs: Dict[str, Any],
    old_overcooked: bool = False,
    disable_auto: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    kwargs = dict(env_kwargs)
    kwargs["layout"] = layout

    # Always route evaluation to overcooked_v2.
    # NOTE:
    # - We intentionally ignore `old_overcooked` / `disable_auto` flags here.
    # - This prevents layout-name based auto-routing to overcooked(v1), which can
    #   cause observation-shape mismatches against v2-trained checkpoints.

    return "overcooked_v2", kwargs


def make_eval_env(
    layout: Any,
    env_kwargs: Dict[str, Any],
    old_overcooked: bool = False,
    disable_auto: bool = False,
):
    env_name, resolved_kwargs = resolve_eval_engine(
        layout=layout,
        env_kwargs=env_kwargs,
        old_overcooked=old_overcooked,
        disable_auto=disable_auto,
    )
    env = jaxmarl.make(env_name, **resolved_kwargs)
    return env, env_name, resolved_kwargs


def extract_global_full_obs(env, state, env_name: str):
    if env_name == "overcooked":
        obs = env.get_obs(state)
        return obs[env.agents[0]].astype(jnp.float32)
    return env.get_obs_default(state)[0].astype(jnp.float32)


def extract_pos_yx(state, env_name: str):
    if env_name == "overcooked":
        pos_x = state.agent_pos[:, 0]
        pos_y = state.agent_pos[:, 1]
        return pos_y, pos_x
    return state.agents.pos.y, state.agents.pos.x


def render_state_frame(state, env_name: str, agent_view_size: Optional[int] = None):
    if env_name == "overcooked":
        # For classic overcooked(v1), always render full-view equivalent.
        # Ignore OV2 partial-view configs (e.g. agent_view_size=2).
        avs = 5
        padding = max(1, avs - 2)
        h, w = int(state.maze_map.shape[0]), int(state.maze_map.shape[1])
        if (2 * padding) >= h or (2 * padding) >= w:
            padding = 1
        grid = np.asarray(state.maze_map[padding : h - padding, padding : w - padding, :])
        return OvercookedVisualizer._render_grid(
            grid,
            highlight_mask=None,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv,
        )

    viz = OvercookedV2Visualizer()
    return np.asarray(viz._render_state(state, agent_view_size))
