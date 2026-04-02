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
from jaxmarl.viz.toy_coop_jitted_visualizer import render_state as toy_coop_render_state
from jaxmarl.viz.grid_spread_visualizer import render_grid as _gs_render_grid, TILE_PIXELS as _GS_TILE


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
    env_name_override: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    # ToyCoop, MPE 등 layout 없는 환경은 직접 지정
    if env_name_override == "ToyCoop":
        tc_kwargs = {k: v for k, v in env_kwargs.items() if k != "layout"}
        return "ToyCoop", tc_kwargs
    if env_name_override is not None and env_name_override.startswith("MPE_"):
        mpe_kwargs = {k: v for k, v in env_kwargs.items() if k != "layout"}
        # checkpoint에서 복원한 값이 JAX array일 수 있으므로 Python native로 변환
        mpe_kwargs = {k: int(v) if hasattr(v, 'item') and jnp.issubdtype(type(v), jnp.integer) else
                      float(v) if hasattr(v, 'item') else v
                      for k, v in mpe_kwargs.items()}
        return env_name_override, mpe_kwargs
    if env_name_override == "GridSpread":
        gs_kwargs = {k: v for k, v in env_kwargs.items() if k != "layout"}
        return "GridSpread", gs_kwargs

    kwargs = dict(env_kwargs)
    kwargs["layout"] = layout
    return "overcooked_v2", kwargs


def _to_python_native(kwargs):
    """JAX array 값을 Python native로 변환 (checkpoint 복원 호환)."""
    result = {}
    for k, v in kwargs.items():
        if hasattr(v, 'item'):
            result[k] = v.item()
        else:
            result[k] = v
    return result


def make_eval_env(
    layout: Any,
    env_kwargs: Dict[str, Any],
    old_overcooked: bool = False,
    disable_auto: bool = False,
    env_name_override: Optional[str] = None,
):
    # ToyCoop, MPE 등 layout 없는 환경은 env_name_override로 직접 지정
    if env_name_override == "ToyCoop":
        tc_kwargs = {k: v for k, v in env_kwargs.items() if k != "layout"}
        env = jaxmarl.make("ToyCoop", **tc_kwargs)
        return env, "ToyCoop", tc_kwargs
    if env_name_override is not None and env_name_override.startswith("MPE_"):
        mpe_kwargs = _to_python_native({k: v for k, v in env_kwargs.items() if k != "layout"})
        env = jaxmarl.make(env_name_override, **mpe_kwargs)
        return env, env_name_override, mpe_kwargs
    if env_name_override == "GridSpread":
        gs_kwargs = _to_python_native({k: v for k, v in env_kwargs.items() if k != "layout"})
        env = jaxmarl.make("GridSpread", **gs_kwargs)
        return env, "GridSpread", gs_kwargs

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
    # MPE: state = concat(obs_agent0, obs_agent1) → global state
    if env_name.startswith("MPE_"):
        obs = env.get_obs(state)
        return jnp.concatenate(
            [obs[a].astype(jnp.float32) for a in env.agents], axis=-1
        )
    # ToyCoop, overcooked_v2 모두 get_obs_default 지원
    return env.get_obs_default(state)[0].astype(jnp.float32)


def extract_pos_yx(state, env_name: str):
    if env_name == "overcooked":
        pos_x = state.agent_pos[:, 0]
        pos_y = state.agent_pos[:, 1]
        return pos_y, pos_x
    if env_name == "ToyCoop":
        pos_x = state.agent_pos[:, 0]
        pos_y = state.agent_pos[:, 1]
        return pos_y, pos_x
    if env_name == "GridSpread":
        return state.agent_pos[:, 1], state.agent_pos[:, 0]
    # MPE: p_pos 사용 (x, y 순서)
    if env_name.startswith("MPE_"):
        # MPE state.p_pos: (num_entities, 2) — 에이전트 + 랜드마크
        pos_x = state.p_pos[:, 0]
        pos_y = state.p_pos[:, 1]
        return pos_y, pos_x
    return state.agents.pos.y, state.agents.pos.x


def render_state_frame(state, env_name: str, agent_view_size: Optional[int] = None):
    if env_name == "overcooked":
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

    if env_name == "ToyCoop":
        img = toy_coop_render_state(state)
        return np.asarray(img)

    if env_name == "GridSpread":
        n_agents = state.agent_pos.shape[0]
        h = int(jnp.max(state.goal_pos).item()) + 1
        img = _gs_render_grid(state.agent_pos, state.goal_pos, n_agents, h, h, _GS_TILE)
        return np.asarray(img)

    # MPE: 시각화 미지원 (학습/평가에는 불필요)
    if env_name.startswith("MPE_"):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    viz = OvercookedV2Visualizer()
    return np.asarray(viz._render_state(state, agent_view_size))
