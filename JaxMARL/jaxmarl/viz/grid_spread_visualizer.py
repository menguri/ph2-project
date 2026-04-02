"""JIT-compiled visualizer for GridSpread.
Agents as colored circles, goals as green rectangles.
Supports up to 10 agents with distinct colors.

Usage:
    render = make_render_fn(env)   # env는 GridSpread 인스턴스
    img = render(state)            # uint8 RGB array
"""

import jax
import jax.numpy as jnp
from jax import lax

TILE_PIXELS = 24  # 11x11 grid → 264x264 image

# 10 에이전트 고유 색상
AGENT_COLORS = jnp.array([
    [220,  50,  50],  # Red
    [50,   50, 220],  # Blue
    [50,  180,  50],  # Green
    [220, 140,  30],  # Orange
    [140,  50, 200],  # Purple
    [30,  180, 200],  # Cyan
    [200, 200,  30],  # Yellow
    [200,  50, 150],  # Pink
    [100, 100, 100],  # Gray
    [150, 100,  50],  # Brown
], dtype=jnp.uint8)

GOAL_COLOR = jnp.array([50, 200, 80],   dtype=jnp.uint8)  # 초록
GRID_COLOR = jnp.array([180, 180, 180], dtype=jnp.uint8)  # 격자
BG_COLOR   = jnp.array([255, 255, 255], dtype=jnp.uint8)  # 흰 배경


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax),
            jnp.logical_and(y >= ymin, y <= ymax),
        )
    return fn


def fill_coords(img, fn, color):
    y, x = jnp.meshgrid(
        jnp.arange(img.shape[0]), jnp.arange(img.shape[1]), indexing='ij'
    )
    yf = (y + 0.5) / img.shape[0]
    xf = (x + 0.5) / img.shape[1]
    mask = fn(xf, yf)
    return jnp.where(mask[:, :, None], color, img)


def render_tile(n_here, first_agent_idx, is_goal, tile_size=TILE_PIXELS):
    img = jnp.full((tile_size, tile_size, 3), BG_COLOR, dtype=jnp.uint8)

    # 격자선 (4변)
    for fn in [
        point_in_rect(0, 0.04, 0, 1),
        point_in_rect(0.96, 1, 0, 1),
        point_in_rect(0, 1, 0, 0.04),
        point_in_rect(0, 1, 0.96, 1),
    ]:
        img = fill_coords(img, fn, GRID_COLOR)

    # Goal: 내부 사각형 (초록)
    def draw_goal(img):
        return fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), GOAL_COLOR)

    img = jax.lax.cond(is_goal, draw_goal, lambda i: i, img)

    # Agent: 컬러 원
    def draw_agent(img):
        color = AGENT_COLORS[jnp.clip(first_agent_idx, 0, 9)]
        img = fill_coords(img, point_in_circle(0.5, 0.5, 0.35), color)
        # 2명 이상 → 흰 내부 원 (중첩 표시)
        img = jax.lax.cond(
            n_here > 1,
            lambda i: fill_coords(
                i, point_in_circle(0.5, 0.5, 0.18),
                jnp.array([255, 255, 255], dtype=jnp.uint8),
            ),
            lambda i: i,
            img,
        )
        return img

    img = jax.lax.cond(first_agent_idx >= 0, draw_agent, lambda i: i, img)
    return img


def render_grid(agent_pos, goal_pos, n_agents, height, width, tile_size=TILE_PIXELS):
    img = jnp.zeros((height * tile_size, width * tile_size, 3), dtype=jnp.uint8)

    def render_tile_at(img, yx):
        y, x = yx[0], yx[1]

        at_pos = jax.vmap(
            lambda ap: jnp.all(jnp.array([x, y]) == ap)
        )(agent_pos)  # (n_agents,) bool

        n_here = jnp.sum(at_pos).astype(jnp.int32)
        first_idx = jnp.where(
            jnp.any(at_pos),
            jnp.argmax(at_pos).astype(jnp.int32),
            jnp.array(-1, dtype=jnp.int32),
        )
        is_goal = jnp.any(
            jax.vmap(lambda gp: jnp.all(jnp.array([x, y]) == gp))(goal_pos)
        )

        tile_img = render_tile(n_here, first_idx, is_goal, tile_size).astype(jnp.uint8)
        img = lax.dynamic_update_slice(
            img, tile_img,
            (y * tile_size, x * tile_size, jnp.array(0, dtype=jnp.int32)),
        )
        return img, None

    ys, xs = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')
    yx_pairs = jnp.stack([ys.ravel(), xs.ravel()], axis=-1)  # (H*W, 2)
    img, _ = lax.scan(render_tile_at, img, yx_pairs)
    return img


def make_render_fn(env):
    """GridSpread 인스턴스로부터 JIT render 함수 생성.

    Usage:
        render = make_render_fn(env)
        img = render(state)  # (H*TILE, W*TILE, 3) uint8
    """
    height = env.height
    width = env.width
    n_agents = env.n_agents

    @jax.jit
    def _render(state):
        return render_grid(
            state.agent_pos, state.goal_pos,
            n_agents, height, width, TILE_PIXELS,
        )

    return _render
