"""Pre-computed CEC 9x9 layout dicts (generated from
cec-zero-shot/jaxmarl/environments/overcooked/layouts.py via
`make_*_9x9(jax.random.PRNGKey(0), ik=False)`).

These are deterministic when `ik=False, rotate=False`, so we can reproduce
the exact layout dicts CEC used at eval time without depending on
cec-zero-shot at runtime. Each compact template is embedded at top-left
of a 9×9 grid with the remaining cells set to WALL — matching CEC's
`make_9x9_layout(..., rotate=False)`.

Note: `wall_idx` lists every non-empty non-agent cell (including pots,
plates, goals, onion piles) following CEC's `layout_array_to_dict`
convention. ph2's v1 `Overcooked` env masks these correctly.
"""
import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


# Compact CEC templates (ik=False, rotate=False). Cell encoding follows
# CEC's layout_array_to_dict:
#   0=empty, 1=wall, 2=agent, 3=goal, 4=plate_pile, 5=onion_pile, 6=pot
_TEMPLATES = {
    "cramped_room_9": [
        [6, 1, 6, 1, 1],
        [5, 2, 0, 2, 5],
        [1, 0, 0, 0, 1],
        [4, 4, 1, 3, 3],
    ],
    "asymm_advantages_9": [
        [5, 0, 1, 3, 1, 5, 1, 0, 3],
        [1, 0, 2, 0, 6, 0, 2, 0, 1],
        [1, 0, 0, 0, 6, 0, 0, 0, 1],
        [1, 1, 1, 4, 1, 4, 1, 1, 1],
    ],
    "coord_ring_9": [
        [4, 1, 1, 6, 1],
        [1, 0, 2, 0, 6],
        [4, 0, 1, 0, 1],
        [5, 0, 2, 0, 1],
        [1, 5, 3, 1, 3],
    ],
    "forced_coord_9": [
        [1, 1, 1, 6, 1],
        [5, 0, 1, 0, 6],
        [5, 2, 1, 2, 1],
        [4, 0, 1, 0, 1],
        [4, 1, 1, 3, 3],
    ],
    "counter_circuit_9": [
        [4, 1, 1, 6, 6, 1, 1, 3],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [4, 2, 1, 1, 1, 1, 2, 3],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 5, 5, 1, 1, 1],
    ],
}


def _template_to_layout_dict(template):
    """Embed compact template at top-left of 9×9 and build CEC-style layout dict.

    Cells outside the template are filled with walls. Returns FrozenDict with
    the same schema as CEC's `layout_array_to_dict(..., num_base_walls)`.
    """
    template = np.array(template, dtype=np.int32)
    th, tw = template.shape
    grid = np.ones((9, 9), dtype=np.int32)  # start all walls
    grid[:th, :tw] = template
    flat = grid.flatten()

    def _find(value):
        return np.where(flat == value)[0].astype(np.int32).tolist()

    agent_idx = _find(2)
    goal_idx = _find(3)
    plate_pile_idx = _find(4)
    onion_pile_idx = _find(5)
    pot_idx = _find(6)
    explicit_walls = _find(1)

    # CEC convention: wall_idx = explicit walls + goals + plates + onions + pots
    wall_idx = explicit_walls + goal_idx + plate_pile_idx + onion_pile_idx + pot_idx

    return FrozenDict({
        "height": 9,
        "width": 9,
        "agent_idx": jnp.array(agent_idx, dtype=jnp.int32),
        "goal_idx": jnp.array(goal_idx, dtype=jnp.int32),
        "onion_pile_idx": jnp.array(onion_pile_idx, dtype=jnp.int32),
        "plate_pile_idx": jnp.array(plate_pile_idx, dtype=jnp.int32),
        "pot_idx": jnp.array(pot_idx, dtype=jnp.int32),
        "wall_idx": jnp.array(wall_idx, dtype=jnp.int32),
    })


CEC_LAYOUTS = {
    name: _template_to_layout_dict(tmpl) for name, tmpl in _TEMPLATES.items()
}
