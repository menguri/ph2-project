"""Pre-computed CEC 9x9 layout dicts (generated from
cec-zero-shot/jaxmarl/environments/overcooked/layouts.py via
`make_*_9x9(jax.random.PRNGKey(0), ik=False)`).

These are deterministic when `ik=False, rotate=False`, so we can ship them
as static data and feed them to ph2-project's own v1 `Overcooked` env (which
accepts the same layout-dict schema). Result: a (9,9,26) obs identical to
what CEC was trained on, without depending on cec-zero-shot at runtime.

Note: `wall_idx` may overlap with `pot_idx`/`goal_idx`/`onion_pile_idx`/
`plate_pile_idx` because CEC's `layout_array_to_dict` lists every non-empty
non-agent cell as a wall too. ph2's v1 `Overcooked` masks these correctly.
"""
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


def _make(d):
    return FrozenDict({k: jnp.array(v) if isinstance(v, list) else v for k, v in d.items()})


CEC_LAYOUTS = {
    "forced_coord_9": _make({
        "height": 9,
        "width": 9,
        "agent_idx": [19, 21],
        "goal_idx": [39, 40],
        "onion_pile_idx": [9, 18],
        "plate_pile_idx": [27, 36],
        "pot_idx": [3, 13],
        "wall_idx": [
            0, 1, 2, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17, 20, 22, 23, 24, 25,
            26, 29, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 39, 40, 27, 36, 9, 18, 3, 13,
        ],
    }),
}
