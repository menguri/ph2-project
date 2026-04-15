"""webapp 의 overcooked-ai env 를 돌리면서, 각 step 에서:
  1. overcooked-ai state (pot ingredients, agents holding, cook_time, etc) 실측 값
  2. obs_adapter.py 가 만드는 OV2-format 30ch obs
  3. obs_adapter_v2 가 만드는 CEC 26ch obs
  4. CEC 가 내는 action
  5. 그 action 이 env 에 주는 효과

를 출력해서, 어느 시점에 state 와 obs 가 불일치하는지 또는 CEC 가 잘못된 행동을
하는지 눈으로 확인.

특히 주목: pot 근처에서 interact 했을 때 pot state 업데이트가 adapter 출력에 반영되는가.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/trace_webapp_gameplay.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np

from app.game.engine import _load_custom_layout


def _auto_cook_full_pots(state, mdp):
    """engine.py::GameSession._auto_cook_full_pots 와 동일 로직."""
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup"
                    and not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()
from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs
from app.game.action_map import jaxmarl_to_overcooked
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec


CKPT_PATH = os.path.join(PROJECT_ROOT, "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final")
LAYOUT = "cramped_room"
NUM_STEPS = 40


def _describe_state(state):
    """overcooked_ai state 를 간결하게 문자열로."""
    pieces = []
    for i, p in enumerate(state.players):
        pos = p.position
        held = p.held_object.name if p.held_object else "none"
        pieces.append(f"p{i}@{pos} held={held}")
    # pots
    for obj in state.all_objects_list:
        if obj.name == "soup":
            ings = [i for i in obj.ingredients]
            pieces.append(f"pot@{obj.position} ings={ings} cooking={obj.is_cooking} ready={obj.is_ready} cook_time={obj.cook_time}")
    return "  ".join(pieces)


def _describe_cec_obs(cec_obs, label):
    """CEC obs 의 주요 정보만 요약."""
    s_yx = np.argwhere(cec_obs[:, :, 0] > 0.5)
    o_yx = np.argwhere(cec_obs[:, :, 1] > 0.5)
    # pot state (at ALL pot positions)
    pot_yx = np.argwhere(cec_obs[:, :, 10] > 0.5)
    pot_info = []
    for (py, px) in pot_yx:
        in_pot = cec_obs[py, px, 16]
        in_soup = cec_obs[py, px, 18]
        cook = cec_obs[py, px, 20]
        ready = cec_obs[py, px, 21]
        if in_pot + in_soup + cook + ready > 0:
            pot_info.append(f"pot({py},{px}):p={in_pot:.0f}s={in_soup:.0f}c={cook:.0f}r={ready:.0f}")
    # self inventory (at self pos)
    if len(s_yx):
        sy, sx = s_yx[0]
        plate = cec_obs[sy, sx, 22]
        onion = cec_obs[sy, sx, 23]
        ready = cec_obs[sy, sx, 21]
        inv = f"self@{sy},{sx}:plate={plate:.0f}onion={onion:.0f}ready={ready:.0f}"
    else:
        inv = "self:?"
    return f"{label}: {inv}  " + "  ".join(pot_info) if pot_info else f"{label}: {inv}"


def main():
    rt = CECRuntime(CKPT_PATH)
    print(f"loaded runtime", flush=True)

    mdp = _load_custom_layout(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    env.reset()
    state = env.state
    print(f"reset: {_describe_state(state)}", flush=True)
    print(f"pot locations: {mdp.get_pot_locations()}", flush=True)

    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(42)
    total = 0.0

    for t in range(NUM_STEPS):
        obs_a0 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=0)
        obs_a1 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=1)
        a0_cec = ov2_obs_to_cec(jnp.array(obs_a0, dtype=jnp.float32), LAYOUT, t, NUM_STEPS)
        a1_cec = ov2_obs_to_cec(jnp.array(obs_a1, dtype=jnp.float32), LAYOUT, t, NUM_STEPS)
        obs_batch = jnp.stack([a0_cec, a1_cec])
        rng, sub = jax.random.split(rng)
        actions, hidden, _ = rt.step(obs_batch, hidden, done_arr, sub)
        a0 = int(actions[0])
        a1 = int(actions[1])

        joint = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, reward, done, _ = env.step(joint)
        _auto_cook_full_pots(next_state, mdp)
        total += float(reward)

        cec_a0_summary = _describe_cec_obs(np.asarray(a0_cec), "a0")

        act_names = ["R", "D", "L", "U", "stay", "INT"]
        print(f"t={t:2d} acts=({act_names[a0]},{act_names[a1]}) r={reward:.0f}  "
              f"{_describe_state(next_state)}  |  {cec_a0_summary}", flush=True)

        state = next_state
        if done:
            break
    print(f"\ntotal reward = {total:.1f}", flush=True)


if __name__ == "__main__":
    main()
