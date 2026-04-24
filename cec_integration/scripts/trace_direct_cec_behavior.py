"""신규 `OvercookedAIToCECAdapter` 로 CEC self-play 돌리면서 각 step 의 action,
agent 위치/방향, 들고 있는 아이템, pot 상태, reward 를 로그.

목적: reward=0 은 맞지만 CEC 의 행동이 의미 있는지 (예: 양파 픽업→pot→plate→soup→배달
시도 하는지) 분석. 의미 있는 행동이면 문제는 dynamics drift 누적. 그렇지 않으면
adapter 나 agent 에 다른 bug 가능.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from app.game.engine import _load_custom_layout
from app.game.action_map import jaxmarl_to_overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_from_ai import OvercookedAIToCECAdapter


CKPT = os.path.join(PROJECT_ROOT, "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final")
LAYOUT = "cramped_room"
NUM_STEPS = 120
ACT_NAMES = ["R", "D", "L", "U", "stay", "INT"]


def _auto_cook(state, mdp):
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup" and not obj.is_cooking and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()


def _describe_pot(state, mdp):
    pieces = []
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if obj.name == "soup":
                ings = [i for i in obj.ingredients]
                pieces.append(f"pot@{pos}={len(ings)}o,cook={obj.is_cooking},ready={obj.is_ready},tick={getattr(obj,'_cooking_tick',-1)}")
            else:
                pieces.append(f"pot@{pos}={obj.name}")
        else:
            pieces.append(f"pot@{pos}=empty")
    return " ".join(pieces)


def _describe_player(p, idx):
    held = p.held_object.name if p.held_object else "_"
    return f"p{idx}@{p.position}face={p.orientation}hold={held}"


def main():
    print(f"ckpt={CKPT}", flush=True)
    rt = CECRuntime(CKPT)
    adapter = OvercookedAIToCECAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)
    mdp = _load_custom_layout(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    env.reset()
    state = env.state
    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(42)
    total = 0.0

    # 이벤트 카운터
    events = {
        "onion_pickup": 0,
        "onion_to_pot": 0,
        "plate_pickup": 0,
        "soup_pickup": 0,
        "delivery": 0,
    }
    prev_held = [None, None]
    prev_pot_ings = {pos: 0 for pos in mdp.get_pot_locations()}

    print(f"\n{'t':>3s} {'act':>8s} {'r':>4s}  state", flush=True)
    print(f"{'':>3s} {'':>8s} {'':>4s}  reset: {_describe_player(state.players[0], 0)}  {_describe_player(state.players[1], 1)}  {_describe_pot(state, mdp)}", flush=True)

    for t in range(NUM_STEPS):
        obs_dict = adapter.get_cec_obs_both(state, mdp, current_step=t)
        obs_batch = jnp.stack([obs_dict["agent_0"], obs_dict["agent_1"]])
        rng, sub = jax.random.split(rng)
        actions, hidden, _ = rt.step(obs_batch, hidden, done_arr, sub)
        a0, a1 = int(actions[0]), int(actions[1])

        joint = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, reward, done, _ = env.step(joint)
        _auto_cook(next_state, mdp)
        total += float(reward)

        # 이벤트 감지
        for i in range(2):
            prev_h = prev_held[i]
            curr_h = next_state.players[i].held_object.name if next_state.players[i].held_object else None
            if prev_h is None and curr_h == "onion":
                events["onion_pickup"] += 1
            if prev_h == "onion" and curr_h is None:
                # onion 이 사라짐 → pot 에 넣었거나 counter 에 놓음
                # pot 에 재료 증가 체크
                for pos in mdp.get_pot_locations():
                    if next_state.has_object(pos):
                        obj = next_state.get_object(pos)
                        if obj.name == "soup":
                            new_ings = len(obj.ingredients)
                            if new_ings > prev_pot_ings.get(pos, 0):
                                events["onion_to_pot"] += 1
                                break
            if prev_h is None and curr_h == "dish":
                events["plate_pickup"] += 1
            if prev_h == "dish" and curr_h == "soup":
                events["soup_pickup"] += 1
            prev_held[i] = curr_h

        if reward > 0:
            events["delivery"] += 1

        # pot 재료 추적
        for pos in mdp.get_pot_locations():
            if next_state.has_object(pos):
                obj = next_state.get_object(pos)
                if obj.name == "soup":
                    prev_pot_ings[pos] = len(obj.ingredients)
            else:
                prev_pot_ings[pos] = 0

        state = next_state

        print(f"{t+1:3d} ({ACT_NAMES[a0]:>2s},{ACT_NAMES[a1]:>2s}) r={reward:>3.0f}  "
              f"{_describe_player(state.players[0], 0)}  {_describe_player(state.players[1], 1)}  "
              f"{_describe_pot(state, mdp)}", flush=True)

        if done:
            break

    print("\n" + "=" * 70)
    print(f"Total reward = {total:.1f} in {t+1} steps")
    print(f"\n행동 통계:")
    for k, v in events.items():
        print(f"  {k:15s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
