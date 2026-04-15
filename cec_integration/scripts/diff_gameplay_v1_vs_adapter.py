"""V1 native Overcooked 와 webapp 의 adapter chain 에서 같은 CEC self-play 궤적을
돌리면서 step 별 CEC obs 를 채널 단위로 비교.

전략:
  - V1 env 와 overcooked-ai env 를 각각 reset (같은 seed)
  - 매 step 에서 V1 obs 로 CEC 에 action 을 뽑는다 (V1 가 훈련 분포라 가장 안정).
  - 같은 action 을 양 env 에 적용 → 각 env 의 state 를 독립적으로 진행.
  - 동시에 overcooked-ai state 를 adapter chain 으로 통과시켜 "adapter CEC obs" 생성.
  - V1 obs (ground truth, CEC 가 보는 것) 와 adapter obs 를 채널별 diff.

두 env 의 state 가 동일하게 유지되려면 초기 상태 (agent 위치/방향) 가 맞아야 하고
dynamics 도 일치해야 한다. 불일치가 누적되면 state 가 divergence → obs 도 divergence.
처음 divergence 가 일어나는 step 과 채널을 찾는 것이 목적.

Run:
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/diff_gameplay_v1_vs_adapter.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
from app.game.engine import _load_custom_layout
from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs
from app.game.action_map import jaxmarl_to_overcooked
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec


CKPT_PATH = os.path.join(PROJECT_ROOT, "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final")
LAYOUT = "cramped_room"
NUM_STEPS = 30

CH_NAMES = [
    "self_pos","other_pos","self_E","self_S","self_W","self_N",
    "other_E","other_S","other_W","other_N",
    "pot","wall","onion_pile","tomato_pile","plate_pile","goal",
    "ons_in_pot","tom_in_pot","ons_in_soup","tom_in_soup",
    "cook_time","soup_ready","plate_on_grid","onion_on_grid","tomato_on_grid","urgency",
]


def _auto_cook(state, mdp):
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup" and not obj.is_cooking and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()


def _dump_state_summary(label, v1_obs_a0, adapter_obs_a0, v1_state=None, oa_state=None):
    """agent_0 기준으로 주요 정보 출력."""
    # V1 obs 기준
    s_v1 = np.argwhere(v1_obs_a0[:, :, 0] > 0.5)
    o_v1 = np.argwhere(v1_obs_a0[:, :, 1] > 0.5)
    # adapter 기준
    s_ad = np.argwhere(adapter_obs_a0[:, :, 0] > 0.5)
    o_ad = np.argwhere(adapter_obs_a0[:, :, 1] > 0.5)
    msg = f"{label}  v1:self={tuple(s_v1[0]) if len(s_v1)==1 else '?'},other={tuple(o_v1[0]) if len(o_v1)==1 else '?'}"
    msg += f"  adapter:self={tuple(s_ad[0]) if len(s_ad)==1 else '?'},other={tuple(o_ad[0]) if len(o_ad)==1 else '?'}"
    return msg


def _diff_channels(v1_obs, adapter_obs, threshold=0.01, top_k=4):
    """채널별 diff 요약."""
    diffs = []
    for ch in range(26):
        d = np.abs(v1_obs[:, :, ch] - adapter_obs[:, :, ch])
        max_d = float(d.max())
        if max_d > threshold:
            coords = np.argwhere(d > threshold)
            samples = []
            for (y, x) in coords[:top_k]:
                samples.append(f"({y},{x}):v1={v1_obs[y,x,ch]:.1f},ad={adapter_obs[y,x,ch]:.1f}")
            diffs.append((ch, CH_NAMES[ch], len(coords), samples))
    return diffs


def main():
    print("=" * 78)
    print(f"V1 native vs adapter chain step-by-step obs diff")
    print(f"ckpt={CKPT_PATH}")
    print(f"layout={LAYOUT}  num_steps={NUM_STEPS}")
    print("=" * 78, flush=True)

    rt = CECRuntime(CKPT_PATH)
    print("loaded CEC runtime", flush=True)

    # V1 env (CEC 훈련 env 그대로)
    v1_env = V1Overcooked(layout=CEC_LAYOUTS[f"{LAYOUT}_9"], random_reset=False,
                           max_steps=NUM_STEPS)
    # webapp 과 동일하게 overcooked-ai env
    mdp = _load_custom_layout(LAYOUT)
    oa_env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)

    # 동일 seed 로 reset
    rng = jax.random.PRNGKey(0)
    v1_obs, v1_state = v1_env.reset(rng)
    oa_env.reset()
    oa_state = oa_env.state

    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    rng_act = jax.random.PRNGKey(42)
    total_v1 = 0.0
    total_oa = 0.0
    first_state_divergence = None

    # agent_0 obs (V1 vs adapter) 비교
    v1_obs_a0 = np.asarray(v1_obs["agent_0"])
    obs_oa_a0 = overcooked_state_to_jaxmarl_obs(oa_state, mdp, agent_idx=0)
    adapter_obs_a0 = np.asarray(ov2_obs_to_cec(jnp.array(obs_oa_a0, dtype=jnp.float32),
                                                 LAYOUT, 0, NUM_STEPS))
    print(f"\n[t=0 reset] {_dump_state_summary('', v1_obs_a0, adapter_obs_a0)}", flush=True)
    diffs0 = _diff_channels(v1_obs_a0, adapter_obs_a0)
    if diffs0:
        print(f"  reset 시 {len(diffs0)} 채널 diff:", flush=True)
        for ch, name, n, samples in diffs0:
            print(f"    ch{ch:2d} {name:16s} n={n}  {samples[:2]}", flush=True)
    else:
        print(f"  reset 시 모든 26 채널 일치", flush=True)

    for t in range(NUM_STEPS):
        # CEC 에 V1 obs 로 action 뽑기 (V1 이 CEC 훈련 분포)
        v1_batch = jnp.stack([v1_obs["agent_0"].flatten(), v1_obs["agent_1"].flatten()])
        rng_act, k1 = jax.random.split(rng_act)
        actions, hidden, _ = rt.step(v1_batch, hidden, done_arr, k1)
        a0, a1 = int(actions[0]), int(actions[1])

        # V1 step
        rng, k_v1 = jax.random.split(rng)
        v1_act = {v1_env.agents[0]: a0, v1_env.agents[1]: a1}
        v1_obs, v1_state, v1_rew, v1_done, _ = v1_env.step(k_v1, v1_state, v1_act)
        total_v1 += float(v1_rew["agent_0"])

        # overcooked-ai step (같은 action)
        oa_act = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        oa_next, oa_rew, oa_done, _ = oa_env.step(oa_act)
        _auto_cook(oa_next, mdp)
        oa_state = oa_next
        total_oa += float(oa_rew)

        # obs 비교
        v1_obs_a0 = np.asarray(v1_obs["agent_0"])
        obs_oa_a0 = overcooked_state_to_jaxmarl_obs(oa_state, mdp, agent_idx=0)
        adapter_obs_a0 = np.asarray(ov2_obs_to_cec(jnp.array(obs_oa_a0, dtype=jnp.float32),
                                                     LAYOUT, t+1, NUM_STEPS))

        # 채널별 diff
        diffs = _diff_channels(v1_obs_a0, adapter_obs_a0)
        act_names = ["R", "D", "L", "U", "stay", "INT"]
        header = (f"[t={t+1:2d}] act=({act_names[a0]},{act_names[a1]})  "
                  f"v1_r={float(v1_rew['agent_0']):3.0f} oa_r={float(oa_rew):3.0f}  "
                  f"{_dump_state_summary('', v1_obs_a0, adapter_obs_a0)}")
        print(header, flush=True)
        if diffs:
            if first_state_divergence is None:
                first_state_divergence = t + 1
            for ch, name, n, samples in diffs:
                print(f"    ch{ch:2d} {name:16s} n={n}  {samples[:3]}", flush=True)

    print("\n" + "=" * 78)
    print(f"V1 total reward = {total_v1:.1f}")
    print(f"overcooked-ai total reward = {total_oa:.1f}")
    if first_state_divergence is not None:
        print(f"first obs divergence at step = {first_state_divergence}")
    else:
        print(f"no obs divergence across {NUM_STEPS} steps")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main() or 0)
