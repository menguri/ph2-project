"""OV2StateToCECDirectAdapter 검증 + CEC reward 측정.

(A) 정적 + gameplay: adapter 출력이 CEC_LAYOUTS ground truth 와 일치 (byte-exact 증명)
(B) CEC self-play reward: OV2 env 에서 direct adapter 경유로 CEC 평가 (dynamics drift 확인)
"""
import os, sys, statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS
from cec_integration.obs_adapter_v2_state_direct import OV2StateToCECDirectAdapter


LAYOUT = "cramped_room"
NUM_STEPS = 400
NUM_EPISODES = 3
CKPT = os.path.join(PROJECT_ROOT, "webapp", "models", LAYOUT, "cec", "run0", "ckpt_final")


def test_static():
    """Reset 시 정적 채널이 CEC_LAYOUTS ground truth 와 일치."""
    print("=" * 70)
    print(f"[test_static] OV2 state direct adapter — layout={LAYOUT}")
    print("=" * 70, flush=True)

    adapter = OV2StateToCECDirectAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)
    env = jaxmarl.make("overcooked_v2", layout=LAYOUT, max_steps=NUM_STEPS,
                       random_reset=False, random_agent_positions=False)
    obs_dict, state = env.reset(jax.random.PRNGKey(0))

    cec_obs = np.asarray(adapter.get_cec_obs(state, agent_idx=0))
    truth = CEC_LAYOUTS[f"{LAYOUT}_9"]

    def _flat_to_yx(flat):
        return {(int(i) // 9, int(i) % 9) for i in flat}

    expected = {
        10: ("pot", _flat_to_yx(truth["pot_idx"])),
        12: ("onion_pile", _flat_to_yx(truth["onion_pile_idx"])),
        14: ("plate_pile", _flat_to_yx(truth["plate_pile_idx"])),
        15: ("goal", _flat_to_yx(truth["goal_idx"])),
    }
    all_pass = True
    for ch, (name, exp) in expected.items():
        got = {(int(y), int(x)) for y, x in np.argwhere(cec_obs[:, :, ch] > 0.5)}
        if got == exp:
            print(f"  [PASS] ch{ch} ({name}): {sorted(got)}")
        else:
            print(f"  [FAIL] ch{ch} ({name}): got={sorted(got)}, exp={sorted(exp)}")
            all_pass = False

    s_sum = float(cec_obs[:, :, 0].sum())
    o_sum = float(cec_obs[:, :, 1].sum())
    print(f"  self_pos sum={s_sum}, other_pos sum={o_sum} "
          f"({'PASS' if s_sum == 1.0 and o_sum == 1.0 else 'FAIL'})")
    if not (s_sum == 1.0 and o_sum == 1.0):
        all_pass = False

    print(f"\n[static] {'ALL PASS' if all_pass else 'FAIL'}", flush=True)
    return all_pass


def test_reward():
    """CEC self-play via direct adapter in OV2 env."""
    print("\n" + "=" * 70)
    print(f"[test_reward] CEC self-play via OV2 direct adapter")
    print("=" * 70, flush=True)

    rt = CECRuntime(CKPT)
    adapter = OV2StateToCECDirectAdapter(target_layout=LAYOUT, max_steps=NUM_STEPS)
    env = jaxmarl.make("overcooked_v2", layout=LAYOUT, max_steps=NUM_STEPS,
                       random_reset=False, random_agent_positions=False)

    rng = jax.random.PRNGKey(42)
    rewards = []
    for ep in range(NUM_EPISODES):
        obs_dict, state = env.reset(rng)
        hidden = rt.init_hidden(2)
        done_arr = jnp.zeros((2,), dtype=jnp.bool_)
        total = 0.0
        for t in range(NUM_STEPS):
            cec_obs_dict = adapter.get_cec_obs_both(state, current_step=t)
            obs_batch = jnp.stack([cec_obs_dict["agent_0"], cec_obs_dict["agent_1"]])

            rng, sub = jax.random.split(rng)
            actions, hidden, _ = rt.step(obs_batch, hidden, done_arr, sub)
            env_act = {"agent_0": jnp.int32(actions[0]), "agent_1": jnp.int32(actions[1])}
            rng, k_env = jax.random.split(rng)
            obs_dict, state, reward, done, _ = env.step(k_env, state, env_act)
            total += float(reward["agent_0"])
            if bool(done["__all__"]):
                break
        print(f"  ep{ep}: reward={total:.1f} steps={t+1}", flush=True)
        rewards.append(total)

    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    print(f"  OV2 direct-adapter self-play: mean={mean:.2f} ± {std:.2f}")
    print(f"  (native V1 baseline=220, cross-eval baseline=156)")
    return mean


def main():
    static_ok = test_static()
    reward = test_reward()
    print("\n" + "=" * 70)
    print(f"Summary: static={static_ok}, reward_mean={reward:.2f}")
    print("=" * 70)
    return 0 if static_ok else 1


if __name__ == "__main__":
    sys.exit(main())
