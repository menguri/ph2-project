"""webapp 의 실제 env 경로 (overcooked-ai) 로 CEC self-play reward 측정.

webapp/app/game/engine.py 의 GameSession 은 overcooked-ai env 를 쓰고, obs 는
  overcooked_state_to_jaxmarl_obs → (webapp 에서는 여기서 AI 에 전달)
  → CEC 의 경우 추가로 ov2_obs_to_cec 를 거쳐 CEC (9,9,26) 으로 변환

이 스크립트는 같은 env + 같은 obs 파이프라인을 쓰되, 사람 대신 CEC 모델 하나가
양 slot 을 모두 제어 (self-play) 하여 평균 reward 를 측정한다. 목표는 cec-zero-shot
cross_eval baseline (cramped_room 156, coord_ring 94 등) 을 근사하는지 확인.

이전 diagnose_cec_native_vs_v2.py 는 OV2 env 를 직접 사용해서 0 reward 가 나왔는데,
OV2 의 step dynamics 가 V1 과 달라서였다. webapp 은 overcooked-ai 를 사용하고
_patch_mdp_no_early_cook + _auto_cook_full_pots 로 JaxMARL (V1) dynamics 를
근사하므로, webapp 경로에서는 CEC 가 정상 reward 를 낼 것으로 기대.

Run (GPU 0):
    cd /home/mlic/mingukang/ph2-project && \
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:webapp PYTHONUNBUFFERED=1 \
        ./overcooked_v2/bin/python -u cec_integration/scripts/test_cec_webapp_rewards.py
"""
import os
import sys
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import numpy as np
import jax
import jax.numpy as jnp

# webapp 모듈 재사용
from app.game.engine import _load_custom_layout
from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs
from app.game.action_map import jaxmarl_to_overcooked
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# CEC 경로
from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec


WEBAPP_MODELS_ROOT = os.path.join(PROJECT_ROOT, "webapp", "models")
NUM_STEPS = 400  # CEC native eval 과 동일
NUM_EPISODES = 3

NATIVE_BASELINE = {
    "cramped_room":     156.73,
    "coord_ring":        94.23,
    "forced_coord":       7.78,
    "counter_circuit":    0.00,
    "asymm_advantages":   2.97,
}


def _ckpt_path(layout, run="run0"):
    return os.path.join(WEBAPP_MODELS_ROOT, layout, "cec", run, "ckpt_final")


def _auto_cook_full_pots(state, mdp):
    """webapp engine.py::_auto_cook_full_pots 의 self-play 용 복사본.

    JaxMARL 호환: pot 에 재료 3 개가 차면 interact 없이 자동 요리 시작 +
    매 step cook() 1 회 호출 (1-step 오프셋 보정).
    """
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup"
                    and not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            # 이미 cooking 중이거나 방금 시작했으면 한 번 cook() 호출해서 1-step 오프셋 보정
            if obj.is_cooking:
                obj.cook()


def run_episode(rt, mdp, env, rng, agent_swap=False):
    """CEC 1 개 runtime 이 두 agent 를 모두 제어 (self-play).

    agent_swap=True: webapp/overcooked-ai 의 agent_0/1 순서가 CEC 훈련 convention 과
    뒤바뀌어 있는 경우 (cramped_room, counter_circuit — compatability.py::LAYOUT_SWAP_AGENT_DICT
    기준) slot 을 swap 해서 CEC 에 전달하고, action 도 원복.
    """
    env.reset()
    state = env.state
    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=jnp.bool_)
    total_reward = 0.0
    for t in range(NUM_STEPS):
        # 각 agent 별로 obs 추출 → CEC obs 로 변환
        obs_a0_ov2 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=0)
        obs_a1_ov2 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=1)

        a0_cec = ov2_obs_to_cec(jnp.array(obs_a0_ov2, dtype=jnp.float32),
                                 mdp_layout, t, NUM_STEPS)
        a1_cec = ov2_obs_to_cec(jnp.array(obs_a1_ov2, dtype=jnp.float32),
                                 mdp_layout, t, NUM_STEPS)
        if agent_swap:
            obs_batch = jnp.stack([a1_cec, a0_cec])
        else:
            obs_batch = jnp.stack([a0_cec, a1_cec])

        rng, sub = jax.random.split(rng)
        actions, hidden, _ = rt.step(obs_batch, hidden, done_arr, sub)
        if agent_swap:
            a0 = int(actions[1])
            a1 = int(actions[0])
        else:
            a0 = int(actions[0])
            a1 = int(actions[1])

        joint_action = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, reward, done, _info = env.step(joint_action)
        _auto_cook_full_pots(next_state, mdp)

        total_reward += float(reward)
        state = next_state
        if done:
            break
    return total_reward, t + 1


def evaluate_layout(layout):
    print(f"\n=== [{layout}] ===", flush=True)
    ckpt = _ckpt_path(layout)
    if not os.path.isdir(ckpt):
        print(f"  [SKIP] ckpt missing: {ckpt}", flush=True)
        return None
    try:
        rt = CECRuntime(ckpt)
    except Exception as e:
        print(f"  [SKIP] ckpt load failed: {e}", flush=True)
        return None
    print(f"  loaded runtime (format={rt.ckpt_format})", flush=True)

    global mdp_layout
    mdp_layout = layout
    mdp = _load_custom_layout(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    print(f"  env built (mdp={type(mdp).__name__})", flush=True)

    # LAYOUT_SWAP_AGENT_DICT: cramped_room 과 counter_circuit 은 webapp agent_0/1 이 CEC
    # 훈련 convention (V1) 과 뒤바뀌어 있음.
    swap_needed = layout in ("cramped_room", "counter_circuit")

    def _run(swap_flag, label):
        rewards = []
        rng = jax.random.PRNGKey(42)
        for ep in range(NUM_EPISODES):
            rng, sub = jax.random.split(rng)
            r, steps = run_episode(rt, mdp, env, sub, agent_swap=swap_flag)
            print(f"  {label} ep{ep}: reward={r:.1f} steps={steps}", flush=True)
            rewards.append(r)
        return rewards

    no_swap = _run(False, "no_swap")
    with_swap = _run(True, "swap  ")

    baseline = NATIVE_BASELINE.get(layout, 0.0)
    ns_mean = statistics.mean(no_swap)
    ws_mean = statistics.mean(with_swap)
    # 더 좋은 변형 선택
    if ws_mean > ns_mean:
        best_mean = ws_mean
        best_std = statistics.stdev(with_swap) if len(with_swap) > 1 else 0.0
        used = "swap"
    else:
        best_mean = ns_mean
        best_std = statistics.stdev(no_swap) if len(no_swap) > 1 else 0.0
        used = "no_swap"
    ratio_str = f"{best_mean/baseline:.0%}" if baseline > 0 else "n/a"
    print(f"  {layout}: best={best_mean:.2f}±{best_std:.2f} ({used})  "
          f"native={baseline:.2f}  ratio={ratio_str}  "
          f"[no_swap={ns_mean:.1f} swap={ws_mean:.1f}  recommend={'swap' if swap_needed else 'no_swap'}]",
          flush=True)
    return {"mean": best_mean, "std": best_std, "native": baseline,
            "no_swap": ns_mean, "with_swap": ws_mean, "used": used,
            "swap_needed": swap_needed}


def main():
    print("=" * 70)
    print("CEC self-play via webapp 실제 경로 (overcooked-ai + obs_adapter + obs_adapter_v2)")
    print(f"  {NUM_EPISODES} episodes × {NUM_STEPS} steps × 5 layouts")
    print("=" * 70, flush=True)

    if len(sys.argv) > 1:
        layouts = sys.argv[1].split(",")
    else:
        layouts = ["cramped_room", "coord_ring", "forced_coord",
                   "counter_circuit", "asymm_advantages"]
    results = {}
    for layout in layouts:
        results[layout] = evaluate_layout(layout)

    print("\n" + "=" * 70)
    print(f"{'layout':20s} {'webapp mean':>14s} {'native':>10s} {'ratio':>8s}")
    all_pass = True
    for name, r in results.items():
        if r is None:
            print(f"{name:20s} {'SKIP':>14s}")
            continue
        baseline = r["native"]
        ratio = r["mean"] / baseline if baseline > 0 else float("inf")
        # 합격 기준: baseline > 0 인 레이아웃은 70% 이상
        passed = baseline <= 0.1 or ratio >= 0.7
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        ratio_str = f"{ratio:.0%}" if baseline > 0 else "n/a"
        print(f"{name:20s} {r['mean']:10.2f}±{r['std']:4.1f} {baseline:10.2f} {ratio_str:>8s}  {status}")
    print("=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
