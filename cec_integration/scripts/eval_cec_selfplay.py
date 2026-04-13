"""CEC self-play / cross-play 평가 + 행동 분석.

forced_coord_9 체크포인트로 CEC × CEC 매칭 평가.
- self-play: 같은 ckpt끼리
- cross-play: 다른 seed/ckpt끼리
- 행동 분석: action 분포, delivery 횟수, 움직임 패턴
"""
import sys
import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "human-proxy", "code"))

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec

CKPT_DIR = os.path.join(PROJECT_ROOT, "cec_integration", "ckpts", "forced_coord_9")
LAYOUT = "forced_coord"
MAX_STEPS = 400
NUM_EVAL_SEEDS = 5  # CPU에서 빠르게 실행

ACTION_NAMES = ["RIGHT", "DOWN", "LEFT", "UP", "STAY", "INTERACT"]


def run_episode_detailed(rt0, rt1, env, rng):
    """에피소드 실행 + 상세 로그 수집."""
    obs_dict, env_state = env.reset(rng)
    hidden_0 = rt0.init_hidden(2)
    hidden_1 = rt1.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=bool)

    total_reward = 0.0
    step_rewards = []
    actions_log = {0: [], 1: []}  # agent별 action 기록
    delivery_steps = []  # delivery가 발생한 step

    for t in range(MAX_STEPS):
        rng, k1, k2, k_env = jax.random.split(rng, 4)

        # obs adapter v2 사용
        a0_obs = ov2_obs_to_cec(
            jnp.array(obs_dict["agent_0"], dtype=jnp.float32), LAYOUT, t, MAX_STEPS
        )
        a1_obs = ov2_obs_to_cec(
            jnp.array(obs_dict["agent_1"], dtype=jnp.float32), LAYOUT, t, MAX_STEPS
        )
        obs_batch = jnp.stack([a0_obs, a1_obs])

        # agent 0 추론 (rt0)
        a0_actions, hidden_0, _ = rt0.step(obs_batch, hidden_0, done_arr, k1)
        # agent 1 추론 (rt1)
        a1_actions, hidden_1, _ = rt1.step(obs_batch, hidden_1, done_arr, k2)

        env_act = {
            "agent_0": int(a0_actions[0]),
            "agent_1": int(a1_actions[1]),
        }
        actions_log[0].append(int(a0_actions[0]))
        actions_log[1].append(int(a1_actions[1]))

        obs_dict, env_state, reward, done, info = env.step(k_env, env_state, env_act)
        done_arr = jnp.array([done["agent_0"], done["agent_1"]])

        r = float(reward["agent_0"])
        step_rewards.append(r)
        total_reward += r
        if r > 0:
            delivery_steps.append(t)

        if bool(done["__all__"]):
            break

    return {
        "total_reward": total_reward,
        "num_steps": t + 1,
        "delivery_steps": delivery_steps,
        "num_deliveries": len(delivery_steps),
        "actions": actions_log,
        "step_rewards": step_rewards,
    }


def analyze_actions(actions_log, num_steps):
    """행동 패턴 분석."""
    analysis = {}
    for agent_id in [0, 1]:
        acts = actions_log[agent_id][:num_steps]
        counter = Counter(acts)
        total = len(acts)

        # 행동 분포
        dist = {ACTION_NAMES[a]: counter.get(a, 0) for a in range(6)}
        dist_pct = {k: v / total * 100 for k, v in dist.items()}

        # interact 비율
        interact_count = counter.get(5, 0)
        interact_pct = interact_count / total * 100

        # 이동 비율 (RIGHT, DOWN, LEFT, UP)
        move_count = sum(counter.get(a, 0) for a in range(4))
        move_pct = move_count / total * 100

        # STAY 비율
        stay_count = counter.get(4, 0)
        stay_pct = stay_count / total * 100

        # 반복 행동 (같은 action 연속)
        repeats = sum(1 for i in range(1, len(acts)) if acts[i] == acts[i - 1])
        repeat_pct = repeats / max(len(acts) - 1, 1) * 100

        analysis[agent_id] = {
            "dist": dist,
            "dist_pct": dist_pct,
            "interact_pct": interact_pct,
            "move_pct": move_pct,
            "stay_pct": stay_pct,
            "repeat_pct": repeat_pct,
        }
    return analysis


def main() -> int:
    print("=" * 70)
    print("CEC Self-Play / Cross-Play 평가 + 행동 분석")
    print(f"Layout: {LAYOUT}, Max Steps: {MAX_STEPS}, Eval Seeds: {NUM_EVAL_SEEDS}")
    print("=" * 70)

    env = jaxmarl.make(
        "overcooked_v2",
        layout=LAYOUT,
        max_steps=MAX_STEPS,
        random_reset=True,
        random_agent_positions=True,
    )
    print(f"Env: {LAYOUT} h={env.height} w={env.width}\n")

    # 체크포인트 로드
    from pathlib import Path
    ckpt_path = Path(CKPT_DIR)
    ckpt_names = sorted([d.name for d in ckpt_path.iterdir()
                         if d.is_dir() and "improved" in d.name])
    print(f"체크포인트 {len(ckpt_names)}개 로드 중...")

    runtimes = {}
    for name in ckpt_names:
        rt = CECRuntime(str(ckpt_path / name))
        runtimes[name] = rt
    print(f"로드 완료: {len(runtimes)}개\n")

    # ═══════════════════════════════════════════════════
    # 1. Self-Play (seed별 대표 ckpt0 + 몇 개 추가)
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("[1] SELF-PLAY (같은 ckpt × 같은 ckpt)")
    print("=" * 70)

    # 대표 ckpt만 선별 (ckpt0 위주 + 일부 ckpt1/2)
    sp_ckpts = [n for n in ckpt_names if "ckpt0" in n]
    sp_ckpts += [n for n in ckpt_names if "seed11_ckpt1" in n or "seed11_ckpt2" in n]

    sp_results = {}
    for name in sp_ckpts:
        rt = runtimes[name]
        rewards = []
        deliveries = []
        for seed in range(NUM_EVAL_SEEDS):
            rng = jax.random.PRNGKey(seed * 1000)
            result = run_episode_detailed(rt, rt, env, rng)
            rewards.append(result["total_reward"])
            deliveries.append(result["num_deliveries"])
        sp_results[name] = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_deliveries": np.mean(deliveries),
            "rewards": rewards,
        }
        print(f"  {name}: mean={np.mean(rewards):.1f}")

    print(f"\n{'Checkpoint':40s} | {'Mean Reward':>12s} | {'Std':>8s} | {'Deliveries':>10s}")
    print("-" * 78)
    all_sp_rewards = []
    for name in sp_ckpts:
        r = sp_results[name]
        print(f"{name:40s} | {r['mean_reward']:12.1f} | {r['std_reward']:8.1f} | {r['mean_deliveries']:10.1f}")
        all_sp_rewards.extend(r["rewards"])
    print("-" * 78)
    print(f"{'전체 평균':40s} | {np.mean(all_sp_rewards):12.1f} | {np.std(all_sp_rewards):8.1f} |")

    # ═══════════════════════════════════════════════════
    # 2. Cross-Play (다른 seed끼리)
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("[2] CROSS-PLAY (다른 ckpt × 다른 ckpt, 샘플)")
    print("=" * 70)

    # 대표 ckpt 선택 (seed별 ckpt0, 최대 4개)
    representative = [n for n in ckpt_names if "ckpt0" in n][:4]
    if len(representative) < 2:
        representative = ckpt_names[:4]

    xp_rewards = []
    xp_details = []
    for i, name_a in enumerate(representative):
        for j, name_b in enumerate(representative):
            if i >= j:
                continue
            rewards = []
            for seed in range(NUM_EVAL_SEEDS):
                rng = jax.random.PRNGKey(seed * 1000 + 500)
                result = run_episode_detailed(runtimes[name_a], runtimes[name_b], env, rng)
                rewards.append(result["total_reward"])
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            xp_rewards.extend(rewards)
            xp_details.append((name_a, name_b, mean_r, std_r))

    print(f"\n{'Agent 0':30s} | {'Agent 1':30s} | {'Mean':>8s} | {'Std':>8s}")
    print("-" * 84)
    for a, b, m, s in xp_details:
        a_short = a.replace("_improved", "")
        b_short = b.replace("_improved", "")
        print(f"{a_short:30s} | {b_short:30s} | {m:8.1f} | {s:8.1f}")
    print("-" * 84)
    print(f"{'Cross-play 전체 평균':62s} | {np.mean(xp_rewards):8.1f} | {np.std(xp_rewards):8.1f}")

    # ═══════════════════════════════════════════════════
    # 3. 행동 분석 (대표 에피소드)
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("[3] 행동 패턴 분석 (대표 에피소드)")
    print("=" * 70)

    # 가장 점수 높은 ckpt와 가장 낮은 ckpt 비교
    best_name = max(sp_results.keys(), key=lambda n: sp_results[n]["mean_reward"])
    worst_name = min(sp_results.keys(), key=lambda n: sp_results[n]["mean_reward"])

    for label, name in [("BEST", best_name), ("WORST", worst_name)]:
        print(f"\n--- {label}: {name} (mean={sp_results[name]['mean_reward']:.1f}) ---")
        rng = jax.random.PRNGKey(42)
        result = run_episode_detailed(runtimes[name], runtimes[name], env, rng)
        analysis = analyze_actions(result["actions"], result["num_steps"])

        print(f"  Total reward: {result['total_reward']:.0f}")
        print(f"  Deliveries: {result['num_deliveries']} at steps {result['delivery_steps']}")
        print(f"  Steps used: {result['num_steps']}")

        for agent_id in [0, 1]:
            a = analysis[agent_id]
            print(f"\n  Agent {agent_id}:")
            print(f"    Action distribution:")
            for act_name, pct in a["dist_pct"].items():
                bar = "#" * int(pct / 2)
                print(f"      {act_name:10s}: {a['dist'][act_name]:4d} ({pct:5.1f}%) {bar}")
            print(f"    Move: {a['move_pct']:.1f}% | Stay: {a['stay_pct']:.1f}% | Interact: {a['interact_pct']:.1f}%")
            print(f"    Action repeat rate: {a['repeat_pct']:.1f}%")

    # ═══════════════════════════════════════════════════
    # 4. 첫 50 step 행동 시퀀스 (게임 해결 흐름 분석)
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("[4] 첫 50 step 행동 시퀀스 (BEST ckpt)")
    print("=" * 70)

    rng = jax.random.PRNGKey(42)
    result = run_episode_detailed(runtimes[best_name], runtimes[best_name], env, rng)

    show_steps = min(50, result["num_steps"])
    print(f"\nStep | A0 Action  | A1 Action  | Reward | Cumul")
    print("-" * 55)
    cumul = 0.0
    for t in range(show_steps):
        a0 = ACTION_NAMES[result["actions"][0][t]]
        a1 = ACTION_NAMES[result["actions"][1][t]]
        r = result["step_rewards"][t]
        cumul += r
        marker = " ***" if r > 0 else ""
        print(f"{t:4d} | {a0:10s} | {a1:10s} | {r:6.0f} | {cumul:5.0f}{marker}")

    # ═══════════════════════════════════════════════════
    # 5. 게임 해결 능력 종합 판단
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("[5] 게임 해결 능력 종합 판단")
    print("=" * 70)

    sp_mean = np.mean(all_sp_rewards)
    xp_mean = np.mean(xp_rewards) if xp_rewards else 0.0

    # 기준 점수 (forced_coord 기준, 다른 알고리즘 참고용)
    print(f"\n  Self-play  전체 평균: {sp_mean:.1f}")
    print(f"  Cross-play 전체 평균: {xp_mean:.1f}")

    # delivery가 있는 에피소드 비율
    sp_has_delivery = sum(1 for name in sp_results for r in sp_results[name]["rewards"] if r > 0)
    sp_total_episodes = len(sp_results) * NUM_EVAL_SEEDS
    delivery_rate = sp_has_delivery / sp_total_episodes * 100
    print(f"  Delivery 발생 에피소드 비율: {sp_has_delivery}/{sp_total_episodes} ({delivery_rate:.1f}%)")

    # 행동 다양성 체크 (best ckpt 기준)
    rng = jax.random.PRNGKey(42)
    result = run_episode_detailed(runtimes[best_name], runtimes[best_name], env, rng)
    analysis = analyze_actions(result["actions"], result["num_steps"])

    interact_pct_avg = np.mean([analysis[a]["interact_pct"] for a in [0, 1]])
    stay_pct_avg = np.mean([analysis[a]["stay_pct"] for a in [0, 1]])
    repeat_pct_avg = np.mean([analysis[a]["repeat_pct"] for a in [0, 1]])

    print(f"\n  BEST ckpt 행동 분석:")
    print(f"    평균 INTERACT 비율: {interact_pct_avg:.1f}%")
    print(f"    평균 STAY 비율: {stay_pct_avg:.1f}%")
    print(f"    평균 반복행동 비율: {repeat_pct_avg:.1f}%")

    # 판단
    print(f"\n  판단:")
    if sp_mean >= 40:
        print(f"    ✓ Self-play 평균 {sp_mean:.0f} — 충분한 게임 해결 능력")
    elif sp_mean >= 15:
        print(f"    △ Self-play 평균 {sp_mean:.0f} — 부분적 게임 해결 (개선 필요)")
    else:
        print(f"    ✗ Self-play 평균 {sp_mean:.0f} — 게임 해결 불충분")

    if interact_pct_avg >= 5:
        print(f"    ✓ INTERACT {interact_pct_avg:.1f}% — 적절한 상호작용")
    else:
        print(f"    ✗ INTERACT {interact_pct_avg:.1f}% — 상호작용 부족")

    if stay_pct_avg <= 30:
        print(f"    ✓ STAY {stay_pct_avg:.1f}% — 적절한 활동량")
    else:
        print(f"    ✗ STAY {stay_pct_avg:.1f}% — 정체 상태 과다")

    if repeat_pct_avg <= 50:
        print(f"    ✓ 반복행동 {repeat_pct_avg:.1f}% — 다양한 행동")
    else:
        print(f"    ✗ 반복행동 {repeat_pct_avg:.1f}% — 행동 다양성 부족 (루프 가능성)")

    if delivery_rate >= 30:
        print(f"    ✓ Delivery 발생률 {delivery_rate:.0f}% — 게임 목표 달성")
    else:
        print(f"    ✗ Delivery 발생률 {delivery_rate:.0f}% — 배달 성공 빈도 낮음")

    print(f"\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
