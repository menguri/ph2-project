#!/usr/bin/env python3
"""human-proxy 의 CEC × BC eval 환경에서:
  1. CECPolicy 가 받는 obs 가 "자기 agent 의 self-perspective" 인지 확인
  2. agent_idx 에 따라 올바른 obs 가 CEC 에 전달되는지
  3. adapter (ov2_obs_to_cec) 가 변환 후 CEC 에 넘어가는 obs 의 self/other 좌표 확인
"""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "human-proxy" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from policy import setup_pythonpath, BCPolicy, CECPolicy


LAYOUT = "cramped_room"


def main():
    setup_pythonpath("baseline")
    # import after pythonpath setup
    from overcooked_v2_experiments.eval.policy import PolicyPairing
    from overcooked_v2_experiments.eval.utils import make_eval_env

    # 환경 reset
    env, _, _ = make_eval_env(LAYOUT, {"max_steps": 100})
    obs_dict, state = env.reset(jax.random.PRNGKey(0))

    # OV2 state 에서 agent 위치 확인
    print("=" * 60)
    print(f"OV2 {LAYOUT} reset state")
    print("=" * 60)
    agents = state.agents
    for i in range(2):
        px = int(agents.pos.x[i])
        py = int(agents.pos.y[i])
        d = int(agents.dir[i])
        print(f"  agent_{i}: pos=({px},{py}) dir_idx={d}")

    # 각 agent 의 OV2 obs 의 self_pos / other_pos 채널 확인
    print(f"\nOV2 obs channels — self_pos(ch0), other_pos(ch8) for each agent perspective:")
    for agent_key in ["agent_0", "agent_1"]:
        obs = np.asarray(obs_dict[agent_key])
        self_yx = np.argwhere(obs[:, :, 0] > 0.5)
        other_yx = np.argwhere(obs[:, :, 8] > 0.5)
        print(f"  {agent_key} obs → self@{tuple(self_yx[0]) if len(self_yx) else '?'} "
              f"other@{tuple(other_yx[0]) if len(other_yx) else '?'}")

    # CECPolicy 두 개 (양 slot 씌워보기)
    cec_ckpt = PROJECT_ROOT / "webapp" / "models" / LAYOUT / "cec" / "run0" / "ckpt_final"
    print(f"\nCEC ckpt: {cec_ckpt}")
    cec_policy_A = CECPolicy(str(cec_ckpt), LAYOUT, stochastic=True)
    cec_policy_B = CECPolicy(str(cec_ckpt), LAYOUT, stochastic=True)

    # CECPolicy.compute_action 을 직접 호출 해서 어떻게 obs 가 들어가는지 추적
    for agent_key in ["agent_0", "agent_1"]:
        print(f"\n--- CECPolicy called as {agent_key} ---")
        obs_input = obs_dict[agent_key]
        hstate = cec_policy_A.init_hstate()
        done = jnp.array(False)
        rng = jax.random.PRNGKey(0)

        # 내부 변환 추적: cec_obs = adapter(obs_input)
        from cec_integration.obs_adapter_v2 import ov2_obs_to_cec
        cec_obs = np.asarray(
            ov2_obs_to_cec(jnp.array(obs_input, dtype=jnp.float32), LAYOUT, 0, 400)
        )
        self_cec = np.argwhere(cec_obs[:, :, 0] > 0.5)
        other_cec = np.argwhere(cec_obs[:, :, 1] > 0.5)
        print(f"  adapter 출력 CEC obs → self@{tuple(self_cec[0]) if len(self_cec) else '?'} "
              f"other@{tuple(other_cec[0]) if len(other_cec) else '?'}")
        print(f"  (OV2 {agent_key} 의 self={tuple(np.argwhere(np.asarray(obs_input)[:,:,0]>0.5)[0])} "
              f"와 일치해야 함)")

        # compute_action 호출
        action, new_hstate, _ = cec_policy_A.compute_action(obs_input, done, hstate, rng)
        print(f"  CEC action: {int(action)}")

    # PolicyPairing 경유 — BC(slot 0) + CEC(slot 1)
    print(f"\n--- PolicyPairing 경유 검증 ---")

    # BC pos_0 policy 하나 로드
    bc_dir = PROJECT_ROOT / "human-proxy" / "models" / LAYOUT / "pos_0" / "seed_0"
    bc_policy = BCPolicy.from_pretrained(bc_dir)
    print(f"BC ckpt: {bc_dir}")

    pairing = PolicyPairing(bc_policy, cec_policy_B)  # BC=slot0, CEC=slot1

    # rollout init
    from overcooked_v2_experiments.eval.rollout import init_rollout
    init_hstate_fn, get_actions_fn = init_rollout(pairing, env)
    hstates = init_hstate_fn(1)
    print(f"  hstates keys: {list(hstates.keys())}")
    print(f"  CEC hstate type: {type(hstates['agent_1']).__name__}")

    # 한 step 시뮬레이션
    done = {"agent_0": jnp.array(False), "agent_1": jnp.array(False)}
    rng = jax.random.split(jax.random.PRNGKey(1), 2)
    key_sample = {"agent_0": rng[0], "agent_1": rng[1]}
    actions, next_hstate, extras = get_actions_fn(obs_dict, done, hstates, key_sample)
    print(f"  actions: agent_0={actions['agent_0']} (BC), agent_1={actions['agent_1']} (CEC)")

    # CEC 가 받은 obs 는 obs_dict['agent_1'] 이어야 함. 간접 확인:
    # CECPolicy.compute_action 호출 시 obs=obs_dict['agent_1'] 이 들어감
    # rollout.py:50 `obs[agent_id]` 로 분기하므로 정확함
    print(f"\n✓ rollout.py 의 분기 로직: obs[agent_id] → CEC(slot=agent_1) 는 obs_dict['agent_1'] 받음")
    print(f"  OV2 env 가 agent_1 기준 obs 제공 → self=agent_1 위치, other=agent_0 위치")
    print(f"  이 obs 가 adapter 거쳐 CEC slot 0 으로 forward. action 은 agent_1 에 적용됨")

    # swap 필요 체크 — cramped_room 은 swap_agents=True
    print(f"\n--- swap_agents=True 영향 ---")
    print(f"  OV2 {LAYOUT} 의 agent_0 은 webapp/overcooked-ai 기준 (3,1), V1 기준 (1,3)")
    print(f"  즉 webapp agent_0 ≠ V1 agent_0. 그러나 CEC 는 IK 훈련(랜덤 위치)이라 slot 무관하게 동작.")


if __name__ == "__main__":
    main()
