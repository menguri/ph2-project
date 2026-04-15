"""V1 native 에서 CEC self-play 로 얻은 action 궤적을 overcooked-ai (webapp) env 에
재생해서 reward 가 비슷하게 나오는지 확인.

목적: overcooked-ai 의 dynamics 가 V1 과 충분히 일치하는지 검증.
  - V1 에서 CEC 가 220 점 달성
  - 같은 action 궤적을 overcooked-ai 에 넣었을 때도 비슷한 점수면 dynamics 동일.
  - 점수가 떨어지면 overcooked-ai dynamics 가 V1 과 달라서 webapp 에서 CEC 실패.

agent 초기 위치 차이 해결: V1 agent_idx 를 webapp overcooked-ai 와 동일하게 맞춘다
(V1 에서 agent_0 을 (1,3) 에 두어 overcooked-ai 와 동일 시작).
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "webapp"))

import jax, jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxmarl.environments.overcooked.overcooked import Overcooked as V1Overcooked
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from app.game.engine import _load_custom_layout
from app.game.action_map import jaxmarl_to_overcooked

from cec_integration.cec_runtime import CECRuntime
from cec_integration.cec_layouts import CEC_LAYOUTS


CKPT = os.path.join(PROJECT_ROOT, "webapp", "models", "cramped_room", "cec", "run0", "ckpt_final")
NUM_STEPS = 400


def _auto_cook(state, mdp):
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            obj = state.get_object(pos)
            if (obj.name == "soup" and not obj.is_cooking and not obj.is_ready
                    and len(obj.ingredients) >= 3):
                obj.begin_cooking()
            if obj.is_cooking:
                obj.cook()


def main():
    rt = CECRuntime(CKPT)
    print("loaded CEC", flush=True)

    # V1: webapp 과 동일한 agent 배치 (agent_0 at (1,3), agent_1 at (1,1))
    base = CEC_LAYOUTS["cramped_room_9"]
    orig_agent = list(map(int, base["agent_idx"]))
    layout_v1 = FrozenDict({**dict(base), "agent_idx": jnp.array([orig_agent[1], orig_agent[0]], dtype=jnp.int32)})
    v1_env = V1Overcooked(layout=layout_v1, random_reset=False, max_steps=NUM_STEPS)

    # V1 reset + rollout — action 기록
    rng = jax.random.PRNGKey(42)
    v1_obs, v1_state = v1_env.reset(rng)
    h = rt.init_hidden(2)
    d = jnp.zeros((2,), dtype=jnp.bool_)
    v1_actions = []  # list of (a0, a1)
    v1_total = 0.0
    for t in range(NUM_STEPS):
        batch = jnp.stack([v1_obs["agent_0"].flatten(), v1_obs["agent_1"].flatten()])
        rng, k1, ke = jax.random.split(rng, 3)
        act, h, _ = rt.step(batch, h, d, k1)
        a0, a1 = int(act[0]), int(act[1])
        v1_actions.append((a0, a1))
        env_act = {v1_env.agents[0]: a0, v1_env.agents[1]: a1}
        v1_obs, v1_state, r, dn, _ = v1_env.step(ke, v1_state, env_act)
        d = jnp.array([dn[a] for a in v1_env.agents])
        v1_total += float(r["agent_0"])
        if bool(dn["__all__"]):
            break
    print(f"V1 native (swapped agent_idx): reward={v1_total:.1f}, steps={len(v1_actions)}", flush=True)

    # overcooked-ai: 같은 action 재생
    mdp = _load_custom_layout("cramped_room")
    oa_env = OvercookedEnv.from_mdp(mdp, horizon=NUM_STEPS)
    oa_env.reset()
    state = oa_env.state
    oa_total = 0.0
    for t, (a0, a1) in enumerate(v1_actions):
        joint = (jaxmarl_to_overcooked(a0), jaxmarl_to_overcooked(a1))
        next_state, r, done, _ = oa_env.step(joint)
        _auto_cook(next_state, mdp)
        state = next_state
        oa_total += float(r)
        if done:
            break
    print(f"overcooked-ai replay (V1 actions): reward={oa_total:.1f}, steps={t+1}", flush=True)

    # 비교
    print("\n=====================================")
    print(f"V1 native            reward = {v1_total:.1f}")
    print(f"overcooked-ai replay reward = {oa_total:.1f}")
    print("=====================================")


if __name__ == "__main__":
    main()
