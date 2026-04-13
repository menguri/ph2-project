"""obs_adapter_v2 검증 스크립트.

1. overcooked_v2 env reset → OV2 obs 생성
2. ov2_obs_to_cec() → (9,9,26) 채널별 sanity check
3. obs_adapter (v1) 출력과 obs_adapter_v2 출력 비교
4. CEC 모델에 v2 adapter 출력 넣어서 추론 검증
"""
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl

from cec_integration.cec_runtime import CECRuntime
from cec_integration.obs_adapter import CECObsAdapter
from cec_integration.obs_adapter_v2 import ov2_obs_to_cec, LAYOUT_PADDING

CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ckpts", "forced_coord_9"
)
LAYOUT = "forced_coord"
NUM_STEPS = 400


def check_channel_sanity(cec_obs: np.ndarray, label: str, layout: str):
    """CEC (9,9,26) obs 채널별 sanity check."""
    h, w = LAYOUT_PADDING[layout][:2]
    y_off, x_off = LAYOUT_PADDING[layout][2:]
    errors = []

    # ch0 (self pos): 게임 영역에서 정확히 1개 셀
    self_pos_sum = cec_obs[y_off:y_off+h, x_off:x_off+w, 0].sum()
    if not np.isclose(self_pos_sum, 1.0):
        errors.append(f"ch0 self_pos sum={self_pos_sum}, expected 1.0")

    # ch1 (other pos): 게임 영역에서 정확히 1개 셀
    other_pos_sum = cec_obs[y_off:y_off+h, x_off:x_off+w, 1].sum()
    if not np.isclose(other_pos_sum, 1.0):
        errors.append(f"ch1 other_pos sum={other_pos_sum}, expected 1.0")

    # ch2-5 (self dir): self_pos 위치에서 one-hot
    self_pos_coords = np.argwhere(cec_obs[:, :, 0] > 0.5)
    if len(self_pos_coords) == 1:
        sy, sx = self_pos_coords[0]
        dir_vec = cec_obs[sy, sx, 2:6]
        if not np.isclose(dir_vec.sum(), 1.0):
            errors.append(f"ch2-5 self_dir at ({sy},{sx}) sum={dir_vec.sum()}, expected 1.0")
    else:
        errors.append(f"ch0 self_pos has {len(self_pos_coords)} cells, expected 1")

    # ch6-9 (other dir): other_pos 위치에서 one-hot
    other_pos_coords = np.argwhere(cec_obs[:, :, 1] > 0.5)
    if len(other_pos_coords) == 1:
        oy, ox = other_pos_coords[0]
        dir_vec = cec_obs[oy, ox, 6:10]
        if not np.isclose(dir_vec.sum(), 1.0):
            errors.append(f"ch6-9 other_dir at ({oy},{ox}) sum={dir_vec.sum()}, expected 1.0")

    # ch10 (pot): forced_coord에는 2개 pot
    pot_sum = cec_obs[y_off:y_off+h, x_off:x_off+w, 10].sum()
    if pot_sum < 1.0:
        errors.append(f"ch10 pot sum={pot_sum}, expected >= 1 (forced_coord has 2 pots)")

    # ch11 (wall): 패딩 영역은 반드시 wall
    padding_wall_count = 0
    total_padding = 0
    for y in range(9):
        for x in range(9):
            in_game = (y_off <= y < y_off + h) and (x_off <= x < x_off + w)
            if not in_game:
                total_padding += 1
                if cec_obs[y, x, 11] > 0.5:
                    padding_wall_count += 1
    if padding_wall_count != total_padding:
        errors.append(f"ch11 padding wall: {padding_wall_count}/{total_padding} cells are wall")

    # ch25 (urgency): step 0이면 0, step >= 360이면 1
    # (이건 호출자가 step을 알려줘야 하므로 여기서는 값 범위만 체크)
    urgency_vals = np.unique(cec_obs[:, :, 25])
    for v in urgency_vals:
        if v not in (0.0, 1.0):
            errors.append(f"ch25 urgency has non-binary value: {v}")

    # NaN/Inf 체크
    if not np.all(np.isfinite(cec_obs)):
        errors.append("NaN/Inf detected in obs")

    if errors:
        print(f"  [{label}] FAIL:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"  [{label}] PASS (self@{self_pos_coords[0] if len(self_pos_coords)==1 else '?'}, "
              f"other@{other_pos_coords[0] if len(other_pos_coords)==1 else '?'}, "
              f"pots={pot_sum:.0f}, walls_pad={padding_wall_count}/{total_padding})")
        return True


def compare_v1_v2(v1_obs: np.ndarray, v2_obs: np.ndarray, agent: str):
    """v1 adapter와 v2 adapter 출력 비교."""
    # v1 obs는 uint8일 수 있음
    v1 = np.asarray(v1_obs, dtype=np.float32)
    v2 = np.asarray(v2_obs, dtype=np.float32)

    print(f"\n  [{agent}] v1 vs v2 비교:")
    ch_names = [
        "self_pos", "other_pos",
        "self_dir_N", "self_dir_S", "self_dir_E", "self_dir_W",
        "other_dir_N", "other_dir_S", "other_dir_E", "other_dir_W",
        "pot", "wall", "onion_pile", "tomato_pile", "plate_pile", "goal",
        "onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
        "cook_time", "soup_ready", "plate_on_grid", "onion_on_grid",
        "tomato_on_grid", "urgency",
    ]
    diffs = []
    for ch in range(26):
        diff = np.abs(v1[:, :, ch] - v2[:, :, ch])
        max_diff = diff.max()
        if max_diff > 0.01:
            diff_cells = np.argwhere(diff > 0.01)
            diffs.append((ch, ch_names[ch], max_diff, len(diff_cells)))

    if diffs:
        print(f"    차이 발견 ({len(diffs)} channels):")
        for ch, name, max_d, n_cells in diffs:
            print(f"      ch{ch:2d} ({name:16s}): max_diff={max_d:.3f}, {n_cells} cells differ")
            # 디테일: 처음 3개 차이 셀
            diff_map = np.abs(v1[:, :, ch] - v2[:, :, ch])
            cells = np.argwhere(diff_map > 0.01)[:3]
            for cy, cx in cells:
                print(f"        ({cy},{cx}): v1={v1[cy,cx,ch]:.3f}  v2={v2[cy,cx,ch]:.3f}")
        return False
    else:
        print(f"    완벽 일치 (26 channels)")
        return True


def main() -> int:
    print("=" * 60)
    print("obs_adapter_v2 검증")
    print("=" * 60)

    # --- Setup ---
    env = jaxmarl.make(
        "overcooked_v2",
        layout=LAYOUT,
        max_steps=NUM_STEPS,
        random_reset=False,       # 결정적 초기화
        random_agent_positions=False,
    )
    print(f"env: {LAYOUT} h={env.height} w={env.width} agents={env.num_agents}")

    adapter_v1 = CECObsAdapter(target_layout="forced_coord_9", max_steps=NUM_STEPS)

    rng = jax.random.PRNGKey(42)
    obs_v2_dict, env_state = env.reset(rng)

    # --- 1. v2 adapter 채널 sanity ---
    print("\n[1] obs_adapter_v2 채널 sanity check (reset)")
    all_pass = True
    for agent_key in ["agent_0", "agent_1"]:
        ov2_obs = obs_v2_dict[agent_key]  # (H, W, 30)
        cec_obs = ov2_obs_to_cec(jnp.array(ov2_obs, dtype=jnp.float32), LAYOUT, 0, NUM_STEPS)
        cec_np = np.asarray(cec_obs)
        ok = check_channel_sanity(cec_np, f"v2-{agent_key}", LAYOUT)
        all_pass = all_pass and ok

    # --- 2. v1 vs v2 비교 ---
    print("\n[2] v1 adapter vs v2 adapter 비교 (reset)")
    v1_obs_dict = adapter_v1.get_cec_obs(env_state)  # {'agent_0': (9,9,26), ...}
    match_all = True
    for agent_key in ["agent_0", "agent_1"]:
        v1_obs = np.asarray(v1_obs_dict[agent_key])
        ov2_obs = obs_v2_dict[agent_key]
        v2_obs = np.asarray(ov2_obs_to_cec(
            jnp.array(ov2_obs, dtype=jnp.float32), LAYOUT, 0, NUM_STEPS
        ))
        ok = compare_v1_v2(v1_obs, v2_obs, agent_key)
        match_all = match_all and ok

    # --- 3. 여러 스텝 진행 후 비교 ---
    print("\n[3] 10 step 진행 후 v1 vs v2 비교")
    rng = jax.random.PRNGKey(42)
    obs_v2_dict, env_state = env.reset(rng)
    for t in range(10):
        rng, k_env = jax.random.split(rng)
        # 랜덤 action
        actions = {"agent_0": int(jax.random.randint(k_env, (), 0, 6)),
                   "agent_1": int(jax.random.randint(k_env, (), 0, 6))}
        rng, k_env = jax.random.split(rng)
        obs_v2_dict, env_state, reward, done, _info = env.step(k_env, env_state, actions)

    # step 10 이후 비교
    v1_obs_dict = adapter_v1.get_cec_obs(env_state)
    step_match = True
    for agent_key in ["agent_0", "agent_1"]:
        v1_obs = np.asarray(v1_obs_dict[agent_key])
        ov2_obs = obs_v2_dict[agent_key]
        v2_obs = np.asarray(ov2_obs_to_cec(
            jnp.array(ov2_obs, dtype=jnp.float32), LAYOUT, 10, NUM_STEPS
        ))
        ok = check_channel_sanity(v2_obs, f"v2-step10-{agent_key}", LAYOUT)
        step_match = step_match and ok
        ok2 = compare_v1_v2(v1_obs, v2_obs, agent_key)
        step_match = step_match and ok2

    # --- 4. v2 adapter 출력으로 CEC 모델 추론 ---
    print("\n[4] v2 adapter → CEC 모델 추론 테스트")
    rt = CECRuntime(os.path.join(CKPT_DIR, "seed11_ckpt0_improved"))
    hidden = rt.init_hidden(2)
    done_arr = jnp.zeros((2,), dtype=bool)
    rng = jax.random.PRNGKey(0)

    obs_v2_dict_fresh, _ = env.reset(jax.random.PRNGKey(99))
    a0_cec = ov2_obs_to_cec(jnp.array(obs_v2_dict_fresh["agent_0"], dtype=jnp.float32), LAYOUT, 0, NUM_STEPS)
    a1_cec = ov2_obs_to_cec(jnp.array(obs_v2_dict_fresh["agent_1"], dtype=jnp.float32), LAYOUT, 0, NUM_STEPS)
    obs_batch = jnp.stack([a0_cec, a1_cec])  # (2, 9, 9, 26)

    rng, sub = jax.random.split(rng)
    actions, new_hidden, probs = rt.step(obs_batch, hidden, done_arr, sub)
    actions_np = np.asarray(actions)
    probs_np = np.asarray(probs)

    infer_ok = True
    if not np.all(np.isfinite(probs_np)):
        print("  [FAIL] NaN/Inf in probs")
        infer_ok = False
    elif not np.allclose(probs_np.sum(-1), 1.0, atol=1e-4):
        print(f"  [FAIL] probs sum != 1: {probs_np.sum(-1)}")
        infer_ok = False
    elif not np.all((actions_np >= 0) & (actions_np < 6)):
        print(f"  [FAIL] invalid actions: {actions_np}")
        infer_ok = False
    else:
        print(f"  [PASS] actions={actions_np.tolist()}, "
              f"probs[0]=[{', '.join(f'{p:.3f}' for p in probs_np[0])}]")

    # --- 요약 ---
    print("\n" + "=" * 60)
    print("요약:")
    print(f"  채널 sanity (reset):      {'PASS' if all_pass else 'FAIL'}")
    print(f"  v1 vs v2 일치 (reset):    {'PASS' if match_all else 'DIFF'}")
    print(f"  채널 sanity (step 10):    {'PASS' if step_match else 'FAIL'}")
    print(f"  v2→CEC 추론:             {'PASS' if infer_ok else 'FAIL'}")
    print("=" * 60)

    return 0 if (all_pass and infer_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
