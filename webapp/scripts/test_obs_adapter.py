#!/usr/bin/env python3
"""
obs_adapter 검증: overcooked-ai state → JaxMARL obs 변환이 올바른지 확인.

JaxMARL 환경을 직접 실행하여 obs를 생성하고,
동일한 state에서 obs_adapter가 만든 obs와 비교.
"""
import os, sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from app.game.obs_adapter import overcooked_state_to_jaxmarl_obs, get_obs_shape


def test_basic_shape():
    """obs shape이 JaxMARL 스펙과 일치하는지 확인."""
    for layout in ["cramped_room"]:
        shape = get_obs_shape(layout, num_ingredients=1)
        expected_c = 18 + 4 * (1 + 2)  # 30
        print(f"  {layout}: shape={shape}, expected C={expected_c}")
        assert shape[2] == expected_c, f"Channel mismatch: {shape[2]} != {expected_c}"
    print("  [OK] Shape test passed")


def test_initial_state_obs():
    """초기 state에서 obs를 생성하고 기본 검증."""
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    state = mdp.get_standard_start_state()

    obs0 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=0)
    obs1 = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=1)

    print(f"  obs0 shape: {obs0.shape}, dtype: {obs0.dtype}")
    print(f"  obs1 shape: {obs1.shape}, dtype: {obs1.dtype}")

    # 기본 검증
    assert obs0.shape == (4, 5, 30), f"Wrong shape: {obs0.shape}"
    assert obs0.dtype == np.uint8

    # self/other agent 채널이 서로 바뀌었는지 확인
    # agent 0 시점: ch[0]이 agent0 위치, ch[8]이 agent1 위치
    # agent 1 시점: ch[0]이 agent1 위치, ch[8]이 agent0 위치
    p0_pos = state.players[0].position
    p1_pos = state.players[1].position

    # obs0: self=agent0 → ch[0]에 agent0 위치 = 1
    assert obs0[p0_pos[1], p0_pos[0], 0] == 1, "obs0 self agent position wrong"
    # obs0: other=agent1 → ch[8]에 agent1 위치 = 1
    assert obs0[p1_pos[1], p1_pos[0], 8] == 1, "obs0 other agent position wrong"

    # obs1: self=agent1 → ch[0]에 agent1 위치 = 1
    assert obs1[p1_pos[1], p1_pos[0], 0] == 1, "obs1 self agent position wrong"
    # obs1: other=agent0 → ch[8]에 agent0 위치 = 1
    assert obs1[p0_pos[1], p0_pos[0], 8] == 1, "obs1 other agent position wrong"

    print("  [OK] Initial state obs test passed")


def test_static_terrain():
    """terrain 인코딩이 올바른지 확인 (wall, pot, goal, etc)."""
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    state = mdp.get_standard_start_state()
    obs = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=0)
    terrain = mdp.terrain_mtx

    # Static objects start at channel 16 (after 2 * agent_ch_size)
    # agent_ch_size = 5 + 2 + 1 = 8
    static_start = 16

    wall_count = 0
    pot_count = 0
    goal_count = 0
    for r in range(len(terrain)):
        for c in range(len(terrain[0])):
            ch = terrain[r][c]
            if ch == "X":
                assert obs[r, c, static_start] == 1, f"Wall at ({r},{c}) not encoded"
                wall_count += 1
            elif ch == "P":
                assert obs[r, c, static_start + 2] == 1, f"Pot at ({r},{c}) not encoded"
                pot_count += 1
            elif ch in ("S", "D"):
                assert obs[r, c, static_start + 1] == 1, f"Goal at ({r},{c}) not encoded"
                goal_count += 1

    print(f"  Terrain: {wall_count} walls, {pot_count} pots, {goal_count} goals")
    assert wall_count > 0 and pot_count > 0 and goal_count > 0
    print("  [OK] Static terrain test passed")


def test_model_inference_sanity():
    """obs_adapter 출력으로 모델 추론이 crash 없이 되는지 확인."""
    from app.agent.loader import load_checkpoint_cpu, detect_model_source, select_params

    models_dir = Path(__file__).resolve().parent.parent / "models"
    ckpt_dirs = list(models_dir.rglob("ckpt_final"))

    if not ckpt_dirs:
        print("  [SKIP] No model checkpoints found in models/")
        return

    ckpt_path = str(ckpt_dirs[0])
    print(f"  Loading checkpoint: {ckpt_path}")

    ckpt = load_checkpoint_cpu(ckpt_path)
    config = ckpt["config"]
    source = detect_model_source(config)
    print(f"  Source: {source}")

    from app.agent.inference import ModelManager
    mm = ModelManager()
    mm.load(ckpt_path=ckpt_path, algo_name="test", seed_id="test")

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    state = mdp.get_standard_start_state()
    obs = overcooked_state_to_jaxmarl_obs(state, mdp, agent_idx=0)

    action = mm.get_action(obs)
    print(f"  Model action: {action} (should be 0-5)")
    assert 0 <= action <= 5, f"Invalid action: {action}"
    print("  [OK] Model inference sanity test passed")


if __name__ == "__main__":
    print("=" * 50)
    print("obs_adapter Verification")
    print("=" * 50)

    print("\n[1] Shape test")
    test_basic_shape()

    print("\n[2] Initial state obs test")
    test_initial_state_obs()

    print("\n[3] Static terrain test")
    test_static_terrain()

    print("\n[4] Model inference sanity test")
    test_model_inference_sanity()

    print("\n" + "=" * 50)
    print("All obs_adapter tests PASSED")
