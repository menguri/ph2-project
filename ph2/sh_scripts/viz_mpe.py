"""MPE 환��� rollout 시각화 스크립트.
사용법: python viz_mpe.py <run_dir> [--run_num 0] [--ckpt final] [--seed 0] [--steps 25]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import jaxmarl

from overcooked_v2_experiments.ppo.utils.store import load_checkpoint, _get_checkpoint_dir
from overcooked_v2_experiments.ppo.policy import PPOPolicy
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer


def _load(run_dir, run_num, ckpt_name, policy_source):
    """ckpt_name: 'final' 또는 숫자 '0', '1' 등."""
    if ckpt_name == "final":
        ckpt_dir = run_dir / f"run_{run_num}" / "ckpt_final"
    else:
        ckpt_dir = run_dir / f"run_{run_num}" / f"ckpt_{ckpt_name}"

    print(f"[VIZ] 체크포인트 로드: {ckpt_dir}")

    import orbax.checkpoint as ocp
    from jax.sharding import SingleDeviceSharding

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        ckpt = checkpointer.restore(str(ckpt_dir.resolve()), item=None)
    except (ValueError, TypeError, AttributeError):
        metadata = checkpointer.metadata(str(ckpt_dir.resolve()))
        target_sharding = SingleDeviceSharding(jax.devices()[0])
        def _make_args(meta):
            if hasattr(meta, 'shape'):
                return ocp.ArrayRestoreArgs(sharding=target_sharding)
            return ocp.RestoreArgs()
        restore_args = jax.tree_util.tree_map(_make_args, metadata)
        ckpt = checkpointer.restore(str(ckpt_dir.resolve()), restore_args=restore_args)

    # config native 변환
    def _to_native(v):
        if isinstance(v, dict):
            return {k: _to_native(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_to_native(val) for val in v)
        if hasattr(v, 'item'):
            return v.item()
        return v

    config = _to_native(ckpt["config"])

    # params 추출
    if policy_source == "params_ind" and "params_ind" in ckpt:
        params = ckpt["params_ind"]
    else:
        params = ckpt.get("params", ckpt.get("train_state", {}).get("params"))

    return config, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="run 디렉토리 경로")
    parser.add_argument("--run_num", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="final",
                        help="'final' 또는 체크포인트 번호 (0, 1, ...)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--policy_source", type=str, default="params",
                        help="params | params_ind (PH2 ind 정책)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).parent.parent / "runs" / run_dir.name

    print(f"[VIZ] run_dir={run_dir}")

    config, params = _load(run_dir, args.run_num, args.ckpt, args.policy_source)

    # 환경 생성
    env_cfg = config.get("env", {})
    env_name = env_cfg.get("ENV_NAME", "MPE_simple_spread_v3")
    env_kwargs = {k: v for k, v in env_cfg.get("ENV_KWARGS", {}).items() if k != "layout"}
    print(f"[VIZ] env={env_name}, kwargs={env_kwargs}")

    env = jaxmarl.make(env_name, **env_kwargs)

    # ind 정책이면 blocked input 비활성화
    if args.policy_source == "params_ind":
        config["LEARNER_USE_BLOCKED_INPUT"] = False

    # ACTION_DIM이 config에 없으면 환경에서 추론
    model_cfg = config.get("model", config)
    if model_cfg.get("ACTION_DIM") is None:
        act_space = env.action_spaces[env.agents[0]]
        action_dim = act_space.n if hasattr(act_space, 'n') else int(act_space.shape[0])
        model_cfg["ACTION_DIM"] = action_dim
        print(f"[VIZ] ACTION_DIM 추론: {action_dim}")

    # 정책 생성 (ACTION_DIM 설정 후)
    policy = PPOPolicy(params, config)

    # Rollout
    key = jax.random.PRNGKey(args.seed)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)
    done = {f"agent_{i}": False for i in range(env.num_agents)}
    done["__all__"] = False
    hstate = {f"agent_{i}": policy.init_hstate(1) for i in range(env.num_agents)}

    state_seq = [state]
    max_steps = min(args.steps, int(getattr(env, "max_steps", 25)))

    for t in range(max_steps):
        key, k_act, k_step = jax.random.split(key, 3)
        act_keys = jax.random.split(k_act, env.num_agents)

        actions = {}
        next_hstates = {}
        for i in range(env.num_agents):
            agent_id = f"agent_{i}"
            action, new_h, _ = policy.compute_action(
                obs[agent_id], done[agent_id], hstate[agent_id], act_keys[i],
            )
            actions[agent_id] = action
            next_hstates[agent_id] = new_h

        obs, state, reward, done, info = env.step(k_step, state, actions)
        hstate = next_hstates
        state_seq.append(state)

    # 시각화 저장
    out_fname = run_dir / f"viz_run{args.run_num}_ckpt{args.ckpt}_seed{args.seed}.gif"
    print(f"[VIZ] {len(state_seq)} frames -> {out_fname}")

    viz = MPEVisualizer(env, state_seq)
    viz.animate(save_fname=str(out_fname), view=False)
    print(f"[VIZ] 저장 완료: {out_fname}")


if __name__ == "__main__":
    main()
