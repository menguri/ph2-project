import os
import pickle
from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import chex
from flax.core.frozen_dict import FrozenDict

from overcooked_v2_experiments.ppo.policy import PPOPolicy, PPOParams


def _stored_filenames(filename_base):
    model_filename = os.path.join(filename_base, "model.pkl")
    config_filename = os.path.join(filename_base, "config.pkl")

    return model_filename, config_filename


def store_model(network_params, config, filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)

    with open(model_filename, "wb") as f:
        pickle.dump(network_params, f)
    with open(config_filename, "wb") as f:
        pickle.dump(config, f)


def load_model(filename_base):
    model_filename, config_filename = _stored_filenames(filename_base)

    with open(model_filename, "rb") as f:
        network_params = pickle.load(f)
    with open(config_filename, "rb") as f:
        config = pickle.load(f)

    return network_params, config


def _get_checkpoint_dir(run_base_dir, run_num, checkpoint, final=False, eval_dir=False):
    ckpt_name = "ckpt_final" if final else f"ckpt_{checkpoint}"
    if eval_dir:
        checkpoint_dir = run_base_dir / "eval" / f"run_{run_num}" / ckpt_name
    else:
        checkpoint_dir = run_base_dir / f"run_{run_num}" / ckpt_name

    return checkpoint_dir.resolve()


def store_checkpoint(config, params, run_num, checkpoint, final=False):
    eval_dir = bool(config.get("SAVE_TO_EVAL_DIR", False))
    checkpoint_dir = _get_checkpoint_dir(
        config["RUN_BASE_DIR"], run_num, checkpoint, final=final, eval_dir=eval_dir
    )

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    # Copy config and convert PosixPath to str to avoid orbax serialization error
    config_copy = dict(config)
    if "RUN_BASE_DIR" in config_copy and isinstance(config_copy["RUN_BASE_DIR"], Path):
        config_copy["RUN_BASE_DIR"] = str(config_copy["RUN_BASE_DIR"])

    checkpoint = {
        "config": config_copy,
        "params": params,
    }
    save_args = orbax_utils.save_args_from_target(checkpoint)
    print(
        f"[DEBUG] store_checkpoint: path={checkpoint_dir} final={final} name={'ckpt_final' if final else f'ckpt_{checkpoint}'}"
    )
    orbax_checkpointer.save(checkpoint_dir, checkpoint, save_args=save_args)


def load_checkpoint(run_dir, run_num, checkpoint, eval_dir=False):
    checkpoint_dir = _get_checkpoint_dir(run_dir, run_num, checkpoint, eval_dir=eval_dir)
    print(f"[DEBUG] Loading checkpoint from: {checkpoint_dir}")

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    # item=None + GPU에서 sharding 에러 방지:
    # restore_args 없이 item=None으로 복원하면 GPU sharding metadata 충돌 가능
    # 해결: 먼저 item=None 시도, 실패 시 abstract target으로 재시도
    import jax
    from jax.sharding import SingleDeviceSharding

    def _restore_to_single_device(checkpointer, directory):
        """multi-GPU sharding 문제 우회: 현재 default device의 SingleDeviceSharding으로 복원."""
        metadata = checkpointer.metadata(directory)
        target_device = jax.devices()[0]  # 현재 사용 가능한 첫 번째 device
        target_sharding = SingleDeviceSharding(target_device)

        def _make_restore_args(meta):
            if hasattr(meta, 'shape'):
                return ocp.ArrayRestoreArgs(sharding=target_sharding)
            return ocp.RestoreArgs()

        restore_args = jax.tree_util.tree_map(_make_restore_args, metadata)
        return checkpointer.restore(directory, restore_args=restore_args)

    try:
        ckpt = orbax_checkpointer.restore(checkpoint_dir, item=None)
    except (ValueError, TypeError, AttributeError):
        ckpt = _restore_to_single_device(orbax_checkpointer, checkpoint_dir)

    # orbax 복원 시 config의 숫자 값이 JAX ArrayImpl로 변환될 수 있음
    # → Python native로 변환하여 flax init 시 tracer 충돌 방지
    config = _convert_config_to_native(ckpt["config"])
    return config, ckpt["params"]


def _convert_config_to_native(config):
    """config dict의 JAX ArrayImpl 값을 Python native로 변환."""
    if isinstance(config, dict):
        return {k: _convert_config_to_native(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return type(config)(_convert_config_to_native(v) for v in config)
    if hasattr(config, 'item'):
        return config.item()
    return config


def load_all_checkpoints(run_dir, final_only=True, skip_initial=False, ckpt_filter=None):
    # ckpt_filter: 특정 ckpt 이름(예: "ckpt_5", "ckpt_final") 만 로드. 지정 시 final_only/skip_initial 무시.
    first_config = None
    all_checkpoints = {}
    configs = {}
    # SAVE_TO_EVAL_DIR=true 인 경우 체크포인트는 run_dir/eval/run_X/ 안에 저장됨.
    # 먼저 top-level 에 run_* 가 있는지 확인하고, 없으면 eval/ 하위를 순회한다.
    eval_subdir = run_dir / "eval"
    top_has_runs = any(
        p.is_dir() and "run_" in p.name for p in run_dir.iterdir()
    )
    eval_has_runs = eval_subdir.is_dir() and any(
        p.is_dir() and "run_" in p.name for p in eval_subdir.iterdir()
    )
    # ckpt_filter 가 주어지고 final 이 아니면 (= 중간 ckpt 요청) → eval/ 우선 (top 에는 보통 ckpt_final 만 있음).
    _need_intermediate = ckpt_filter is not None and "final" not in ckpt_filter
    if (_need_intermediate or not top_has_runs) and eval_has_runs:
        scan_root = eval_subdir
        use_eval_dir = True
        print(f"[DEBUG] Scanning eval/ subdir: {eval_subdir}")
    else:
        scan_root = run_dir
        use_eval_dir = False
    for run_num_dir in scan_root.iterdir():
        print(f"[DEBUG] Examining directory: {run_num_dir.name}")
        if not run_num_dir.is_dir() or "run_" not in run_num_dir.name:
            print(f"[DEBUG] Skipping (not a run dir): {run_num_dir.name}")
            continue
        run_num = int(run_num_dir.name.split("_")[1])
        print(f"[DEBUG] Processing run_num={run_num}")
        checkpoints = {}
        for checkpoint_dir in run_num_dir.iterdir():
            print(f"[DEBUG]   Found checkpoint dir: {checkpoint_dir.name}")
            if not checkpoint_dir.is_dir() or "ckpt_" not in checkpoint_dir.name:
                print(f"[DEBUG]   Skipping (not a ckpt dir): {checkpoint_dir.name}")
                continue
            if ckpt_filter is not None:
                if checkpoint_dir.name != ckpt_filter:
                    continue
            else:
                if final_only and "final" not in checkpoint_dir.name:
                    print(f"[DEBUG]   Skipping (final_only and not final): {checkpoint_dir.name}")
                    continue
                if skip_initial and "ckpt_0" in checkpoint_dir.name:
                    print(f"[DEBUG]   Skipping (skip_initial): {checkpoint_dir.name}")
                    continue
            ckpt_id = checkpoint_dir.name.split("_")[1]
            print(f"[DEBUG]   Loading ckpt_id={ckpt_id} for run_num={run_num}")
            config, params = load_checkpoint(run_dir, run_num, ckpt_id, eval_dir=use_eval_dir)
            policy = PPOParams(params=params)
            checkpoints[checkpoint_dir.name] = policy
            if not first_config:
                first_config = config
        all_checkpoints[run_num_dir.name] = checkpoints
        configs[run_num_dir.name] = config
        print(f"[DEBUG] Loaded {len(checkpoints)} checkpoints for {run_num_dir.name}")
    return all_checkpoints, first_config, configs
