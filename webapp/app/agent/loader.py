"""
Phase 2: Checkpoint 로드 — GPU→CPU 복원, baseline/ph2 자동 감지.
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import jax
import orbax.checkpoint as ocp
from jax.sharding import SingleDeviceSharding


CPU_DEVICE = jax.devices("cpu")[0]
CPU_SHARDING = SingleDeviceSharding(CPU_DEVICE)


def _build_cpu_restore_args(tree):
    """checkpoint metadata 트리를 순회하며 CPU sharding restore args 생성."""
    if isinstance(tree, dict):
        return {k: _build_cpu_restore_args(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_build_cpu_restore_args(v) for v in tree)
    type_name = type(tree).__name__
    if "ArrayMetadata" in type_name:
        return ocp.ArrayRestoreArgs(sharding=CPU_SHARDING)
    return ocp.RestoreArgs()


def _load_checkpoint_sync(ckpt_path: Path) -> Dict[str, Any]:
    """동기 checkpoint 로드 (별도 스레드에서 호출).
    PH2 체크포인트는 ckpt_final/model_ckpt/ 하위에 orbax 데이터가 있으므로 자동 감지."""
    # PH2 구조: ckpt_final/model_ckpt/_METADATA 가 있으면 model_ckpt/ 를 로드
    model_ckpt_sub = ckpt_path / "model_ckpt"
    if model_ckpt_sub.is_dir() and (model_ckpt_sub / "_METADATA").exists():
        actual_path = model_ckpt_sub
    else:
        actual_path = ckpt_path

    handler = ocp.PyTreeCheckpointHandler()
    meta = handler.metadata(actual_path)
    restore_args = _build_cpu_restore_args(meta.tree)
    checkpointer = ocp.PyTreeCheckpointer()
    result = checkpointer.restore(str(actual_path), restore_args=restore_args)

    # PH2 구조: metadata.json에서 config가 없으면 mode_tildes.npz에서 로드
    if "config" not in result:
        import json, numpy as np
        meta_json = ckpt_path / "metadata.json"
        tildes = ckpt_path / "mode_tildes.npz"
        if meta_json.exists():
            with open(meta_json) as f:
                result["_run_metadata"] = json.load(f)
    return result


def load_checkpoint_cpu(ckpt_dir: str) -> Dict[str, Any]:
    """GPU에서 저장된 Orbax checkpoint를 CPU에서 안전하게 로드.
    orbax 내부의 nest_asyncio가 uvloop과 충돌하므로 별도 스레드에서 실행."""
    import concurrent.futures
    ckpt_path = Path(ckpt_dir).resolve()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_checkpoint_sync, ckpt_path)
        return future.result()


def detect_model_source(config: dict) -> str:
    """checkpoint config에서 baseline vs ph2 판별.

    Returns:
        "ph2": PH2 계열 (PH2, PH2-E3T 등)
        "baseline_native": baseline 코드 직접 사용 (MEP, E3T 등 인코더 구조가 ph2와 다른 모델)
        "baseline": baseline 중 param 리매핑으로 ph2 코드에서 로드 가능한 모델 (SP, FCP)
    """
    alg = str(config.get("ALG_NAME", ""))
    if "PH2" in alg.upper() or config.get("TRANSFORMER_ACTION", False):
        return "ph2"
    # MEP/HSP/Gamma: CNNGamma 인코더 사용 → ph2의 shared_encoder(CNN)와 구조 불일치
    # baseline ActorCriticRNN을 직접 로드하여 actor_only=True로 추론
    if alg.upper() in ("MEP", "HSP", "GAMMA"):
        return "baseline_native"
    # E3T 등 CNNSimple 인코더 사용 모델: ph2 리매핑 불가 (Conv 구조가 다름)
    # config["model"]["OBS_ENCODER"]에 인코더 타입이 저장됨
    obs_encoder = config.get("model", {}).get("OBS_ENCODER", "CNN")
    if str(obs_encoder).upper() != "CNN":
        return "baseline_native"
    return "baseline"


def select_params(ckpt: dict, policy_source: str = "params"):
    """checkpoint에서 원하는 policy params 선택."""
    source = policy_source.strip().lower()
    if source == "ind":
        return ckpt.get("params_ind", ckpt.get("params", ckpt.get("params_spec")))
    if source == "spec":
        return ckpt.get("params_spec", ckpt.get("params", ckpt.get("params_ind")))
    return ckpt.get("params", ckpt.get("params_ind", ckpt.get("params_spec")))


def scan_models_dir(models_dir: str) -> dict:
    """models/ 디렉토리 스캔 → layout별 모델 목록.

    구조: models/{layout}/{algo_name}/{run0,run1,...}/ckpt_final/
    반환: {layout: [{algo_name, seed_id, ckpt_path}, ...]}
    """
    results = {}
    base = Path(models_dir)
    if not base.exists():
        return results
    for layout_dir in sorted(base.iterdir()):
        if not layout_dir.is_dir():
            continue
        layout_name = layout_dir.name
        models_for_layout = []
        for algo_dir in sorted(layout_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for run_dir in sorted(algo_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                ckpt_final = run_dir / "ckpt_final"
                if ckpt_final.exists() and ckpt_final.is_dir():
                    models_for_layout.append({
                        "algo_name": algo_dir.name,
                        "seed_id": run_dir.name,
                        "ckpt_path": str(ckpt_final),
                    })
                    continue
                if (run_dir / "_CHECKPOINT_METADATA").exists():
                    models_for_layout.append({
                        "algo_name": algo_dir.name,
                        "seed_id": run_dir.name,
                        "ckpt_path": str(run_dir),
                    })
        if models_for_layout:
            results[layout_name] = models_for_layout
    return results


def scan_runs_dir(runs_dir: str) -> list:
    """runs/ 디렉토리 스캔 → 사용 가능한 체크포인트 목록."""
    results = []
    base = Path(runs_dir)
    if not base.exists():
        return results
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        for run_num_dir in sorted(run_dir.iterdir()):
            if not run_num_dir.is_dir() or "run_" not in run_num_dir.name:
                continue
            ckpt_final = run_num_dir / "ckpt_final"
            if ckpt_final.exists():
                results.append({
                    "algo_name": run_dir.name,
                    "seed_id": run_num_dir.name,
                    "ckpt_path": str(ckpt_final),
                })
    return results
