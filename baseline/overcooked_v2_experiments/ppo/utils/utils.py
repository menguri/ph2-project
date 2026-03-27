from datetime import datetime
from pathlib import Path
import os
import jax
import jax.numpy as jnp


def _infer_run_suffix(config) -> str:
    """Return the suffix (sp/sa/uc/...) used for folder + default run names."""
    exp = config.get("EXP", "")
    if exp and "-" in exp:
        suffix = exp.split("-")[-1]
    else:
        model_name = config["model"]["TYPE"].upper()
        if "NUM_ITERATIONS" in config:
            # state augmentation(k-step state buffer) 실험 구분용
            suffix = "sa"
        elif model_name == "RNN":
            suffix = "sp"
        elif model_name == "CNN":
            suffix = "sp"
        else:
            suffix = model_name.lower()

    # E3T 알고리즘인 경우 suffix에 e3t 추가
    if config.get("ALG_NAME") == "E3T":
        suffix = f"e3t"
    if config.get("ALG_NAME") == "E3D":
        suffix = f"e3d"
    if config.get("ALG_NAME") == "MEP_S1":
        suffix = "m1"
    if config.get("ALG_NAME") == "MEP_S2":
        suffix = "m2"
    if config.get("ALG_NAME") == "GAMMA_S1":
        suffix = "g1"
    if config.get("ALG_NAME") == "GAMMA_S2":
        suffix = "g2"
    if config.get("ALG_NAME") == "HSP_S1":
        suffix = "h1"
    if config.get("ALG_NAME") == "HSP_S2":
        suffix = "h2"
    # 통합 파이프라인 suffix
    if config.get("ALG_NAME") == "MEP":
        suffix = "mep"
    if config.get("ALG_NAME") == "GAMMA":
        method = config.get("GAMMA_S2_METHOD", "rl")
        suffix = f"gamma-{method}" if method != "rl" else "gamma"
    if config.get("ALG_NAME") == "HSP":
        suffix = "hsp"

    return suffix


def build_default_run_name(config) -> str:
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    layout_name = config["env"]["ENV_KWARGS"].get("layout", config["env"].get("ENV_NAME", "unknown"))
    model_name = config["model"]["TYPE"]

    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    suffix = _infer_run_suffix(config)

    return f"{suffix}_{layout_name}_{model_name.lower()}_{avs_str}"


def _build_param_tag(config) -> str:
    """알고리즘별 핵심 파라미터를 짧은 태그로 생성. 디렉토리 이름에 포함."""
    alg = config.get("ALG_NAME", "SP")
    model = config.get("model", {})
    parts = []

    gru = model.get("GRU_HIDDEN_DIM", 128)
    if gru != 128:
        parts.append(f"h{gru}")

    if alg in ("MEP", "MEP_S1", "GAMMA", "GAMMA_S1"):
        pop = config.get("MEP_POPULATION_SIZE", 4)
        parts.append(f"pop{pop}")
        alpha = config.get("MEP_ENTROPY_ALPHA", 0.01)
        if alpha != 0.01:
            parts.append(f"ea{alpha}")

    if alg in ("HSP", "HSP_S1"):
        n = config.get("HSP_POPULATION_SIZE", 36)
        k = config.get("HSP_SELECTED_K", 18)
        parts.append(f"n{n}k{k}")

    if alg == "GAMMA" and config.get("GAMMA_S2_METHOD") == "vae":
        z = config.get("GAMMA_VAE_Z_DIM", 16)
        parts.append(f"z{z}")

    nenv = model.get("NUM_ENVS", 64)
    if nenv not in (64, 256):  # 비표준 값만 표시
        parts.append(f"e{nenv}")

    return "_".join(parts)


def get_run_base_dir(run_id: str, config) -> str:
    optional_prefix = config.get("OPTIONAL_PREFIX", "")

    layout_name = config["env"]["ENV_KWARGS"].get("layout", config["env"].get("ENV_NAME", "unknown"))

    results_dir = "runs"
    if optional_prefix != "":
        results_dir = os.path.join(results_dir, optional_prefix)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    suffix = _infer_run_suffix(config)
    param_tag = _build_param_tag(config)

    if "FCP" in config:
        dir = Path(config["FCP"])
        f = dir.name
        run_dir = os.path.join(results_dir, f"{timestamp}_{run_id}_{layout_name}_fcp")
    else:
        name_parts = [timestamp, run_id, layout_name, suffix]
        if param_tag:
            name_parts.append(param_tag)
        run_dir = os.path.join(results_dir, "_".join(name_parts))
    os.makedirs(run_dir, exist_ok=True)

    print("run_dir", run_dir)

    return Path(run_dir)


def combine_first_two_tree_dim(tree):
    return jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), tree)


def get_num_devices():
    num_devices = 1

    try:
        devices = jax.devices("gpu")
        if devices:
            num_devices = len(devices)
            print(f"GPU is available! Using {num_devices} GPUs.")
        else:
            print("No GPU found, falling back to CPU.")
    except RuntimeError as e:
        print("Warning: Falling back to CPU.")

    return num_devices
