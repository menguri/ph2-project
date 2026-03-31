from datetime import datetime
from pathlib import Path
import os
import jax
import jax.numpy as jnp


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def _float_to_slug(value, fallback="na"):
    try:
        x = float(value)
    except Exception:
        return fallback

    if abs(x - round(x)) < 1e-9:
        text = str(int(round(x)))
    else:
        text = f"{x:.6f}".rstrip("0").rstrip(".")

    return text.replace("-", "m").replace(".", "p")


def _ph_param_suffix(config) -> str:
    """Add PH1/PH2 run-name params for easier run tracking.

    Included fields:
    - PH1_EPSILON
    - PH1_OMEGA
    - PH1_SIGMA
    - PH1_MAX_PENALTY_COUNT (k)
    - TRANSFORMER_ACTION (ct0 / ct1)
    """
    alg_name = str(config.get("ALG_NAME", "")).upper()
    if ("PH1" not in alg_name) and ("PH2" not in alg_name):
        return ""

    eps = _float_to_slug(config.get("PH1_EPSILON", None))
    omega = _float_to_slug(config.get("PH1_OMEGA", None))
    sigma = _float_to_slug(config.get("PH1_SIGMA", None))

    # penalty 슬롯 수 (PH1_MAX_PENALTY_COUNT)
    max_k = config.get("PH1_MAX_PENALTY_COUNT", None)
    k_str = f"_k{int(max_k)}" if max_k is not None else ""

    # CycleTransformer 활성화 여부
    ct_on = _as_bool(config.get("TRANSFORMER_ACTION", False))
    ct_str = "_ct1" if ct_on else "_ct0"

    # Pair mode
    pair_on = _as_bool(config.get("PH1_PAIR_MODE", False))
    pair_str = "_pair" if pair_on else ""

    return f"_e{eps}_o{omega}_s{sigma}{k_str}{ct_str}{pair_str}"


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
        # STL (Anchor) 활성화 시 suffix에 stl 추가
        if config.get("model", {}).get("anchor", False):
            suffix = f"stl"
    if config.get("ALG_NAME") == "E3D":
        suffix = f"e3d"

    # PH1 표시: e3t/e3d 뒤에 ph1 추가
    alg_name = str(config.get("ALG_NAME", "")).upper()
    if "PH1" in alg_name:
        if "E3T" in alg_name:
            suffix = "e3t_ph1"
        elif "E3D" in alg_name:
            suffix = "e3d_ph1"
        else:
            suffix = f"{suffix}_ph1"
    elif "PH2" in alg_name:
        if "E3T" in alg_name:
            suffix = "e3t_ph2"
        elif "E3D" in alg_name:
            suffix = "e3d_ph2"
        else:
            suffix = f"{suffix}_ph2"
        # CycleTransformer 변형: e3t_ph2_ct / e3t_ph2_ct3
        if "CT" in alg_name:
            if _as_bool(config.get("TRANSFORMER_V3", False)):
                suffix = f"{suffix}_ct3"
            else:
                suffix = f"{suffix}_ct"

    return suffix


def build_default_run_name(config) -> str:
    agent_view_size = config["env"]["ENV_KWARGS"].get("agent_view_size", None)
    layout_name = config["env"]["ENV_KWARGS"].get("layout", config["env"].get("ENV_NAME", "unknown"))
    model_name = config["model"]["TYPE"]

    avs_str = f"avs-{agent_view_size}" if agent_view_size is not None else "avs-full"
    suffix = _infer_run_suffix(config) + _ph_param_suffix(config)

    return f"{suffix}_{layout_name}_{model_name.lower()}_{avs_str}"


def get_run_base_dir(run_id: str, config) -> str:
    optional_prefix = config.get("OPTIONAL_PREFIX", "")

    layout_name = config["env"]["ENV_KWARGS"].get("layout", config["env"].get("ENV_NAME", "unknown"))
    # ToyCoop + random_reset → layout_name에 _Ran suffix 추가
    if layout_name == "ToyCoop" and config["env"].get("ENV_KWARGS", {}).get("random_reset", False):
        layout_name = "ToyCoop_Ran"

    results_dir = "runs"
    if optional_prefix != "":
        results_dir = os.path.join(results_dir, optional_prefix)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    suffix = _infer_run_suffix(config) + _ph_param_suffix(config)

    if "FCP" in config:
        dir = Path(config["FCP"])
        f = dir.name
        run_dir = os.path.join(results_dir, f"{timestamp}_{run_id}_{layout_name}_fcp")
    else:
        run_dir = os.path.join(
            results_dir, f"{timestamp}_{run_id}_{layout_name}_{suffix}"
        )
    run_dir = Path(run_dir).resolve()
    os.makedirs(run_dir, exist_ok=True)

    print("run_dir", str(run_dir))

    return run_dir


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
