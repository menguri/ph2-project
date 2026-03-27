import copy
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
import wandb
import jax
import os
import pickle
from datetime import datetime
import jax.numpy as jnp

from overcooked_v2_experiments.human_rl.imitation.bc_policy import BCPolicy
from overcooked_v2_experiments.ppo.policy import PPOParams
from overcooked_v2_experiments.ppo.utils.fcp import FCPWrapperPolicy
from .ippo import make_train
from .utils.store import (
    load_all_checkpoints,
    store_checkpoint,
)
from .utils.utils import get_num_devices, get_run_base_dir
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.ppo.utils.visualize_ppo import visualize_ppo_policy

jax.config.update("jax_debug_nans", True)


def load_fcp_populations(population_dir: Path):
    """
    FCP population 디렉토리 아래 모든 fcp_* 폴더에서
    PPOParams 체크포인트를 전부 모아 하나의 population으로 만든다.

    - 폴더마다 policy 개수(pop_size)는 달라도 상관 없음.
    - 단, 각 policy의 params 트리 구조와 leaf shape는 동일해야 함.

    Returns
    -------
    stacked_populations : PyTree of JAX arrays
        각 leaf shape: (num_policies_total, ...)  # 모든 폴더에서 모은 policy 수
    first_fcp_config : DictConfig
        첫 번째로 발견된 fcp_config (대부분 동일할 것)
    """

    def _load_policies_from_dir(dir: Path):
        """
        하나의 fcp_* 디렉토리에서 PPOParams들을 전부 꺼내서
        [params_tree, params_tree, ...] 리스트로 반환.
        """
        all_checkpoints, fcp_config, _ = load_all_checkpoints(
            dir,
            final_only=False,
            skip_initial=True,   # 원본과 동일 동작. 필요하면 False로 바꿔도 됨.
        )

        # all_checkpoints 안에서 PPOParams만 leaf로 취급해서 리스트로 뽑기
        ppo_params_list, _ = jax.tree_util.tree_flatten(
            all_checkpoints,
            is_leaf=lambda x: isinstance(x, PPOParams),
        )

        print(
            f"Loaded FCP population params for {len(ppo_params_list)} policies from {dir}"
        )

        # 각 PPOParams에서 .params만 꺼내서 순수 params 트리로 변환
        params_list = [p.params for p in ppo_params_list]

        # 디버그: 첫 번째 policy의 shape 한 번 찍어보기
        if params_list:
            shapes = jax.tree_util.tree_map(lambda x: x.shape, params_list[0])
            print(f"[DEBUG] Example policy shapes in {dir.name}: {shapes}")

        return params_list, fcp_config

    # ----------------------------------------------------------------------
    # 1. population_dir 아래 모든 fcp_* 폴더에서 policy params 모으기
    # ----------------------------------------------------------------------
    all_policy_params = []   # 모든 폴더의 params를 여기로 평탄하게 모음
    first_fcp_config = None

    population_dir = Path(population_dir)
    if not population_dir.exists():
        raise ValueError(f"Population dir does not exist: {population_dir}")

    for dir in sorted(population_dir.iterdir()):
        if not dir.is_dir() or "fcp_" not in dir.name:
            continue

        print(f"Loading FCP population from {dir}")
        params_list, fcp_config = _load_policies_from_dir(dir)

        # 이 폴더의 policy들을 전체 리스트에 추가
        all_policy_params.extend(params_list)

        if first_fcp_config is None:
            first_fcp_config = fcp_config

    if len(all_policy_params) == 0:
        raise ValueError(f"No PPOParams found under {population_dir}")

    print(f"Successfully collected {len(all_policy_params)} FCP policies in total.")

    # ----------------------------------------------------------------------
    # 2. 모든 policy params를 한 번에 stack → (num_policies_total, ...)
    # ----------------------------------------------------------------------
    stacked_populations = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *all_policy_params
    )

    # 디버그: 최종 population shape
    fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, stacked_populations)
    print("[DEBUG] Final stacked FCP params shape:", fcp_params_shape)

    return stacked_populations, first_fcp_config


def load_mep_population(population_dir: Path):
    """
    MEP S1이 저장한 population 디렉토리에서 actor params를 로드한다.
    각 member는 init / mid / final 3개 체크포인트를 저장한다.
    N members × 3 ckpts = N×3 policies (stacked pytree, leaf shape (N*3, ...)).

    Returns
    -------
    stacked_params : PyTree with leaf shape (N*3, ...)
    """
    population_dir = Path(population_dir)
    if not population_dir.exists():
        raise ValueError(f"MEP population dir not found: {population_dir}")

    member_dirs = sorted(
        [d for d in population_dir.iterdir() if d.is_dir() and d.name.startswith("member_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not member_dirs:
        raise ValueError(f"No member_* dirs found in {population_dir}")

    _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
    params_list = []
    for md in member_dirs:
        for ckpt_name in _CKPT_NAMES:
            pkl = md / ckpt_name
            if not pkl.exists():
                raise ValueError(f"Missing {ckpt_name} in {md}")
            with open(pkl, "rb") as f:
                params_list.append(pickle.load(f))

    n_policies = len(params_list)
    print(f"[MEP] Loaded {n_policies} policies ({len(member_dirs)} members × 3 ckpts) from {population_dir}")

    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params_list)
    return stacked


def _run_s2_multi_seed(config, pop_params, label="S2"):
    """S2 multi-seed 학습 공통 로직 (MEP/GAMMA/HSP S2 모두 동일)."""
    from overcooked_v2_experiments.ppo.mep.mep_s2 import make_train_mep_s2
    train_fn = make_train_mep_s2(config)

    num_seeds = config["NUM_SEEDS"]
    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        rngs = jax.device_put(rngs, jax.devices("cpu")[0])

        num_devices = get_num_devices()
        print(f"[{label}] num_seeds={num_seeds}, num_devices={num_devices}")

        def _train_s2(rng_s):
            return train_fn(rng_s, population=pop_params)

        if num_devices <= 1:
            train_jit = jax.jit(_train_s2)
            if num_seeds == 1:
                out = train_jit(rngs[0])
            else:
                out = jax.vmap(train_jit)(rngs)
        else:
            if num_seeds == num_devices:
                out = jax.pmap(_train_s2)(rngs)
            elif num_seeds % num_devices == 0:
                seeds_per_device = num_seeds // num_devices
                rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                out = jax.pmap(jax.vmap(_train_s2))(rngs_2d)
                out = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_seeds, *x.shape[2:])), out
                )
            else:
                print(f"[warn] num_seeds({num_seeds}) % num_devices({num_devices}) != 0; fallback to vmap")
                train_jit = jax.jit(_train_s2)
                out = jax.vmap(train_jit)(rngs)

    # 체크포인트 저장
    num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
    if num_checkpoints > 0:
        actor_ckpts = out["actor_ckpts"]
        sample_leaf = jax.tree_util.tree_leaves(actor_ckpts)[0]
        has_seed_axis = sample_leaf.shape[0] == num_seeds and num_seeds > 1
        for s in range(num_seeds):
            if has_seed_axis:
                seed_ckpts = jax.tree_util.tree_map(lambda x: x[s], actor_ckpts)
            else:
                seed_ckpts = actor_ckpts
            for slot in range(num_checkpoints - 1):
                params = jax.tree_util.tree_map(lambda x: x[slot], seed_ckpts)
                store_checkpoint(config, params, s, slot, final=False)
            params_final = jax.tree_util.tree_map(
                lambda x: x[num_checkpoints - 1], seed_ckpts
            )
            store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)
        print(f"[{label}] Saved {num_seeds} seeds × {num_checkpoints} ckpts to {config['RUN_BASE_DIR']}")

    return out


def _save_population(pop_ckpts, pop_dir, label="POP"):
    """Population checkpoint을 디스크에 저장 (MEP/GAMMA 공통)."""
    N = jax.tree_util.tree_leaves(pop_ckpts)[0].shape[0]
    _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
    pop_dir = Path(pop_dir)
    pop_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        member_dir = pop_dir / f"member_{i}"
        member_dir.mkdir(exist_ok=True)
        for slot, ckpt_name in enumerate(_CKPT_NAMES):
            member_ckpt = jax.tree_util.tree_map(lambda x: x[i, slot], pop_ckpts)
            with open(member_dir / ckpt_name, "wb") as f:
                pickle.dump(member_ckpt, f)
    print(f"[{label}] Saved {N} members × 3 ckpts to {pop_dir}")
    return N


def _run_unified_mep(config):
    """MEP S1 → S2 통합 실행."""
    from overcooked_v2_experiments.ppo.mep.mep_s1 import make_train_mep_s1

    run_base_dir = Path(config["RUN_BASE_DIR"])

    # ---- S1: Population training ----
    print("=" * 60)
    print("[MEP] Stage 1: Population training")
    print("=" * 60)
    train_fn = make_train_mep_s1(config)
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        out_s1 = jax.jit(train_fn)(rng)

    pop_dir = run_base_dir / "mep_population"
    _save_population(out_s1["pop_actor_ckpts"], pop_dir, "MEP S1")

    # ---- S2: Adaptive agent training ----
    print("=" * 60)
    print("[MEP] Stage 2: Adaptive agent training")
    print("=" * 60)
    pop_params = load_mep_population(pop_dir)
    s2_config = _make_s2_config(config)
    out_s2 = _run_s2_multi_seed(s2_config, pop_params, "MEP S2")

    return {"s1": out_s1, "s2": out_s2}


def _run_unified_gamma(config):
    """GAMMA S1 → S2 통합 실행. S2는 method에 따라 rl 또는 vae."""
    from overcooked_v2_experiments.ppo.mep.mep_s1 import make_train_mep_s1

    run_base_dir = Path(config["RUN_BASE_DIR"])
    method = config.get("GAMMA_S2_METHOD", "rl")

    # ---- S1: Population training (MEP S1과 동일) ----
    print("=" * 60)
    print("[GAMMA] Stage 1: Population training")
    print("=" * 60)
    train_fn = make_train_mep_s1(config)
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        out_s1 = jax.jit(train_fn)(rng)

    pop_dir = run_base_dir / "gamma_population"
    _save_population(out_s1["pop_actor_ckpts"], pop_dir, "GAMMA S1")

    # ---- S2 ----
    if method == "vae":
        print("=" * 60)
        print("[GAMMA] Stage 2: VAE training + z-conditioned RL")
        print("=" * 60)
        from overcooked_v2_experiments.ppo.gamma.gamma_s2_vae import run_gamma_s2_vae
        pop_params = load_mep_population(pop_dir)
        s2_config = _make_s2_config(config)
        out_s2 = run_gamma_s2_vae(s2_config, pop_params, pop_dir)
    else:
        print("=" * 60)
        print("[GAMMA] Stage 2: Adaptive agent training (standard RL)")
        print("=" * 60)
        pop_params = load_mep_population(pop_dir)
        s2_config = _make_s2_config(config)
        out_s2 = _run_s2_multi_seed(s2_config, pop_params, "GAMMA S2")

    return {"s1": out_s1, "s2": out_s2}


def _run_unified_hsp(config):
    """HSP S1 → Greedy Selection → S2 통합 실행."""
    from overcooked_v2_experiments.ppo.hsp.hsp_s1 import make_train_hsp_s1
    from overcooked_v2_experiments.ppo.hsp.greedy_selector import (
        collect_event_features, greedy_select_policies,
    )
    import numpy as np
    import json

    run_base_dir = Path(config["RUN_BASE_DIR"])

    # ---- S1: Utility weight population training ----
    print("=" * 60)
    print("[HSP] Stage 1: Utility weight population training")
    print("=" * 60)
    train_fn = make_train_hsp_s1(config)
    rng = jax.random.PRNGKey(config["SEED"])
    all_ckpts, all_weights = train_fn(rng)

    # 전체 N개 policy 저장
    pop_all_dir = run_base_dir / "hsp_population_all"
    pop_all_dir.mkdir(parents=True, exist_ok=True)
    _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
    for i, ckpts_i in enumerate(all_ckpts):
        member_dir = pop_all_dir / f"member_{i}"
        member_dir.mkdir(exist_ok=True)
        for slot, ckpt_name in enumerate(_CKPT_NAMES):
            member_ckpt = jax.tree_util.tree_map(lambda x: x[slot], ckpts_i)
            with open(member_dir / ckpt_name, "wb") as f:
                pickle.dump(member_ckpt, f)
    np.save(pop_all_dir / "utility_weights.npy", np.array(all_weights))
    print(f"[HSP S1] Saved {len(all_ckpts)} policies to {pop_all_dir}")

    # ---- Greedy Selection ----
    print("=" * 60)
    print("[HSP] Greedy diversity selection")
    print("=" * 60)
    event_matrix = collect_event_features(pop_all_dir, config)
    K = config.get("HSP_SELECTED_K", 18)
    selected = greedy_select_policies(event_matrix, K)

    pop_dir = run_base_dir / "hsp_population"
    pop_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, orig_idx in enumerate(selected):
        src = pop_all_dir / f"member_{orig_idx}"
        dst = pop_dir / f"member_{new_idx}"
        shutil.copytree(src, dst)
    with open(pop_dir / "selected_indices.json", "w") as f:
        json.dump(selected, f)
    print(f"[HSP] Selected {K} from {len(all_ckpts)} → {pop_dir}")

    # ---- S2: Adaptive agent training ----
    print("=" * 60)
    print("[HSP] Stage 2: Adaptive agent training")
    print("=" * 60)
    pop_params = load_mep_population(pop_dir)
    s2_config = _make_s2_config(config)
    # HSP S2에서는 shaping horizon을 S2용으로 교체
    if "S2_REW_SHAPING_HORIZON" in config.get("model", {}):
        s2_config["model"]["REW_SHAPING_HORIZON"] = config["model"]["S2_REW_SHAPING_HORIZON"]
    out_s2 = _run_s2_multi_seed(s2_config, pop_params, "HSP S2")

    return {"s1_ckpts": all_ckpts, "selected": selected, "s2": out_s2}


def _make_s2_config(config):
    """S1 config에서 S2용 config를 생성한다."""
    import copy
    s2 = copy.deepcopy(config)
    # S2 timesteps
    if "S2_TOTAL_TIMESTEPS" in s2.get("model", {}):
        s2["model"]["TOTAL_TIMESTEPS"] = s2["model"]["S2_TOTAL_TIMESTEPS"]
    # S2 seeds
    if "S2_NUM_SEEDS" in s2:
        s2["NUM_SEEDS"] = s2["S2_NUM_SEEDS"]
    # S2 shaping horizon
    if "S2_REW_SHAPING_HORIZON" in s2.get("model", {}):
        s2["model"]["REW_SHAPING_HORIZON"] = s2["model"]["S2_REW_SHAPING_HORIZON"]
    return s2


def single_run(config):
    alg_name = config.get("ALG_NAME", "SP")

    # ================================================================
    # 통합 파이프라인: ALG_NAME이 "MEP", "GAMMA", "HSP"이면
    # S1 → (greedy selection) → S2를 하나의 run에서 자동 진행.
    # 결과 디렉토리 구조:
    #   {RUN_BASE_DIR}/
    #   ├── {alg}_population/    # S1 결과
    #   └── run_{s}/ckpt_*/      # S2 결과
    # ================================================================
    if alg_name == "MEP":
        return _run_unified_mep(config)
    if alg_name == "GAMMA":
        return _run_unified_gamma(config)
    if alg_name == "HSP":
        return _run_unified_hsp(config)

    # ----------------------------------------------------------------
    # MEP_S1: population training (레거시 — 개별 S1/S2도 계속 지원)
    # ----------------------------------------------------------------
    if alg_name == "MEP_S1":
        from overcooked_v2_experiments.ppo.mep.mep_s1 import make_train_mep_s1
        train_fn = make_train_mep_s1(config)
        rng = jax.random.PRNGKey(config["SEED"])
        with jax.disable_jit(False):
            out = jax.jit(train_fn)(rng)

        # Save N members × 3 checkpoints (init/mid/final) per member
        # pop_actor_ckpts leaf shape: (N, 3, ...)
        pop_ckpts = out["pop_actor_ckpts"]
        N = jax.tree_util.tree_leaves(pop_ckpts)[0].shape[0]
        _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
        run_base_dir = Path(config["RUN_BASE_DIR"])
        pop_dir = run_base_dir / "mep_population"
        pop_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N):
            member_dir = pop_dir / f"member_{i}"
            member_dir.mkdir(exist_ok=True)
            for slot, ckpt_name in enumerate(_CKPT_NAMES):
                member_ckpt = jax.tree_util.tree_map(lambda x: x[i, slot], pop_ckpts)
                with open(member_dir / ckpt_name, "wb") as f:
                    pickle.dump(member_ckpt, f)
        print(f"[MEP S1] Saved {N} members × 3 ckpts to {pop_dir}")
        return out

    # ----------------------------------------------------------------
    # MEP_S2: adaptive agent training (multi-seed, same as standard training)
    # ----------------------------------------------------------------
    if alg_name == "MEP_S2":
        from overcooked_v2_experiments.ppo.mep.mep_s2 import make_train_mep_s2
        pop_dir = Path(config["MEP_POPULATION_DIR"])
        if not Path(pop_dir).exists():
            raise ValueError(
                f"[MEP S2] Population dir not found: {pop_dir}\n"
                "Run MEP Stage 1 first (./sh_scripts/run_factory_mep_s1.sh) "
                "or provide --pop-dir."
            )
        pop_params = load_mep_population(pop_dir)
        train_fn = make_train_mep_s2(config)

        num_seeds = config["NUM_SEEDS"]
        with jax.disable_jit(False):
            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng, num_seeds)
            rngs = jax.device_put(rngs, jax.devices("cpu")[0])

            num_devices = get_num_devices()
            print(f"[MEP S2] num_seeds={num_seeds}, num_devices={num_devices}")

            def _train_s2(rng_s):
                return train_fn(rng_s, population=pop_params)

            if num_devices <= 1:
                train_jit = jax.jit(_train_s2)
                if num_seeds == 1:
                    out = train_jit(rngs[0])
                else:
                    out = jax.vmap(train_jit)(rngs)
            else:
                if num_seeds == num_devices:
                    out = jax.pmap(_train_s2)(rngs)
                elif num_seeds % num_devices == 0:
                    seeds_per_device = num_seeds // num_devices
                    rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                    out = jax.pmap(jax.vmap(_train_s2))(rngs_2d)
                    out = jax.tree_util.tree_map(
                        lambda x: x.reshape((num_seeds, *x.shape[2:])), out
                    )
                else:
                    print(f"[warn] num_seeds({num_seeds}) % num_devices({num_devices}) != 0; fallback to vmap")
                    train_jit = jax.jit(_train_s2)
                    out = jax.vmap(train_jit)(rngs)

        # Save checkpoints: run_{s}/ckpt_0, ckpt_1, ckpt_final (orbax format)
        num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
        if num_checkpoints > 0:
            actor_ckpts = out["actor_ckpts"]
            sample_leaf = jax.tree_util.tree_leaves(actor_ckpts)[0]
            # num_seeds==1: shape (num_checkpoints, ...) — no seed axis
            # num_seeds > 1: shape (num_seeds, num_checkpoints, ...)
            has_seed_axis = sample_leaf.shape[0] == num_seeds and num_seeds > 1

            for s in range(num_seeds):
                # extract (num_checkpoints, ...) for this seed
                if has_seed_axis:
                    seed_ckpts = jax.tree_util.tree_map(lambda x: x[s], actor_ckpts)
                else:
                    seed_ckpts = actor_ckpts
                # save non-final slots (0 .. num_checkpoints-2)
                for slot in range(num_checkpoints - 1):
                    params = jax.tree_util.tree_map(lambda x: x[slot], seed_ckpts)
                    store_checkpoint(config, params, s, slot, final=False)
                # save last slot as ckpt_final
                params_final = jax.tree_util.tree_map(
                    lambda x: x[num_checkpoints - 1], seed_ckpts
                )
                store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)
            print(f"[MEP S2] Saved {num_seeds} seeds × {num_checkpoints} ckpts to {config['RUN_BASE_DIR']}")

        return out

    # ----------------------------------------------------------------
    # GAMMA_S1: MEP S1과 동일 알고리즘, population dir만 gamma_population으로 변경
    # ----------------------------------------------------------------
    if alg_name == "GAMMA_S1":
        from overcooked_v2_experiments.ppo.mep.mep_s1 import make_train_mep_s1
        train_fn = make_train_mep_s1(config)
        rng = jax.random.PRNGKey(config["SEED"])
        with jax.disable_jit(False):
            out = jax.jit(train_fn)(rng)

        pop_ckpts = out["pop_actor_ckpts"]
        N = jax.tree_util.tree_leaves(pop_ckpts)[0].shape[0]
        _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
        run_base_dir = Path(config["RUN_BASE_DIR"])
        pop_dir = run_base_dir / "gamma_population"
        pop_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N):
            member_dir = pop_dir / f"member_{i}"
            member_dir.mkdir(exist_ok=True)
            for slot, ckpt_name in enumerate(_CKPT_NAMES):
                member_ckpt = jax.tree_util.tree_map(lambda x: x[i, slot], pop_ckpts)
                with open(member_dir / ckpt_name, "wb") as f:
                    pickle.dump(member_ckpt, f)
        print(f"[GAMMA S1] Saved {N} members × 3 ckpts to {pop_dir}")
        return out

    # ----------------------------------------------------------------
    # GAMMA_S2: MEP S2와 동일, GAMMA_POPULATION_DIR에서 population 로드
    # ----------------------------------------------------------------
    if alg_name == "GAMMA_S2":
        from overcooked_v2_experiments.ppo.mep.mep_s2 import make_train_mep_s2
        pop_dir = Path(config.get("GAMMA_POPULATION_DIR", config.get("MEP_POPULATION_DIR", "")))
        if not pop_dir.exists():
            raise ValueError(
                f"[GAMMA S2] Population dir not found: {pop_dir}\n"
                "Run GAMMA Stage 1 first or provide +GAMMA_POPULATION_DIR."
            )
        pop_params = load_mep_population(pop_dir)
        train_fn = make_train_mep_s2(config)

        num_seeds = config["NUM_SEEDS"]
        with jax.disable_jit(False):
            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng, num_seeds)
            rngs = jax.device_put(rngs, jax.devices("cpu")[0])

            num_devices = get_num_devices()
            print(f"[GAMMA S2] num_seeds={num_seeds}, num_devices={num_devices}")

            def _train_gamma_s2(rng_s):
                return train_fn(rng_s, population=pop_params)

            if num_devices <= 1:
                train_jit = jax.jit(_train_gamma_s2)
                if num_seeds == 1:
                    out = train_jit(rngs[0])
                else:
                    out = jax.vmap(train_jit)(rngs)
            else:
                if num_seeds == num_devices:
                    out = jax.pmap(_train_gamma_s2)(rngs)
                elif num_seeds % num_devices == 0:
                    seeds_per_device = num_seeds // num_devices
                    rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                    out = jax.pmap(jax.vmap(_train_gamma_s2))(rngs_2d)
                    out = jax.tree_util.tree_map(
                        lambda x: x.reshape((num_seeds, *x.shape[2:])), out
                    )
                else:
                    print(f"[warn] num_seeds({num_seeds}) % num_devices({num_devices}) != 0; fallback to vmap")
                    train_jit = jax.jit(_train_gamma_s2)
                    out = jax.vmap(train_jit)(rngs)

        # 체크포인트 저장 (MEP S2와 동일)
        num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
        if num_checkpoints > 0:
            actor_ckpts = out["actor_ckpts"]
            sample_leaf = jax.tree_util.tree_leaves(actor_ckpts)[0]
            has_seed_axis = sample_leaf.shape[0] == num_seeds and num_seeds > 1

            for s in range(num_seeds):
                if has_seed_axis:
                    seed_ckpts = jax.tree_util.tree_map(lambda x: x[s], actor_ckpts)
                else:
                    seed_ckpts = actor_ckpts
                for slot in range(num_checkpoints - 1):
                    params = jax.tree_util.tree_map(lambda x: x[slot], seed_ckpts)
                    store_checkpoint(config, params, s, slot, final=False)
                params_final = jax.tree_util.tree_map(
                    lambda x: x[num_checkpoints - 1], seed_ckpts
                )
                store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)
            print(f"[GAMMA S2] Saved {num_seeds} seeds × {num_checkpoints} ckpts to {config['RUN_BASE_DIR']}")

        return out

    # ----------------------------------------------------------------
    # HSP_S1: N개 독립 정책을 utility weight로 순차 학습 + greedy selection
    # ----------------------------------------------------------------
    if alg_name == "HSP_S1":
        from overcooked_v2_experiments.ppo.hsp.hsp_s1 import make_train_hsp_s1
        train_fn = make_train_hsp_s1(config)
        rng = jax.random.PRNGKey(config["SEED"])
        all_ckpts, all_weights = train_fn(rng)

        # 전체 N개 policy를 hsp_population_all/에 저장
        import numpy as np
        run_base_dir = Path(config["RUN_BASE_DIR"])
        pop_all_dir = run_base_dir / "hsp_population_all"
        pop_all_dir.mkdir(parents=True, exist_ok=True)
        _CKPT_NAMES = ["ckpt_init_actor.pkl", "ckpt_mid_actor.pkl", "ckpt_final_actor.pkl"]
        for i, ckpts_i in enumerate(all_ckpts):
            member_dir = pop_all_dir / f"member_{i}"
            member_dir.mkdir(exist_ok=True)
            for slot, ckpt_name in enumerate(_CKPT_NAMES):
                member_ckpt = jax.tree_util.tree_map(lambda x: x[slot], ckpts_i)
                with open(member_dir / ckpt_name, "wb") as f:
                    pickle.dump(member_ckpt, f)
        np.save(pop_all_dir / "utility_weights.npy", np.array(all_weights))
        print(f"[HSP S1] Saved {len(all_ckpts)} policies to {pop_all_dir}")

        # Greedy selection
        from overcooked_v2_experiments.ppo.hsp.greedy_selector import (
            collect_event_features, greedy_select_policies,
        )
        event_matrix = collect_event_features(pop_all_dir, config)
        K = config.get("HSP_SELECTED_K", 18)
        selected = greedy_select_policies(event_matrix, K)

        # 선택된 policy만 hsp_population/에 복사
        pop_dir = run_base_dir / "hsp_population"
        pop_dir.mkdir(parents=True, exist_ok=True)
        for new_idx, orig_idx in enumerate(selected):
            src = pop_all_dir / f"member_{orig_idx}"
            dst = pop_dir / f"member_{new_idx}"
            shutil.copytree(src, dst)

        import json
        with open(pop_dir / "selected_indices.json", "w") as f:
            json.dump(selected, f)
        print(f"[HSP S1] {len(all_ckpts)} policies trained, {K} selected → {pop_dir}")
        return {"all_ckpts": all_ckpts, "selected": selected}

    # ----------------------------------------------------------------
    # HSP_S2: MEP S2와 동일, HSP_POPULATION_DIR에서 population 로드
    # ----------------------------------------------------------------
    if alg_name == "HSP_S2":
        from overcooked_v2_experiments.ppo.mep.mep_s2 import make_train_mep_s2
        pop_dir = Path(config["HSP_POPULATION_DIR"])
        if not pop_dir.exists():
            raise ValueError(
                f"[HSP S2] Population dir not found: {pop_dir}\n"
                "Run HSP Stage 1 first or provide +HSP_POPULATION_DIR."
            )
        pop_params = load_mep_population(pop_dir)
        train_fn = make_train_mep_s2(config)

        num_seeds = config["NUM_SEEDS"]
        with jax.disable_jit(False):
            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng, num_seeds)
            rngs = jax.device_put(rngs, jax.devices("cpu")[0])

            num_devices = get_num_devices()
            print(f"[HSP S2] num_seeds={num_seeds}, num_devices={num_devices}")

            def _train_hsp_s2(rng_s):
                return train_fn(rng_s, population=pop_params)

            if num_devices <= 1:
                train_jit = jax.jit(_train_hsp_s2)
                if num_seeds == 1:
                    out = train_jit(rngs[0])
                else:
                    out = jax.vmap(train_jit)(rngs)
            else:
                if num_seeds == num_devices:
                    out = jax.pmap(_train_hsp_s2)(rngs)
                elif num_seeds % num_devices == 0:
                    seeds_per_device = num_seeds // num_devices
                    rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                    out = jax.pmap(jax.vmap(_train_hsp_s2))(rngs_2d)
                    out = jax.tree_util.tree_map(
                        lambda x: x.reshape((num_seeds, *x.shape[2:])), out
                    )
                else:
                    print(f"[warn] num_seeds({num_seeds}) % num_devices({num_devices}) != 0; fallback to vmap")
                    train_jit = jax.jit(_train_hsp_s2)
                    out = jax.vmap(train_jit)(rngs)

        # 체크포인트 저장 (MEP S2와 동일)
        num_checkpoints = config.get("NUM_CHECKPOINTS", 0)
        if num_checkpoints > 0:
            actor_ckpts = out["actor_ckpts"]
            sample_leaf = jax.tree_util.tree_leaves(actor_ckpts)[0]
            has_seed_axis = sample_leaf.shape[0] == num_seeds and num_seeds > 1

            for s in range(num_seeds):
                if has_seed_axis:
                    seed_ckpts = jax.tree_util.tree_map(lambda x: x[s], actor_ckpts)
                else:
                    seed_ckpts = actor_ckpts
                for slot in range(num_checkpoints - 1):
                    params = jax.tree_util.tree_map(lambda x: x[slot], seed_ckpts)
                    store_checkpoint(config, params, s, slot, final=False)
                params_final = jax.tree_util.tree_map(
                    lambda x: x[num_checkpoints - 1], seed_ckpts
                )
                store_checkpoint(config, params_final, s, num_checkpoints - 1, final=True)
            print(f"[HSP S2] Saved {num_seeds} seeds × {num_checkpoints} ckpts to {config['RUN_BASE_DIR']}")

        return out

    # ----------------------------------------------------------------
    # Standard SP / FCP / BC
    # ----------------------------------------------------------------
    num_seeds = config["NUM_SEEDS"]
    num_runs = num_seeds

    all_populations = None
    if "FCP" in config:
        print("Training FCP")
        # assert num_seeds == 1
        print("Loading population from", config["FCP"])
        population_dir = Path(config["FCP"])

        all_populations, fcp_population_config = load_fcp_populations(population_dir)
        # all_populations = all_populations.params

        # fcp population은 num_runs와 별도의 축 
        pop_size = jax.tree_util.tree_flatten(all_populations)[0][0].shape[0]

        print(f"Loaded FCP population with {num_runs} runs")

        fcp_params_shape = jax.tree_util.tree_map(lambda x: x.shape, all_populations)
        print("FCP params shape", fcp_params_shape)

    bc_policy = None
    if "BC" in config:
        print("Training with BC")
        layout_name = config["env"]["ENV_KWARGS"].get("layout", config["env"].get("ENV_NAME", "unknown"))
        split = "all"
        run_id = 1
        print(f"Loading BC policy from {layout_name}-{split}-{run_id}")
        bc_policy = BCPolicy.from_pretrained(layout_name, split, run_id)

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_runs)
        
        # 디버그: 생성된 시드 값 확인
        print(f"[DEBUG] Generated rngs (first elements): {[int(k[0]) for k in rngs]}")

        config_copy = copy.deepcopy(config)
        if bc_policy is not None:
            config_copy["env"]["ENV_KWARGS"]["force_path_planning"] = True

        population_config = None
        if all_populations is not None:
            population_config = fcp_population_config

        train_func = make_train(
            config_copy,
            population_config=population_config,
        )

        # num_devices = len(jax.devices("gpu"))
        num_devices = get_num_devices()
        print("Using", num_devices, "devices")

        # ---- FCP일 때: population은 클로저로 고정 ----
        # if all_populations is not None:
        #     print("Training with FCP")

        #     def train_with_pop(rng):
        #         # 여기서 population을 고정 파라미터로 넣어줌
        #         return train_func(rng, population=all_populations)

        #     train_with_pop_jit = jax.jit(train_with_pop)

        #     # ✅ 여기서 mini_batch_pmap 재사용
        #     # out = mini_batch_pmap(train_with_pop_jit, num_devices)(rngs)
        #     # return out
            
        #     # Explicit pmap logic to avoid ambiguity
        #     seed_n = rngs.shape[0]
        #     print(f"[DEBUG] seed_n={seed_n}, num_devices={num_devices}")
        #     if num_devices <= 1:
        #         if seed_n == 1:
        #             print("[DEBUG] Running single device, single seed")
        #             out = train_with_pop_jit(rngs[0])
        #         else:
        #             print("[DEBUG] Running single device, vmap")
        #             out = jax.vmap(train_with_pop_jit)(rngs)
        #     else:
        #         if seed_n == num_devices:
        #             print("[DEBUG] Running pmap (1 seed per device)")
        #             out = jax.pmap(train_with_pop_jit)(rngs)
        #         elif seed_n % num_devices == 0:
        #             seeds_per_device = seed_n // num_devices
        #             print(f"[DEBUG] Running pmap+vmap (seeds_per_device={seeds_per_device})")
        #             rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
        #             out = jax.pmap(jax.vmap(train_with_pop_jit))(rngs_2d)
        #             out = jax.tree_util.tree_map(lambda x: x.reshape((seed_n, *x.shape[2:])), out)
        #         else:
        #             print(f"[warn] num_seeds({seed_n}) % num_devices({num_devices}) != 0; falling back to single-device vmap")
        #             out = jax.vmap(train_with_pop_jit)(rngs)
        #     return out


        # 시드 하나 단위로 학습을 돌리기 위해, 그 학습을 병렬로 처리
        # train_jit = jax.jit(train_func)  <-- pmap 내부에서 jit을 또 부르는 것을 방지하기 위해 주석 처리

        train_extra_args = {}
        if all_populations is not None:
            print("Training with FCP")
            train_extra_args["population"] = all_populations
        elif bc_policy is not None:
            print("Training with BC")
            print("Using BC policy", bc_policy)
            train_extra_args["population"] = bc_policy

        # out = mini_batch_pmap(train_jit, num_devices)(rngs, **train_extra_args)
        
        # Explicit pmap logic for SP/BC
        def train_wrapper(rng):
            # pmap/vmap 내부에서 실행되므로 여기서 train_func를 직접 호출 (JAX가 알아서 컴파일)
            return train_func(rng, **train_extra_args)
            
        seed_n = rngs.shape[0]
        
        # 중요: 멀티 GPU 분배 시 P2P 복사 문제를 방지하기 위해 rngs를 CPU로 이동
        rngs = jax.device_put(rngs, jax.devices("cpu")[0])
        
        if num_devices <= 1:
            # 단일 디바이스일 때는 JIT를 명시적으로 걸어주는 것이 좋음 (pmap을 안 쓰므로)
            train_jit = jax.jit(train_func)
            def train_wrapper_jit(rng):
                return train_jit(rng, **train_extra_args)

            if seed_n == 1:
                out = train_wrapper_jit(rngs[0])
            else:
                out = jax.vmap(train_wrapper_jit)(rngs)
        else:
            if seed_n == num_devices:
                out = jax.pmap(train_wrapper)(rngs)
            elif seed_n % num_devices == 0:
                seeds_per_device = seed_n // num_devices
                rngs_2d = rngs.reshape((num_devices, seeds_per_device, *rngs.shape[1:]))
                
                # Debug: Check if rngs_2d contains zeros
                # We can't easily print JAX arrays here without triggering computation, 
                # but we can check if it's all zeros if we suspect initialization issues.
                # Instead, let's rely on ippo.py's debug print.
                
                out = jax.pmap(jax.vmap(train_wrapper))(rngs_2d)
                out = jax.tree_util.tree_map(lambda x: x.reshape((seed_n, *x.shape[2:])), out)
            else:
                print(f"[warn] num_seeds({seed_n}) % num_devices({num_devices}) != 0; falling back to single-device vmap")
                # fallback 시에도 jit 사용
                train_jit = jax.jit(train_func)
                def train_wrapper_jit(rng):
                    return train_jit(rng, **train_extra_args)
                out = jax.vmap(train_wrapper_jit)(rngs)

        return out
