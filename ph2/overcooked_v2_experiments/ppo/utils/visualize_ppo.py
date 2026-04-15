import argparse
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import copy
import numpy as np
from datetime import datetime
from pathlib import Path
import chex
import imageio
import csv


DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from overcooked_v2_experiments.ppo.policy import (
    PPOParams,
    PPOPolicy,
    policy_checkoints_to_policy_pairing,
)
from overcooked_v2_experiments.ppo.utils.store import (
    load_all_checkpoints,
)
from overcooked_v2_experiments.helper.plots import visualize_cross_play_matrix
from overcooked_v2_experiments.utils.utils import (
    mini_batch_pmap,          # 지금은 안 쓰지만, 기존 인터페이스 유지용으로 둠
    scanned_mini_batch_map,
)
from overcooked_v2_experiments.eval.evaluate import eval_pairing, PolicyVizualization
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.eval.rollout import get_rollout
from overcooked_v2_experiments.eval.utils import (
    make_eval_env,
    render_state_frame,
)


def visualize_ppo_policy(
    run_base_dir,
    key,
    final_only=True,
    extra_env_kwargs={},
    num_seeds=None,
    cross=False,
    cross_mode="full",
    no_viz=False,
    pairing_policy=None,
    latent_analysis=False,
    value_analysis=False,
    policy_source="params",
    eval_reward="sparse",
    old_overcooked_override=None,
    disable_old_overcooked_auto_override=None,
    ckpt_filter=None,
):
    # cross-play인데 모든 ckpt를 쓰려고 하면 모순 → 방어 코드 (ckpt_filter 지정 시 허용)
    if cross and not final_only and ckpt_filter is None:
        raise ValueError("Cannot run cross play with all checkpoints")

    if value_analysis and cross:
        raise ValueError("value_analysis is only supported for self-play")

    # 1) PPO 파라미터 로드: all_params: dict[run_id][ckpt_id] -> PPOParams
    all_params, config, configs = load_all_checkpoints(
        run_base_dir,
        final_only=final_only,
        policy_source=policy_source,
        ckpt_filter=ckpt_filter,
    )

    # 2) 환경 생성
    # 일부 알고리즘은 ckpt 저장 시 OmegaConf nested 직렬화 누락으로 config["env"]가 None일 수 있음
    # → extra_env_kwargs로 fallback.
    _env_section = config.get("env") if isinstance(config, dict) else None
    if _env_section and isinstance(_env_section, dict) and _env_section.get("ENV_KWARGS"):
        initial_env_kwargs = copy.deepcopy(_env_section["ENV_KWARGS"])
    else:
        initial_env_kwargs = {}
    env_kwargs = initial_env_kwargs | extra_env_kwargs
    _extra_env_name = env_kwargs.pop("ENV_NAME", None)
    cfg_old_overcooked = False
    cfg_disable_old_auto = True
    if old_overcooked_override or disable_old_overcooked_auto_override:
        print("[WARN] --old_overcooked/--disable_old_overcooked_auto are ignored. Eval is fixed to overcooked_v2.")
    env_layout = env_kwargs.get("layout")
    env_kwargs_no_layout = copy.deepcopy(env_kwargs)
    env_kwargs_no_layout.pop("layout", None)
    # ToyCoop, MPE 등 layout 없는 환경 감지
    if _env_section and isinstance(_env_section, dict):
        _env_name = _env_section.get("ENV_NAME", "overcooked_v2")
    elif _extra_env_name:
        _env_name = _extra_env_name
    else:
        _env_name = "overcooked_v2"
    _env_name_override = _env_name if (_env_name in ("ToyCoop", "GridSpread") or _env_name.startswith("MPE_")) else None
    # GridSpread: 학습 config의 early_terminate 값을 그대로 사용 (학습/eval 정합).
    env, engine_name, _resolved_kwargs = make_eval_env(
        env_layout,
        env_kwargs_no_layout,
        old_overcooked=False,
        disable_auto=True,
        env_name_override=_env_name_override,
    )

    num_actors = env.num_agents

    # 환경의 실제 ACTION_DIM, NUM_PARTNERS를 모든 config에 주입 (체크포인트에 없을 수 있음)
    _action_dim = env.action_space(env.agents[0]).n
    _num_partners = env.num_agents - 1
    for _cfg in configs.values():
        if isinstance(_cfg, dict) and "model" in _cfg:
            _cfg["model"]["ACTION_DIM"] = _action_dim
            _cfg["model"]["NUM_PARTNERS"] = _num_partners
    if config and isinstance(config, dict) and "model" in config:
        config["model"]["ACTION_DIM"] = _action_dim
        config["model"]["NUM_PARTNERS"] = _num_partners

    # CT v2: TRANSFORMER_STATE_SHAPE가 누락된 기존 체크포인트 대응
    # env를 reset해서 full global obs shape을 추론
    def _inject_ct_state_shape(cfg):
        if not cfg.get("TRANSFORMER_V2", False):
            return
        if cfg.get("TRANSFORMER_STATE_SHAPE"):
            return
        _rng = jax.random.PRNGKey(0)
        _obs, _state = env.reset(_rng)
        _full_obs = env.get_obs_default(_state)  # (num_agents, H, W, C)
        cfg["TRANSFORMER_STATE_SHAPE"] = list(_full_obs[0].shape)

    _inject_ct_state_shape(config)
    for _cfg in configs.values():
        if isinstance(_cfg, dict):
            _inject_ct_state_shape(_cfg)

    run_keys = list(all_params.keys())

    def _resolve_policy_config(cfg):
        out = copy.deepcopy(cfg)
        source = str(policy_source).strip().lower()
        if source not in ("ind", "spec"):
            return out

        key = "PH2_IND_USE_BLOCKED_INPUT" if source == "ind" else "PH2_SPEC_USE_BLOCKED_INPUT"
        use_blocked_input = out.get(key, None)
        if "alg" in out and isinstance(out["alg"], dict):
            use_blocked_input = out["alg"].get(key, use_blocked_input)
        if use_blocked_input is None:
            # Backward-compatible fallback for old PH2 checkpoints:
            # prefer an existing learner flag if present, otherwise
            # default to ind=False / spec=True.
            learner_flag = out.get("LEARNER_USE_BLOCKED_INPUT", None)
            if "alg" in out and isinstance(out["alg"], dict):
                learner_flag = out["alg"].get(
                    "LEARNER_USE_BLOCKED_INPUT", learner_flag
                )
            if learner_flag is None:
                learner_flag = (source != "ind")
            use_blocked_input = learner_flag

        out["LEARNER_USE_BLOCKED_INPUT"] = bool(use_blocked_input)
        if "alg" in out and isinstance(out["alg"], dict):
            out["alg"]["LEARNER_USE_BLOCKED_INPUT"] = bool(use_blocked_input)
        return out

    # 3) cross-play 모드: 서로 다른 run 조합으로 PolicyPairing 구성
    if cross:
        num_runs = len(run_keys)
        MAX_CROSS_COMBOS = 200  # GridSpread N≥4 등 조합 폭발 방지: 최대 200개 cross-play 조합

        # self-play 조합은 항상 포함
        self_play_combos = [[i] * num_actors for i in range(num_runs)]

        if cross_mode == "level_one":
            # Level-One: pair (A,B)마다 3:1, 2:2, 1:3 조합 생성 (position 순서 포함)
            level_one_combos = []
            for i in range(num_runs):
                for j in range(i + 1, num_runs):
                    # 3:1 / 1:3: 1개 position만 교체
                    for pos in range(num_actors):
                        combo_a = [i] * num_actors
                        combo_a[pos] = j
                        level_one_combos.append(combo_a)
                        combo_b = [j] * num_actors
                        combo_b[pos] = i
                        level_one_combos.append(combo_b)
                    # 2:2: 2개 position을 교체 (C(N,2) 조합)
                    if num_actors >= 4:
                        for p in range(num_actors):
                            for q in range(p + 1, num_actors):
                                combo_c = [i] * num_actors
                                combo_c[p] = j
                                combo_c[q] = j
                                level_one_combos.append(combo_c)
                                combo_d = [j] * num_actors
                                combo_d[p] = i
                                combo_d[q] = i
                                level_one_combos.append(combo_d)
            cross_combos = level_one_combos
            run_combinations = cross_combos + self_play_combos
            print(f"[EVAL] Level-One cross-play: {len(cross_combos)} cross + {len(self_play_combos)} self = {len(run_combinations)} total")
        else:
            all_cross_combos = list(itertools.permutations(range(num_runs), num_actors))
            if len(all_cross_combos) <= MAX_CROSS_COMBOS:
                cross_combos = [list(c) for c in all_cross_combos]
            else:
                _rng_np = np.random.default_rng(42)
                indices = _rng_np.choice(len(all_cross_combos), size=MAX_CROSS_COMBOS, replace=False)
                cross_combos = [list(all_cross_combos[i]) for i in indices]
                print(f"[EVAL] Cross-play: {len(all_cross_combos)} permutations → sampled {MAX_CROSS_COMBOS}")

            run_combinations = cross_combos + self_play_combos
            print(f"Run combinations: {len(run_combinations)} total ({len(cross_combos)} cross + {len(self_play_combos)} self)")

        # 특정 run을 고정 파트너로 쓰고 싶을 때
        if pairing_policy is not None:
            run_combinations = [
                [pairing_policy, i] for i in range(num_runs) if i != pairing_policy
            ]
            run_combinations += [
                [i, pairing_policy] for i in range(num_runs) if i != pairing_policy
            ]

        print("Run combinations: ", run_combinations)

        # 각 run의 지정된 ckpt(기본 ckpt_final)만 모아놓기
        _ckpt_key = ckpt_filter if ckpt_filter is not None else "ckpt_final"
        policy_pairings = [
            all_params[run_keys[i]][_ckpt_key] for i in range(num_runs)
        ]

        cross_combinations = {}
        for run_combination in run_combinations:
            run_combination = list(run_combination)

            # run_7 → "7" 이런 식으로 label 만들기
            run_ids = [run_keys[i].replace("run_", "") for i in run_combination]
            run_combination_key = "cross-" + "_".join(run_ids)

            # PolicyPairing(PPOParams_0, PPOParams_1, ...) 형태로 묶음
            policy_combination = PolicyPairing(
                *[policy_pairings[i] for i in run_combination]
            )

            cross_combinations[run_combination_key] = policy_combination

        # cross 모드 구조: {"cross": {comb_key: PolicyPairing}}
        all_params = {"cross": cross_combinations}

    else:
        # self-play 모드: 한 run의 PPOParams를 num_actors개로 복제해서 PolicyPairing 하나로 만들기
        # 구조: dict[run_id][ckpt_id] -> PolicyPairing
        all_params = jax.tree_util.tree_map(
            lambda x: PolicyPairing.from_single_policy(x, num_actors),
            all_params,
            is_leaf=lambda x: type(x) is PPOParams,
        )

    # JIT Cache
    jit_cache = {}
    results_structure = {}

    # Iterate over all pairings
    for first_level, first_level_runs in all_params.items():
        results_structure[first_level] = {}
        for second_level, pairing in first_level_runs.items():
            # Determine algs and configs
            if first_level == "cross":
                # Parse second_level (comb_key) e.g. "cross-0_1"
                run_indices = second_level.replace("cross-", "").split("_")
                
                # Map back to run_ids using run_keys
                # Note: run_keys are "run_0", "run_1" etc.
                # run_indices are "0", "1" etc.
                # We assume run_keys are sorted or consistent with indices used in run_combinations
                # run_combinations used range(num_runs), so indices correspond to run_keys index
                
                # But wait, run_keys is list(all_params.keys()) BEFORE modification.
                # We need to ensure run_keys is consistent.
                # run_keys was created before 'if cross:' block.
                
                # run_indices are indices into run_keys list?
                # In 'run_ids = [run_keys[i].replace("run_", "") for i in run_combination]'
                # Yes, run_combination contains indices into run_keys.
                # But 'run_indices' here comes from 'second_level' string which used 'run_ids'.
                # 'run_ids' are "0", "1" etc.
                # So we need to find run_key that has "run_0".
                
                current_run_ids = [f"run_{idx}" for idx in run_indices]
                current_configs = [_resolve_policy_config(configs[rid]) for rid in current_run_ids]
                
                # Get alg names
                current_algs = []
                for cfg in current_configs:
                    alg = cfg.get("ALG_NAME", "PPO")
                    if "alg" in cfg:
                        alg = cfg["alg"].get("ALG_NAME", alg)
                    ph1_enabled = bool(cfg.get("PH1_ENABLED", False))
                    if "alg" in cfg:
                        ph1_enabled = ph1_enabled or bool(cfg["alg"].get("PH1_ENABLED", False))
                    
                    # Check for STL (anchor)
                    # config 구조가 다양할 수 있으므로 여러 경로 확인
                    is_anchor = False
                    if "anchor" in cfg:
                        is_anchor = cfg["anchor"]
                    elif "alg" in cfg and "anchor" in cfg["alg"]:
                        is_anchor = cfg["alg"]["anchor"]
                    # model config 내부에 있을 수도 있음 (예: config.model.anchor)
                    elif "model" in cfg and "anchor" in cfg["model"]:
                        is_anchor = cfg["model"]["anchor"]
                    
                    if alg == "E3T" and is_anchor:
                        alg = "STL"

                    if ph1_enabled and "PH1" not in alg:
                        alg = f"PH1-{alg}"
                        
                    current_algs.append(alg)
                current_algs = tuple(current_algs)
                
            else:
                # Self-play
                # first_level is run_id
                run_id = first_level
                cfg = _resolve_policy_config(configs[run_id])
                alg = cfg.get("ALG_NAME", "PPO")
                if "alg" in cfg:
                    alg = cfg["alg"].get("ALG_NAME", alg)
                ph1_enabled = bool(cfg.get("PH1_ENABLED", False))
                if "alg" in cfg:
                    ph1_enabled = ph1_enabled or bool(cfg["alg"].get("PH1_ENABLED", False))
                
                # Check for STL (anchor)
                is_anchor = False
                if "anchor" in cfg:
                    is_anchor = cfg["anchor"]
                elif "alg" in cfg and "anchor" in cfg["alg"]:
                    is_anchor = cfg["alg"]["anchor"]
                elif "model" in cfg and "anchor" in cfg["model"]:
                    is_anchor = cfg["model"]["anchor"]
                
                if alg == "E3T" and is_anchor:
                    alg = "STL"

                if ph1_enabled and "PH1" not in alg:
                    alg = f"PH1-{alg}"
                
                current_configs = [cfg] * num_actors
                current_algs = tuple([alg] * num_actors)
            
            # Determine alg_arg for eval_pairing
            # Heuristic: keep PH1 flag if present, otherwise prefer E3T/STL.
            if any("PH1" in alg for alg in current_algs):
                if any("E3T" in alg for alg in current_algs):
                    alg_arg = "PH1-E3T"
                elif any("STL" in alg for alg in current_algs):
                    alg_arg = "PH1-STL"
                else:
                    alg_arg = "PH1"
            elif any("E3T" in alg for alg in current_algs):
                alg_arg = "E3T"
            elif any("STL" in alg for alg in current_algs):
                alg_arg = "STL"
            else:
                alg_arg = current_algs[0]

            # JIT Key: include alg tuple + behavior-affecting config values per agent
            # so that different config variants (e.g. different LEARNER_USE_BLOCKED_INPUT)
            # are compiled separately rather than sharing the first-compiled closure.
            config_fp = tuple(
                bool(cfg.get("LEARNER_USE_BLOCKED_INPUT", True))
                for cfg in current_configs
            )
            jit_key = current_algs + config_fp
            
            if jit_key not in jit_cache:
                print(f"Compiling JIT for pair: {jit_key}")

                is_simple_viz = not no_viz and not latent_analysis and not value_analysis

                if no_viz and not latent_analysis and not value_analysis:
                    # no_viz: JIT rollout으로 빠른 평가 (viz 불필요)
                    # eval_key를 인자로 받아서 여러 seed 평가 가능
                    def _rollout_impl_no_viz(pairing_params, eval_key):
                        policies = [
                            PPOPolicy(pairing_params[i].params, current_configs[i])
                            for i in range(num_actors)
                        ]
                        policy_pairing = PolicyPairing(*policies)
                        return get_rollout(
                            policy_pairing, env, eval_key, algorithm=alg_arg, use_jit=True,
                            eval_reward=eval_reward,
                        )
                    jit_cache[jit_key] = jax.jit(_rollout_impl_no_viz)
                elif is_simple_viz:
                    # viz: rollout만 JIT, 렌더링은 Python 루프에서 별도 처리
                    def _rollout_impl(pairing_params):
                        policies = [
                            PPOPolicy(pairing_params[i].params, current_configs[i])
                            for i in range(num_actors)
                        ]
                        policy_pairing = PolicyPairing(*policies)
                        return get_rollout(
                            policy_pairing,
                            env,
                            key,
                            algorithm=alg_arg,
                            use_jit=True,
                            eval_reward=eval_reward,
                        )
                    jit_cache[jit_key] = jax.jit(_rollout_impl)
                else:
                    # latent_analysis / value_analysis: Python 루프 필요, JIT 없음
                    def _viz_impl(pairing_params):
                        policies = [
                            PPOPolicy(pairing_params[i].params, current_configs[i])
                            for i in range(num_actors)
                        ]
                        policy_pairing = PolicyPairing(*policies)
                        _ekw = copy.deepcopy(env_kwargs)
                        _layout = _ekw.pop("layout", _env_name if (_env_name in ("ToyCoop", "GridSpread") or _env_name.startswith("MPE_")) else None)
                        return eval_pairing(
                            policy_pairing,
                            _layout,
                            key,
                            env_kwargs=_ekw,
                            num_seeds=num_seeds,
                            all_recipes=num_seeds is None,
                            no_viz=no_viz,
                            algorithm=alg_arg,
                            latent_analysis=latent_analysis,
                            value_analysis=value_analysis,
                            old_overcooked=False,
                            disable_old_overcooked_auto=True,
                        )
                    jit_cache[jit_key] = _viz_impl

            # Execute
            if no_viz:
                print(f"[EVAL] Running {first_level}/{second_level}")
                _jit_fn = jit_cache[jit_key]
                _n_eval = num_seeds or 1
                viz_result = {}
                for si in range(_n_eval):
                    eval_key = jax.random.PRNGKey(si)
                    r = _jit_fn(pairing, eval_key)
                    viz_result[f"seed-{si}"] = PolicyVizualization(
                        frame_seq=None,
                        total_reward=float(r.total_reward),
                        total_reward_combined=float(r.total_reward_combined) if r.total_reward_combined is not None else None,
                        prediction_accuracy=r.prediction_accuracy,
                        value_by_partner_pos=None,
                    )
            elif not latent_analysis and not value_analysis:
                print(f"[VIZ] Pairing: {first_level} / {second_level}")
                rollout = jit_cache[jit_key](pairing)
                # JIT 후 Python 루프에서 렌더링
                agent_view_size = env_kwargs.get("agent_view_size", None)
                frames = []
                num_steps = jax.tree_util.tree_leaves(rollout.state_seq)[0].shape[0]
                for t in range(num_steps):
                    state_t = jax.tree_util.tree_map(lambda x: x[t], rollout.state_seq)
                    frame = render_state_frame(state_t, engine_name, agent_view_size)
                    frames.append(np.array(frame))
                viz_result = {"seed-0": PolicyVizualization(
                    frame_seq=np.stack(frames),
                    total_reward=float(rollout.total_reward),
                    total_reward_combined=float(rollout.total_reward_combined) if rollout.total_reward_combined is not None else None,
                    prediction_accuracy=rollout.prediction_accuracy,
                    value_by_partner_pos=None,
                )}
            else:
                print(f"[VIZ] Pairing: {first_level} / {second_level}")
                viz_result = jit_cache[jit_key](pairing)
            results_structure[first_level][second_level] = viz_result
    
    # Update all_params with results
    all_params = results_structure

    labels = ["run", "checkpoint"]
    if cross:
        labels[1] = "policy_labels"

    # 6) reward summary + gif 저장
    rows_sparse = []
    rows_combined = []
    has_combined = False
    for first_level, first_level_runs in all_params.items():
        for second_level, second_level_runs in first_level_runs.items():
            sparse_sum = 0.0
            combined_sum = 0.0
            acc_sum = jnp.zeros(num_actors)
            acc_count = 0

            print(f"{labels[0]}: {first_level}, {labels[1]}: {second_level}")
            for annotation, viz in second_level_runs.items():
                frame_seq = viz.frame_seq
                total_reward = viz.total_reward
                total_reward_combined = getattr(viz, "total_reward_combined", None)
                pred_acc = viz.prediction_accuracy

                if not no_viz and frame_seq is not None:
                    viz_dir = run_base_dir / first_level / second_level
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_filename = viz_dir / f"{annotation}.gif"
                    imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

                sparse_sum += total_reward
                row_sparse = [first_level, second_level, annotation, total_reward]

                if total_reward_combined is not None:
                    has_combined = True
                    combined_sum += float(total_reward_combined)
                    row_combined = [first_level, second_level, annotation, float(total_reward_combined)]
                else:
                    combined_sum += total_reward
                    row_combined = [first_level, second_level, annotation, total_reward]

                if pred_acc is not None:
                    acc_sum += pred_acc
                    acc_count += 1
                    for i in range(pred_acc.shape[0]):
                        row_sparse.append(float(pred_acc[i]))
                        row_combined.append(float(pred_acc[i]))
                else:
                    for i in range(num_actors):
                        row_sparse.append(0.0)
                        row_combined.append(0.0)

                rows_sparse.append(row_sparse)
                rows_combined.append(row_combined)
                print(f"\t{annotation}:\t{total_reward}")

            n_runs = len(second_level_runs)
            sparse_mean = sparse_sum / n_runs
            combined_mean = combined_sum / n_runs
            print(f"\tMean reward (sparse):\t{sparse_mean}")

            mean_row_sparse = [first_level, second_level, "mean", sparse_mean]
            mean_row_combined = [first_level, second_level, "mean", combined_mean]
            if acc_count > 0:
                acc_mean = acc_sum / acc_count
                for i in range(acc_mean.shape[0]):
                    mean_row_sparse.append(float(acc_mean[i]))
                    mean_row_combined.append(float(acc_mean[i]))
            else:
                for i in range(num_actors):
                    mean_row_sparse.append(0.0)
                    mean_row_combined.append(0.0)

            rows_sparse.append(mean_row_sparse)
            rows_combined.append(mean_row_combined)

    # 7) CSV로 요약 저장 (viz 모드에서는 스킵)
    if no_viz:
        fieldnames = [labels[0], labels[1], "annotation", "total_reward"]
        for i in range(num_actors):
            fieldnames.append(f"pred_acc_agent_{i}")

        _suffix = f"__{ckpt_filter}" if ckpt_filter is not None else ""
        # sparse CSV (항상 저장)
        summery_name = f"reward_summary_cross{_suffix}.csv" if cross else f"reward_summary_sp{_suffix}.csv"
        summery_file = run_base_dir / summery_name
        with open(summery_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for row in rows_sparse:
                writer.writerow(row)
        print(f"Summary (sparse) written to {summery_file}")

        # combined CSV (GridSpread 등 combined_reward가 있을 때만)
        if has_combined:
            summery_name_c = f"reward_summary_cross_combined{_suffix}.csv" if cross else f"reward_summary_sp_combined{_suffix}.csv"
            summery_file_c = run_base_dir / summery_name_c
            with open(summery_file_c, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for row in rows_combined:
                    writer.writerow(row)
            print(f"Summary (combined) written to {summery_file_c}")

        print(f"Summary written to {summery_file}")

    # 7-1) value_analysis 결과: partner position별 value 평균 저장
    if value_analysis:
        value_rows = []
        agg_sum = {}
        agg_count = {}

        for first_level, first_level_runs in all_params.items():
            for second_level, second_level_runs in first_level_runs.items():
                for annotation, viz in second_level_runs.items():
                    if viz.value_by_partner_pos:
                        for row in viz.value_by_partner_pos:
                            value_rows.append([
                                first_level,
                                second_level,
                                annotation,
                                row[0],
                                row[1],
                                row[2],
                                row[3],
                            ])

                            key = (row[0], row[1])
                            agg_sum[key] = agg_sum.get(key, 0.0) + float(row[2]) * int(row[3])
                            agg_count[key] = agg_count.get(key, 0) + int(row[3])

        value_summary_file = run_base_dir / "value_by_partner_pos.csv"
        with open(value_summary_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    labels[0],
                    labels[1],
                    "annotation",
                    "e_t",
                    "partner_pos",
                    "mean_value",
                    "count",
                ]
            )
            for row in value_rows:
                writer.writerow(row)

        print(f"Value summary written to {value_summary_file}")

        mean_summary_file = run_base_dir / "value_by_partner_pos_mean.csv"
        with open(mean_summary_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "e_t",
                    "partner_pos",
                    "mean_value",
                    "count",
                ]
            )
            for key, total in sorted(agg_sum.items()):
                count = agg_count[key]
                mean_value = total / max(count, 1)
                writer.writerow([key[0], key[1], mean_value, count])

        print(f"Value mean summary written to {mean_summary_file}")

    # 7-1) latent_analysis 결과: partner position별 value 평균 저장
    if latent_analysis:
        value_rows = []
        for first_level, first_level_runs in all_params.items():
            for second_level, second_level_runs in first_level_runs.items():
                for annotation, viz in second_level_runs.items():
                    if viz.value_by_partner_pos:
                        for row in viz.value_by_partner_pos:
                            value_rows.append([
                                first_level,
                                second_level,
                                annotation,
                                row[0],
                                row[1],
                                row[2],
                                row[3],
                            ])

        value_summary_file = run_base_dir / "value_by_partner_pos.csv"
        with open(value_summary_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    labels[0],
                    labels[1],
                    "annotation",
                    "e_t",
                    "partner_pos",
                    "mean_value",
                    "count",
                ]
            )
            for row in value_rows:
                writer.writerow(row)

        print(f"Value summary written to {value_summary_file}")

    # 8) cross-play면 교차플레이 매트릭스도 그림 (no_viz일 때만, CSV 있을 때)
    if no_viz and cross:
        visualize_cross_play_matrix(summery_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int)
    parser.add_argument("--all_ckpt", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--per_ckpt_cross", action="store_true",
                        help="Run cross-play once per ckpt index found under eval/ subdir.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--pairing_policy", type=int)
    parser.add_argument("--latent_analysis", action="store_true")
    parser.add_argument("--value_analysis", action="store_true")
    parser.add_argument("--old_overcooked", action="store_true")
    parser.add_argument("--disable_old_overcooked_auto", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--cross_mode",
        type=str,
        default="full",
        choices=["full", "level_one"],
        help="Cross-play combination mode. full=all permutations, level_one=leave-one-out (3:1 + 1:3).",
    )
    parser.add_argument(
        "--eval_reward",
        type=str,
        default="sparse",
        choices=["sparse", "combined"],
        help="Eval reward mode. sparse=sparse only (GridSpread), combined=sparse+shaped.",
    )
    parser.add_argument(
        "--policy_source",
        type=str,
        default="auto",
        choices=["auto", "params", "ind", "spec"],
        help="Checkpoint branch selector. auto=ind for *_ph2* dirs, else params.",
    )

    args = parser.parse_args()

    directory = args.d
    num_seeds = args.num_seeds
    final_only = not args.all_ckpt
    cross = args.cross
    run_dir_name = Path(directory).name.lower()
    if args.policy_source == "auto":
        policy_source = "ind" if "ph2" in run_dir_name else "params"
    else:
        policy_source = args.policy_source
    print(f"[INFO] policy_source={policy_source}")

    key = jax.random.PRNGKey(args.seed)
    key_sp, key_cross = jax.random.split(key, 2)

    viz_mode = {
        "sp": (not cross) or args.all,
        "cross": cross or args.all,
    }
    modes = [m for m, v in viz_mode.items() if v]

    extra_env_kwargs = {}
    if args.no_reset:
        extra_env_kwargs["random_reset"] = False
        extra_env_kwargs["op_ingredient_permutations"] = False
    if args.max_steps is not None:
        extra_env_kwargs["max_steps"] = int(args.max_steps)

    if args.per_ckpt_cross:
        # eval/ 또는 top-level run_*/ 안에서 ckpt 이름들을 모아 각 ckpt 마다 cross-play 실행
        run_root = Path(directory)
        eval_sub = run_root / "eval"
        if eval_sub.is_dir() and any(p.is_dir() and "run_" in p.name for p in eval_sub.iterdir()):
            scan_root = eval_sub
        else:
            scan_root = run_root
        ckpt_names = set()
        for run_dir_p in scan_root.iterdir():
            if not run_dir_p.is_dir() or "run_" not in run_dir_p.name:
                continue
            for c in run_dir_p.iterdir():
                if c.is_dir() and c.name.startswith("ckpt_"):
                    ckpt_names.add(c.name)
        def _ckpt_sort_key(name):
            tail = name.split("_", 1)[1]
            try:
                return (0, int(tail))
            except ValueError:
                return (1, tail)
        sorted_ckpts = sorted(ckpt_names, key=_ckpt_sort_key)
        print(f"[per_ckpt_cross] Found {len(sorted_ckpts)} ckpts: {sorted_ckpts}")
        for ckpt_name in sorted_ckpts:
            print(f"\n========== per_ckpt_cross: {ckpt_name} ==========")
            visualize_ppo_policy(
                run_root,
                key_cross,
                num_seeds=num_seeds,
                final_only=False,
                cross=True,
                cross_mode=args.cross_mode,
                eval_reward=args.eval_reward,
                no_viz=args.no_viz,
                extra_env_kwargs=extra_env_kwargs,
                pairing_policy=args.pairing_policy,
                latent_analysis=args.latent_analysis,
                value_analysis=args.value_analysis,
                policy_source=policy_source,
                old_overcooked_override=args.old_overcooked,
                disable_old_overcooked_auto_override=args.disable_old_overcooked_auto,
                ckpt_filter=ckpt_name,
            )
        sys.exit(0)

    for mode in modes:
        fo = final_only or (mode == "cross")
        visualize_ppo_policy(
            Path(directory),
            key_sp if mode == "sp" else key_cross,
            num_seeds=num_seeds,
            final_only=fo,
            cross=mode == "cross",
            cross_mode=args.cross_mode,
            eval_reward=args.eval_reward,
            no_viz=args.no_viz,
            extra_env_kwargs=extra_env_kwargs,
            pairing_policy=args.pairing_policy,
            latent_analysis=args.latent_analysis,
            value_analysis=args.value_analysis,
            policy_source=policy_source,
            old_overcooked_override=args.old_overcooked,
            disable_old_overcooked_auto_override=args.disable_old_overcooked_auto,
        )
