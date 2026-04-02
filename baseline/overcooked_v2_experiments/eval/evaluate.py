import argparse
from collections import defaultdict
from typing import List
import jaxmarl
import sys
import os
import itertools
import jax.numpy as jnp
import jax
import numpy as np
import copy
from datetime import datetime
from pathlib import Path
import chex
import imageio
import csv

from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))

from .policy import AbstractPolicy, PolicyPairing
from .rollout import get_rollout
from .utils import (
    get_recipe_identifier,
    make_eval_env,
    render_state_frame,
)


@chex.dataclass
class PolicyVizualization:
    frame_seq: chex.Array
    total_reward: chex.Scalar
    prediction_accuracy: chex.Array = None


def visualize_pairing(
    output_dir: Path,
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    no_csv=False,
    algorithm="PPO",
):
    runs = eval_pairing(
        policies, layout_name, key, env_kwargs, num_seeds, all_recipes, no_viz, algorithm=algorithm
    )

    reward_sum = 0.0
    rows = []
    for annotation, viz in runs.items():
        frame_seq = viz.frame_seq
        total_reward = viz.total_reward
        pred_acc = viz.prediction_accuracy

        if not no_viz:
            viz_dir = output_dir / "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            viz_filename = viz_dir / f"{annotation}.gif"

            imageio.mimsave(viz_filename, frame_seq, "GIF", duration=0.5)

        reward_sum += total_reward
        row = [annotation, total_reward]
        if pred_acc is not None:
            # pred_acc is (num_agents,)
            # Add to row
            for i in range(pred_acc.shape[0]):
                row.append(float(pred_acc[i]))
        
        rows.append(row)
        print(f"\t{annotation}:\t{total_reward}")
    reward_mean = reward_sum / len(runs)
    print(f"\tMean reward:\t{reward_mean}")

    if not no_csv:
        summery_name = "reward_summary.csv"
        summery_file = output_dir / summery_name
        with open(summery_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            fieldnames = ["annotation", "total_reward"]
            # Add prediction accuracy columns if available
            if rows and len(rows[0]) > 2:
                num_agents = len(rows[0]) - 2
                for i in range(num_agents):
                    fieldnames.append(f"pred_acc_agent_{i}")

            writer.writerow(fieldnames)

            for row in rows:
                writer.writerow(row)
        print(f"Summary written to {summery_file}")


def eval_pairing(
    policies: PolicyPairing,
    layout_name,
    key,
    env_kwargs={},
    num_seeds=1,
    all_recipes=False,
    no_viz=False,
    algorithm="PPO",
    old_overcooked=False,
    disable_old_overcooked_auto=False,
    env=None,
):
    assert not (
        all_recipes and num_seeds is not None
    ), "Only one of all_recipes and num_seeds can be set"
    assert "layout" not in env_kwargs, "Layout should be passed as layout_name"

    # ToyCoop, MPE 감지: layout_name에서 env_name_override 설정
    if layout_name == "ToyCoop":
        _env_name_override = "ToyCoop"
    elif layout_name is not None and str(layout_name).startswith("MPE_"):
        _env_name_override = str(layout_name)
    else:
        _env_name_override = None

    if all_recipes:
        engine_name, _tmp_kwargs = None, None
        if old_overcooked:
            raise ValueError("all_recipes evaluation is not supported with overcooked(v1) engine")
        layout = overcooked_v2_layouts[layout_name]
        env_kwargs.pop("layout")

        possible_recipes = jnp.array(layout.possible_recipes)

        def _rollout_recipe(recipe):
            _layout = copy.deepcopy(layout)
            _layout.possible_recipes = [recipe]
            env = OvercookedV2(layout=_layout, **env_kwargs)

            rollout = get_rollout(policies, env, key, algorithm=algorithm)

            return rollout

        rollouts = jax.vmap(_rollout_recipe)(possible_recipes)
        annotations = [
            "recipe-" + get_recipe_identifier(r) for r in layout.possible_recipes
        ]

    else:
        if env is None:
            env, engine_name, _resolved_kwargs = make_eval_env(
                layout_name,
                env_kwargs,
                old_overcooked=old_overcooked,
                disable_auto=disable_old_overcooked_auto,
                env_name_override=_env_name_override,
            )
        else:
            engine_name = _env_name_override or "overcooked_v2"

        keys = jax.random.split(key, num_seeds)
        annotations = [f"seed-{i}" for i in range(num_seeds)]

        _eval_use_jit = True
        # DEBUG: params 구조 확인
        for pi, p in enumerate(policies):
            if hasattr(p, 'params'):
                pk = list(p.params.keys()) if isinstance(p.params, dict) else type(p.params)
                print(f"[EVAL-DEBUG] policy[{pi}].params keys: {pk}")

        if no_viz:
            if _eval_use_jit:
                # JIT + lax.scan으로 배치 실행 (CNN 전용, 기존 코드)
                def _rollout_seed_body(carry, key):
                    rollout = get_rollout(policies, env, key, algorithm=algorithm, use_jit=True)
                    return carry, rollout
                _, rollouts = jax.lax.scan(_rollout_seed_body, None, keys)
            else:
                # MLP encoder: Python loop + 개별 rollout (flax tracer 충돌 방지)
                rollout_list = []
                for ki in range(num_seeds):
                    r = get_rollout(policies, env, keys[ki], algorithm=algorithm, use_jit=False)
                    rollout_list.append(r)
                rollouts = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rollout_list)
        else:
            # viz 모드: seed 1개만 롤아웃, 렌더링은 Python 루프
            rollout_viz = get_rollout(policies, env, keys[0], algorithm=algorithm, use_jit=_eval_use_jit)
            rollouts = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), rollout_viz)
            annotations = ["seed-0"]


    if no_viz:
        frame_seqs = [None] * len(annotations)
    else:
        agent_view_size = env_kwargs.get("agent_view_size", None)

        # seed-0 한 개만 렌더링
        frame_seqs = [None] * len(annotations)
        state_seq_0 = jax.tree_util.tree_map(lambda x: x[0], rollouts.state_seq)
        frames = []
        num_steps = jax.tree_util.tree_leaves(state_seq_0)[0].shape[0]
        for t in range(num_steps):
            state_t = jax.tree_util.tree_map(lambda x: x[t], state_seq_0)
            render_engine = "overcooked_v2" if all_recipes else engine_name
            frame = render_state_frame(state_t, render_engine, agent_view_size)
            frames.append(np.array(frame))
        frame_seqs[0] = np.stack(frames)

    return {
        annotation: PolicyVizualization(
            frame_seq=frame_seqs[i],
            total_reward=rollouts.total_reward[i],
            prediction_accuracy=rollouts.prediction_accuracy[i] if rollouts.prediction_accuracy is not None else None
        )
        for i, annotation in enumerate(annotations)
    }
