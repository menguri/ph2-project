#!/usr/bin/env python3
"""Pos-unified BC 학습 — 1 layout × N seeds.

데이터 소스: data/bc_unified/{layout}/ (prepare_unified_data.py 결과)
모델 저장: models_new/{layout}/pos_{0,1}/seed_{i}/checkpoint.pkl
  (pos-unified: 하나의 모델을 pos_0, pos_1 양쪽에 동일하게 저장 → 기존 eval 로더 호환)

사용법:
  python code/train_unified.py --layout cramped_room --num-seeds 3 --epochs 200
"""
import argparse
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np

# 기존 학습 루프/모델 재사용
from train import BCModel, make_train, save_checkpoint


def load_unified(data_dir: Path):
    obs = np.load(data_dir / "obs.npy").astype(np.float32) / 255.0
    acts = np.load(data_dir / "actions.npy").astype(np.int32)
    return {"input": jnp.array(obs), "target": jnp.array(acts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--data-dir", default="data/bc_unified")
    ap.add_argument("--model-dir", default="models_new")
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cnn-features", type=int, default=32)
    ap.add_argument("--fc-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=30)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute(): data_dir = here / data_dir
    model_root = Path(args.model_dir)
    if not model_root.is_absolute(): model_root = here / model_root

    data_layout = data_dir / args.layout
    if not (data_layout / "obs.npy").exists():
        print(f"[{args.layout}] 데이터 없음: {data_layout}")
        return 1

    print(f"[{args.layout}] 데이터 로딩: {data_layout}")
    data = load_unified(data_layout)
    print(f"  obs={data['input'].shape}, actions={data['target'].shape}")

    config = {
        "layout": args.layout,
        "position": -1,         # pos-unified 표식 (기존 per-pos 와 구분)
        "action_dim": 6,
        "cnn_features": args.cnn_features,
        "fc_dim": args.fc_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "adam_eps": args.adam_eps,
        "batch_size": args.batch_size,
        "num_seeds": args.num_seeds,
        "data_source": str(data_layout),
        "pos_unified": True,
    }

    key = jax.random.PRNGKey(args.seed)
    train_fn = jax.jit(make_train(config, data))
    keys = jax.random.split(key, args.num_seeds)
    all_params, all_metrics = jax.vmap(train_fn)(keys)

    for i in range(args.num_seeds):
        tr, va = jax.tree_util.tree_map(lambda x: x[i], all_metrics)
        print(f"  Seed {i}: train_acc={tr['accuracy'][-1]:.4f} "
              f"train_loss={tr['loss'][-1]:.4f} "
              f"val_acc={va['accuracy'][-1]:.4f} val_loss={va['loss'][-1]:.4f}")
        params_i = jax.tree_util.tree_map(lambda x: x[i], all_params)
        # pos_0, pos_1 양쪽에 동일 ckpt 저장 → 기존 eval 로더 호환.
        for pos in (0, 1):
            save_checkpoint(config, params_i,
                            model_root / args.layout / f"pos_{pos}" / f"seed_{i}")
    print(f"[{args.layout}] 저장: {model_root / args.layout}/pos_{{0,1}}/seed_*")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
