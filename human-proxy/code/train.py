#!/usr/bin/env python3
"""
BC (Behavioral Cloning) 모델 정의 + 데이터 로딩 + 훈련.

Position별로 학습, seed 1개.

사용법:
    python code/train.py --layout cramped_room --position 0
    python code/train.py --layout cramped_room --position 1

모델 저장:
    models/{layout}/pos_{position}/seed_{i}/
"""
import argparse
from pathlib import Path

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training import train_state


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────

class BCModel(nn.Module):
    """CNN 기반 Behavioral Cloning 모델. JaxMARL obs (H,W,C) → 6 action logits."""
    action_dim: int = 6
    cnn_features: int = 32
    fc_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) float32 [0,1]
        x = nn.Conv(
            self.cnn_features, (3, 3), padding="SAME",
            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.cnn_features, (3, 3), padding="SAME",
            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        # flatten spatial dims: (B, H*W*cnn_features)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            self.fc_dim,
            kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(x)
        return x  # (B, action_dim)


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

def load_data(data_dir: Path):
    """obs.npy + actions.npy 로드, float32 정규화."""
    obs = np.load(data_dir / "obs.npy").astype(np.float32) / 255.0
    actions = np.load(data_dir / "actions.npy").astype(np.int32)
    return {"input": jnp.array(obs), "target": jnp.array(actions)}


def load_and_merge_data(data_dirs):
    """여러 데이터 디렉토리의 obs/actions를 합쳐서 로드."""
    all_obs, all_actions = [], []
    for d in data_dirs:
        d = Path(d)
        obs_file = d / "obs.npy"
        if not obs_file.exists():
            continue
        obs = np.load(obs_file).astype(np.float32) / 255.0
        actions = np.load(d / "actions.npy").astype(np.int32)
        all_obs.append(obs)
        all_actions.append(actions)
        print(f"    {d}: {obs.shape[0]:,} samples")
    if not all_obs:
        return None
    obs_merged = jnp.array(np.concatenate(all_obs, axis=0))
    act_merged = jnp.array(np.concatenate(all_actions, axis=0))
    return {"input": obs_merged, "target": act_merged}


def split_data(data, key, val_split=0.10):
    """train/val 분리 (9:1 랜덤 split)."""
    N = data["input"].shape[0]
    num_val = int(N * val_split)
    perm = jax.random.permutation(key, N)
    val_idx, train_idx = perm[:num_val], perm[num_val:]

    def _split(d, idx):
        return jax.tree_util.tree_map(lambda x: x[idx], d)

    return _split(data, train_idx), _split(data, val_idx)


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def make_train(config, data):
    """jax.vmap 가능한 학습 함수 생성."""

    def train(key):
        key, split_key = jax.random.split(key)
        train_data, val_data = split_data(data, split_key)

        model = BCModel(
            action_dim=config["action_dim"],
            cnn_features=config["cnn_features"],
            fc_dim=config["fc_dim"],
        )

        num_train = train_data["input"].shape[0]
        batch_size = config["batch_size"]
        num_epochs = config["epochs"]

        # 모델 초기화
        input_shape = train_data["input"].shape[1:]  # (H, W, C)
        key, init_key = jax.random.split(key)
        dummy = jnp.zeros((1, *input_shape))
        params = model.init(init_key, dummy)["params"]

        tx = optax.adam(learning_rate=config["lr"], eps=config["adam_eps"])
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx,
        )

        def _batch(d, batch_size):
            n = d["input"].shape[0]
            nb = n // batch_size
            return jax.tree_util.tree_map(
                lambda x: x[:nb * batch_size].reshape((nb, batch_size, *x.shape[1:])),
                d,
            )

        def _train_step(state, batch):
            def _loss_fn(params):
                logits = state.apply_fn({"params": params}, batch["input"])
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=batch["target"],
                ).mean()
            grads = jax.grad(_loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, None

        def _eval_metrics(params, batched_data):
            """배치 데이터에 대한 loss/accuracy 계산."""
            def _step(carry, batch):
                total_loss, total_correct, total_count = carry
                logits = model.apply({"params": params}, batch["input"])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=batch["target"],
                ).mean()
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum(preds == batch["target"])
                count = batch["target"].shape[0]
                return (total_loss + loss, total_correct + correct, total_count + count), None

            (total_loss, total_correct, total_count), _ = jax.lax.scan(
                _step, (0.0, 0, 0), batched_data,
            )
            num_batches = jax.tree_util.tree_leaves(batched_data)[0].shape[0]
            return {
                "loss": total_loss / num_batches,
                "accuracy": total_correct / total_count,
            }

        def _epoch(carry, epoch_key):
            state = carry

            # 셔플 + 배치
            perm = jax.random.permutation(epoch_key, num_train)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], train_data)
            train_batched = _batch(shuffled, batch_size)
            val_batched = _batch(val_data, batch_size)

            # 학습
            state, _ = jax.lax.scan(_train_step, state, train_batched)

            # 메트릭
            train_metrics = _eval_metrics(state.params, train_batched)
            val_metrics = _eval_metrics(state.params, val_batched)

            return state, (train_metrics, val_metrics)

        epoch_keys = jax.random.split(key, num_epochs)
        state, all_metrics = jax.lax.scan(_epoch, state, epoch_keys)

        return state.params, all_metrics

    return train


# ──────────────────────────────────────────────
# 체크포인트 저장
# ──────────────────────────────────────────────

def save_checkpoint(config, params, model_dir: Path):
    """pickle로 체크포인트 저장. params는 numpy로 변환."""
    model_dir.mkdir(parents=True, exist_ok=True)
    # JAX array → numpy 변환 (직렬화용)
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    checkpoint = {"config": config, "params": params_np}
    with open(model_dir / "checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC 모델 훈련")
    parser.add_argument("--layout", required=True, help="레이아웃 이름")
    parser.add_argument("--position", type=int, required=True, help="포지션 (0 또는 1)")
    parser.add_argument("--num-seeds", type=int, default=1, help="시드 수")
    parser.add_argument("--epochs", type=int, default=200, help="에폭 수")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--cnn-features", type=int, default=32, help="CNN 채널 수")
    parser.add_argument("--fc-dim", type=int, default=64, help="FC 레이어 차원")
    parser.add_argument("--seed", type=int, default=30, help="기본 시드")
    parser.add_argument("--data-dir", nargs="+", default=["data/bc"],
                        help="데이터 디렉토리 (여러 개 지정 가능, 합산)")
    parser.add_argument("--model-dir", default="models", help="모델 저장 디렉토리")
    args = parser.parse_args()

    config = {
        "layout": args.layout,
        "position": args.position,
        "action_dim": 6,
        "cnn_features": args.cnn_features,
        "fc_dim": args.fc_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "adam_eps": args.adam_eps,
        "batch_size": args.batch_size,
        "num_seeds": args.num_seeds,
    }

    # 데이터 로드
    LAYOUT_ALIASES = {
        "cramped_room": ["cramped_room"],
        "asymm_advantages": ["asymm_advantages", "asymmetric_advantages"],
        "coord_ring": ["coord_ring", "coordination_ring"],
        "counter_circuit": ["counter_circuit"],
        "forced_coord": ["forced_coord", "forced_coordination"],
    }
    aliases = LAYOUT_ALIASES.get(args.layout, [args.layout])

    data_dirs = []
    for base in args.data_dir:
        for alias in aliases:
            d = Path(base) / alias / f"pos_{args.position}"
            if (d / "obs.npy").exists():
                data_dirs.append(d)

    if not data_dirs:
        print(f"데이터 없음: {args.data_dir} / {args.layout} / pos_{args.position}")
        return

    print(f"데이터 로딩 ({len(data_dirs)}개 소스 합산):")
    data = load_and_merge_data(data_dirs)
    if data is None:
        print("데이터 로드 실패")
        return
    print(f"  합산: obs={data['input'].shape}, actions={data['target'].shape}")

    # 학습
    key = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config, data)
    train_jit = jax.jit(train_fn)

    print(f"훈련 시작: {args.num_seeds} seeds × {args.epochs} epochs")
    train_keys = jax.random.split(key, args.num_seeds)
    all_params, all_metrics = jax.vmap(train_jit)(train_keys)

    # 결과 출력 + 체크포인트 저장
    for i in range(args.num_seeds):
        train_m, val_m = jax.tree_util.tree_map(lambda x: x[i], all_metrics)
        train_acc = train_m["accuracy"][-1].item()
        train_loss = train_m["loss"][-1].item()
        val_acc = val_m["accuracy"][-1].item()
        val_loss = val_m["loss"][-1].item()

        print(f"  Seed {i}: train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

        params_i = jax.tree_util.tree_map(lambda x: x[i], all_params)
        model_path = Path(args.model_dir) / args.layout / f"pos_{args.position}" / f"seed_{i}"
        save_checkpoint(config, params_i, model_path)
        print(f"    → 저장: {model_path}")

    print("훈련 완료!")


if __name__ == "__main__":
    main()
