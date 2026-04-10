"""Checkpoint loader lifted from cec-zero-shot/baselines/CEC/checkpoint_io.py.

Supports orbax directories and legacy pickle files. Only the load path is
exposed; saving is intentionally omitted (we never write CEC ckpts here).
"""
import os
import pickle
import types
from typing import Any, Dict, Tuple

import jax
import orbax.checkpoint as ocp


def _ensure_jax_tree_namespace() -> None:
    if hasattr(jax, "tree"):
        return
    jax.tree = types.SimpleNamespace(  # type: ignore[attr-defined]
        map=jax.tree_util.tree_map,
        leaves=jax.tree_util.tree_leaves,
        flatten=jax.tree_util.tree_flatten,
        structure=jax.tree_util.tree_structure,
        unflatten=jax.tree_util.tree_unflatten,
    )


_ensure_jax_tree_namespace()


def normalize_ckpt_format(fmt: str) -> str:
    value = str(fmt).strip().lower()
    if value in ("pkl", "pickle"):
        return "pkl"
    if value in ("orbax", "orbax_pytree"):
        return "orbax"
    raise ValueError(f"Unsupported checkpoint format: {fmt}")


def _load_orbax(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path, item=None)


def _load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_checkpoint(
    ckpt_path: str,
    prefer_format: str = "orbax",
    allow_fallback: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Load a CEC checkpoint.

    `ckpt_path` may point to either an orbax directory or a `.pkl` file. The
    sibling other-format file is tried as a fallback when allowed.
    """
    ckpt_path = os.path.abspath(ckpt_path)
    fmt = normalize_ckpt_format(prefer_format)

    # Derive sibling paths.
    if ckpt_path.endswith(".pkl"):
        pkl_path = ckpt_path
        orbax_dir = ckpt_path[:-4]
    else:
        orbax_dir = ckpt_path
        pkl_path = ckpt_path + ".pkl"

    if fmt == "orbax":
        if os.path.isdir(orbax_dir):
            return _load_orbax(orbax_dir), "orbax"
        if allow_fallback and os.path.isfile(pkl_path):
            return _load_pkl(pkl_path), "pkl"
    else:
        if os.path.isfile(pkl_path):
            return _load_pkl(pkl_path), "pkl"
        if allow_fallback and os.path.isdir(orbax_dir):
            return _load_orbax(orbax_dir), "orbax"

    raise FileNotFoundError(
        f"Checkpoint not found. orbax_dir={orbax_dir} legacy_pkl={pkl_path}"
    )
