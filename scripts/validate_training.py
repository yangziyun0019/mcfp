"""Evaluate trained checkpoints on prepared splits and write summary metrics.

This script provides a quick offline validation pass for trained position and orientation models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch

from mcfp.data.morph_spec_io import load_morphology_spec
from mcfp.models.encodings import ReferenceQuaternionEncoder, position_encoding
from mcfp.models.morph_encoder import MorphologyEncoder
from mcfp.models.orient_sdf import OrientationSDFModel
from mcfp.models.pos_sdf import PositionSDFModel
from mcfp.utils.config import load_config


# ===== Quick Settings (edit here) =====
POS_CONFIG = "configs/train_pos.yaml"
ORIENT_CONFIG = "configs/train_orient.yaml"

# If empty, auto use: <run_dir>/checkpoint_final.pt from the config.
POS_CKPT = ""
ORIENT_CKPT = ""

SPLIT = "val"  # train | val | test
RUN_POSITION = True
RUN_ORIENTATION = True

POS_SAMPLES = 10_00
ORIENT_SAMPLES = 10_00
INFER_BATCH_SIZE = 4096
SEED = 0

# Orientation inference temperature (same meaning as training/inference).
ORIENT_TAU = 0.05
# If None, follow train_orient.yaml loss.sign_thr.
ORIENT_SIGN_THR_OVERRIDE = None

# Relative error denominator floor.
REL_EPS = 1e-6

OUT_JSON = "outputs/val_metrics_fast.json"
# =====================================


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _split_id(name: str) -> int:
    key = str(name).strip().lower()
    if key in ("train", "0"):
        return 0
    if key in ("val", "valid", "validation", "1"):
        return 1
    if key in ("test", "2"):
        return 2
    raise ValueError(f"Unsupported split: {name}")


def _resolve_ckpt(cfg: Any, repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = _resolve_path(explicit, repo_root)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    run_dir = _get_path(cfg, "paths.run_dir", None)
    if run_dir:
        p = _resolve_path(Path(run_dir) / "checkpoint_final.pt", repo_root)
        if p.is_file():
            return p

    raise FileNotFoundError(
        "Checkpoint not found. Set POS_CKPT/ORIENT_CKPT (or --pos-ckpt/--orient-ckpt), "
        "or ensure paths.run_dir/checkpoint_final.pt exists."
    )


def _infer_pos_arch(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    layers = 0
    in_dim = None
    hidden_dim = None
    cond_dim = None
    for k, v in state.items():
        if k.startswith("decoder.layers.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 4:
                try:
                    layer_idx = int(parts[2])
                except Exception:
                    layer_idx = -1
                if layer_idx >= 0:
                    layers = max(layers, layer_idx + 1)
                    if layer_idx == 0:
                        in_dim = int(v.shape[1])
                        hidden_dim = int(v.shape[0])
        if k == "decoder.film.0.weight":
            cond_dim = int(v.shape[1])

    out: Dict[str, int] = {"num_layers": layers if layers > 0 else 5}
    if in_dim is not None:
        out["in_dim"] = in_dim
    if hidden_dim is not None:
        out["hidden_dim"] = hidden_dim
    if cond_dim is not None:
        out["cond_dim"] = cond_dim
    return out


def _infer_orient_arch(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    experts = 0
    layers = 0
    orient_in_dim = None
    hidden_dim = None
    context_dim = None
    cond_in_dim = None

    for k, v in state.items():
        if k.startswith("decoders.") and ".layers." in k and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 5:
                try:
                    dec_idx = int(parts[1])
                    layer_idx = int(parts[3])
                except Exception:
                    dec_idx = -1
                    layer_idx = -1
                if dec_idx >= 0 and layer_idx >= 0:
                    experts = max(experts, dec_idx + 1)
                    layers = max(layers, layer_idx + 1)
                    if dec_idx == 0 and layer_idx == 0:
                        orient_in_dim = int(v.shape[1])
                        hidden_dim = int(v.shape[0])
        if k == "context.2.weight":
            context_dim = int(v.shape[0])
        if k == "context.0.weight":
            cond_in_dim = int(v.shape[1])

    out: Dict[str, int] = {
        "experts": experts if experts > 0 else 4,
        "num_layers": layers if layers > 0 else 4,
    }
    if orient_in_dim is not None:
        out["orient_in_dim"] = orient_in_dim
    if hidden_dim is not None:
        out["hidden_dim"] = hidden_dim
    if context_dim is not None:
        out["context_dim"] = context_dim
    if cond_in_dim is not None:
        out["cond_in_dim"] = cond_in_dim
    return out


def _load_q_ref_from_h5(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if "/meta/q_ref" in h5:
            return np.asarray(h5["/meta/q_ref"], dtype=np.float32)
        if "/q_ref" in h5:
            return np.asarray(h5["/q_ref"], dtype=np.float32)
        if "/meta/ref_quat" in h5:
            return np.asarray(h5["/meta/ref_quat"], dtype=np.float32)
    return None


def _read_1d_by_indices(ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=ds.dtype)
    uniq, inv = np.unique(idx, return_inverse=True)
    vals = np.asarray(ds[uniq])
    return vals[inv]


def _lookup_sdf_sparse(ds_sdf: h5py.Dataset, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.empty((x.shape[0],), dtype=np.float32)
    x_i = x.astype(np.int64, copy=False)
    y_i = y.astype(np.int64, copy=False)
    z_i = z.astype(np.int64, copy=False)
    for xi in np.unique(x_i):
        mask = x_i == xi
        slab = np.asarray(ds_sdf[int(xi), :, :], dtype=np.float32)
        out[mask] = slab[y_i[mask], z_i[mask]]
    return out


def _normalize_orient_label(label: np.ndarray) -> np.ndarray:
    uniq = np.unique(label)
    if set(uniq.tolist()) <= {0, 1, 2}:
        # legacy: 1 reachable, 0 unreachable, 2 boundary
        return np.where(label == 0, -1, 1).astype(np.int8)
    if set(uniq.tolist()) <= {-1, 1}:
        return label.astype(np.int8)
    return np.where(label >= 0, 1, -1).astype(np.int8)


def _percent_error(pred: np.ndarray, target: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(np.abs(target), float(eps))
    return (np.abs(pred - target) / denom) * 100.0


def _load_position_stack(
    cfg_path: str,
    ckpt_override: str | None,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()

    pos_h5 = _resolve_path(_get_path(cfg, "paths.position_h5"), repo_root)
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    ckpt_path = _resolve_ckpt(cfg, repo_root, ckpt_override)

    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    k_p = int(_get_path(cfg, "model.position.k_p", 10))
    morph_cfg = _get_path(cfg, "model.morph", None)
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "morph_encoder" in ckpt:
        morph_encoder.load_state_dict(ckpt["morph_encoder"])

    pos_state = ckpt["pos_model"] if "pos_model" in ckpt else ckpt
    arch = _infer_pos_arch(pos_state)
    in_dim = int(arch.get("in_dim", 3 + 2 * 3 * k_p))
    expected_in_dim = 3 + 2 * 3 * k_p
    if in_dim != expected_in_dim:
        raise RuntimeError(
            f"Position input dim mismatch: ckpt={in_dim}, expected={expected_in_dim}. "
            "Check model.position.k_p."
        )

    pos_model = PositionSDFModel(
        in_dim=in_dim,
        hidden_dim=int(arch.get("hidden_dim", _get_path(cfg, "model.position.hidden_dim", 256))),
        num_layers=int(arch.get("num_layers", _get_path(cfg, "model.position.num_layers", 5))),
        w0_first=float(_get_path(cfg, "model.position.w0_first", 30.0)),
        w0=float(_get_path(cfg, "model.position.w0", 1.0)),
        cond_dim=int(arch.get("cond_dim", _get(morph_cfg, "d_model", 256))),
    ).to(device)
    pos_model.load_state_dict(pos_state)

    pos_model.eval()
    morph_encoder.eval()
    with torch.no_grad():
        morph_emb, _ = morph_encoder(morph_spec, device)

    trunc_k = float(_get_path(cfg, "loss.trunc_k", 40.0))

    return {
        "cfg": cfg,
        "repo_root": repo_root,
        "pos_h5": pos_h5,
        "model": pos_model,
        "morph_emb": morph_emb,
        "k_p": k_p,
        "l_ref": l_ref,
        "trunc_k": trunc_k,
        "ckpt_path": ckpt_path,
    }


def _load_orientation_stack(
    cfg_path: str,
    ckpt_override: str | None,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()

    orient_h5 = _resolve_path(_get_path(cfg, "paths.orientation_h5"), repo_root)
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    ckpt_path = _resolve_ckpt(cfg, repo_root, ckpt_override)

    q_ref = _load_q_ref_from_h5(orient_h5)
    if q_ref is None:
        raise RuntimeError(f"q_ref not found in orientation dataset: {orient_h5}")

    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    k_p = int(_get_path(cfg, "model.position.k_p", 10))
    morph_cfg = _get_path(cfg, "model.morph", None)
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)

    ref_encoder = ReferenceQuaternionEncoder(q_ref=torch.from_numpy(q_ref)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "morph_encoder" in ckpt:
        morph_encoder.load_state_dict(ckpt["morph_encoder"])

    orient_state = ckpt["orient_model"] if "orient_model" in ckpt else ckpt
    arch = _infer_orient_arch(orient_state)
    orient_in_dim = int(arch.get("orient_in_dim", 2 * int(q_ref.shape[0])))
    expected_orient_in_dim = 2 * int(q_ref.shape[0])
    if orient_in_dim != expected_orient_in_dim:
        raise RuntimeError(
            f"Orientation input dim mismatch: ckpt={orient_in_dim}, expected={expected_orient_in_dim}. "
            "Dataset q_ref and checkpoint may be inconsistent."
        )

    orient_model = OrientationSDFModel(
        orient_in_dim=orient_in_dim,
        context_dim=int(arch.get("context_dim", _get_path(cfg, "model.orientation.context_dim", 128))),
        hidden_dim=int(arch.get("hidden_dim", _get_path(cfg, "model.orientation.hidden_dim", 256))),
        num_layers=int(arch.get("num_layers", _get_path(cfg, "model.orientation.num_layers", 4))),
        experts=int(arch.get("experts", _get_path(cfg, "model.orientation.experts", 4))),
        cond_in_dim=int(
            arch.get(
                "cond_in_dim",
                int(_get(morph_cfg, "d_model", 256)) + (3 + 2 * 3 * k_p) + 1,
            )
        ),
    ).to(device)
    orient_model.load_state_dict(orient_state)

    orient_model.eval()
    morph_encoder.eval()
    with torch.no_grad():
        morph_emb, _ = morph_encoder(morph_spec, device)

    sign_thr = abs(float(_get_path(cfg, "loss.sign_thr", 0.35)))

    return {
        "cfg": cfg,
        "repo_root": repo_root,
        "orient_h5": orient_h5,
        "model": orient_model,
        "ref_encoder": ref_encoder,
        "morph_emb": morph_emb,
        "k_p": k_p,
        "l_ref": l_ref,
        "sign_thr": sign_thr,
        "ckpt_path": ckpt_path,
    }


def _sample_position_points(
    pos_h5: Path,
    split_id: int,
    sample_count: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    with h5py.File(pos_h5, "r") as h5:
        dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
        origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
        voxel_size = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])

        grp = h5[f"/index/split{split_id}"]
        nb_ids_ds = grp["nb_ids"]
        bd_ids_ds = grp["bd_ids"]

        n_nb_total = int(nb_ids_ds.shape[0])
        n_bd_total = int(bd_ids_ds.shape[0])
        n_total = n_nb_total + n_bd_total
        if n_total <= 0:
            raise RuntimeError("No position samples in the requested split")

        n_take = int(min(max(sample_count, 1), n_total))
        n_nb_take = int(round(n_take * (n_nb_total / n_total))) if n_nb_total > 0 else 0
        n_nb_take = min(n_nb_take, n_nb_total)
        n_bd_take = n_take - n_nb_take
        n_bd_take = min(n_bd_take, n_bd_total)

        if n_nb_take + n_bd_take <= 0:
            raise RuntimeError("No position samples selected")

        nb_sel = np.zeros((0,), dtype=np.int64)
        bd_sel = np.zeros((0,), dtype=np.int64)

        if n_nb_take > 0:
            idx = rng.choice(n_nb_total, size=n_nb_take, replace=False)
            nb_sel = _read_1d_by_indices(nb_ids_ds, idx).astype(np.int64)

        if n_bd_take > 0:
            idx = rng.choice(n_bd_total, size=n_bd_take, replace=False)
            bd_sel = _read_1d_by_indices(bd_ids_ds, idx).astype(np.int64)

        voxel_ids = np.concatenate([nb_sel, bd_sel], axis=0)
        source = np.concatenate(
            [
                np.ones((nb_sel.shape[0],), dtype=np.uint8),
                np.zeros((bd_sel.shape[0],), dtype=np.uint8),
            ],
            axis=0,
        )

        # shuffle once so source groups are mixed
        perm = rng.permutation(voxel_ids.shape[0])
        voxel_ids = voxel_ids[perm]
        source = source[perm]

        ny = int(dims[1])
        nz = int(dims[2])
        x = voxel_ids // (ny * nz)
        y = (voxel_ids % (ny * nz)) // nz
        z = voxel_ids % nz

        sdf = _lookup_sdf_sparse(h5["/grid/sdf"], x, y, z).astype(np.float32)

    p = origin + voxel_size * (np.stack([x, y, z], axis=1).astype(np.float64) + 0.5)

    return {
        "p": p.astype(np.float32),
        "sdf": sdf,
        "source": source,
        "voxel_size": np.asarray([voxel_size], dtype=np.float32),
    }


def _sample_orientation_points(
    orient_h5: Path,
    split_id: int,
    sample_count: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))

    with h5py.File(orient_h5, "r") as h5:
        anchor_pos = np.asarray(h5["/anchors/pos"], dtype=np.float32)
        if "/anchors/s_v_norm" in h5:
            anchor_s = np.asarray(h5["/anchors/s_v_norm"], dtype=np.float32)
        else:
            anchor_s = np.asarray(h5["/anchors/s_v"], dtype=np.float32)

        anchor_start = np.asarray(h5["/csr/anchor_start"], dtype=np.uint64)
        sample_index_ds = h5["/csr/sample_index"]

        if "/anchors/split" in h5:
            anchor_split = np.asarray(h5["/anchors/split"], dtype=np.uint8)
        else:
            anchor_split = None

        per_anchor = np.diff(anchor_start).astype(np.int64)
        valid = np.where(per_anchor > 0)[0]
        if anchor_split is not None:
            valid = valid[anchor_split[valid] == split_id]
        if valid.size == 0:
            raise RuntimeError("No orientation anchors in the requested split")

        counts = per_anchor[valid].astype(np.float64)
        probs = counts / float(np.sum(counts))

        n_take = int(max(sample_count, 1))
        chosen_anchor = rng.choice(valid, size=n_take, replace=True, p=probs)

        starts = anchor_start[chosen_anchor].astype(np.int64)
        lens = (anchor_start[chosen_anchor + 1] - anchor_start[chosen_anchor]).astype(np.int64)
        offsets = (rng.random(n_take) * lens).astype(np.int64)
        ptr = starts + offsets

        sample_ids = _read_1d_by_indices(sample_index_ds, ptr).astype(np.int64)

        uniq, inv = np.unique(sample_ids, return_inverse=True)
        quat_raw = np.asarray(h5["/samples/quat"][uniq], dtype=np.float32)[inv]
        phi_raw = np.asarray(h5["/samples/phi"][uniq], dtype=np.float32)[inv]

        method = None
        if "/samples/method" in h5:
            method = np.asarray(h5["/samples/method"][uniq], dtype=np.uint8)[inv]

        label = None
        if "/samples/label" in h5:
            label = np.asarray(h5["/samples/label"][uniq], dtype=np.int8)[inv]

    quat = quat_raw / np.clip(np.linalg.norm(quat_raw, axis=1, keepdims=True), 1e-9, None)

    if label is not None:
        lbl = _normalize_orient_label(label)
    else:
        lbl = np.where(phi_raw >= 0.0, 1, -1).astype(np.int8)

    phi = np.where(lbl > 0, np.abs(phi_raw), -np.abs(phi_raw)).astype(np.float32)
    if method is not None:
        phi = np.where(method == 2, 0.0, phi).astype(np.float32)

    p = anchor_pos[chosen_anchor].astype(np.float32)
    s = np.abs(anchor_s[chosen_anchor]).astype(np.float32)

    return {
        "p": p,
        "s": s,
        "quat": quat.astype(np.float32),
        "phi": phi,
        "label": lbl.astype(np.int8),
        "method": method.astype(np.uint8) if method is not None else np.full((p.shape[0],), 255, dtype=np.uint8),
    }


def validate_position_fast(
    cfg_path: str,
    ckpt_override: str | None,
    split: str,
    sample_count: int,
    infer_batch_size: int,
    seed: int,
    rel_eps: float,
    device: torch.device,
) -> Dict[str, Any]:
    stack = _load_position_stack(cfg_path, ckpt_override, device)

    split_id = _split_id(split)
    sampled = _sample_position_points(stack["pos_h5"], split_id=split_id, sample_count=sample_count, seed=seed)
    p = sampled["p"]
    s = sampled["sdf"]
    source = sampled["source"]
    voxel_size = float(sampled["voxel_size"][0])

    l_ref = float(stack["l_ref"])
    trunc_k = float(stack["trunc_k"])
    s_trunc = trunc_k * voxel_size
    y_target = np.clip(s, -s_trunc, s_trunc) / l_ref

    model: PositionSDFModel = stack["model"]
    morph_emb = stack["morph_emb"]
    k_p = int(stack["k_p"])

    pred = np.empty((p.shape[0],), dtype=np.float32)
    bs = max(1, int(infer_batch_size))

    with torch.no_grad():
        for i in range(0, p.shape[0], bs):
            j = min(p.shape[0], i + bs)
            p_t = torch.from_numpy(p[i:j]).to(device)
            e_p = position_encoding(p_t / l_ref, k_p=k_p)
            s_pred = model(e_p, morph_emb)
            pred[i:j] = s_pred.detach().cpu().numpy().reshape(-1).astype(np.float32)

    mask_nb = source == 1
    if not np.any(mask_nb):
        raise RuntimeError("No non-boundary position samples selected; cannot compute sign/value metrics.")

    sign_acc = float(np.mean((pred[mask_nb] >= 0.0) == (y_target[mask_nb] >= 0.0)))
    ape = _percent_error(pred[mask_nb], y_target[mask_nb], eps=rel_eps)
    mape = float(np.mean(ape))
    medape = float(np.median(ape))

    mae_norm = float(np.mean(np.abs(pred[mask_nb] - y_target[mask_nb])))
    mae_meter = float(mae_norm * l_ref)

    metrics: Dict[str, Any] = {
        "split": split,
        "split_id": split_id,
        "samples_total": int(p.shape[0]),
        "samples_non_boundary": int(np.sum(mask_nb)),
        "sign_acc_non_boundary": sign_acc,
        "mape_non_boundary_percent": mape,
        "median_ape_non_boundary_percent": medape,
        "mae_non_boundary_norm": mae_norm,
        "mae_non_boundary_meter": mae_meter,
        "voxel_size": voxel_size,
        "l_ref": l_ref,
        "checkpoint": str(stack["ckpt_path"]),
    }

    print(
        f"[pos][{split}] N={metrics['samples_total']} nb={metrics['samples_non_boundary']} "
        f"sign_acc={metrics['sign_acc_non_boundary']:.4f} "
        f"MAPE={metrics['mape_non_boundary_percent']:.2f}% "
        f"MAE={metrics['mae_non_boundary_meter']:.6f}m"
    )
    return metrics


def validate_orientation_fast(
    cfg_path: str,
    ckpt_override: str | None,
    split: str,
    sample_count: int,
    infer_batch_size: int,
    seed: int,
    rel_eps: float,
    tau: float,
    sign_thr_override: float | None,
    device: torch.device,
) -> Dict[str, Any]:
    stack = _load_orientation_stack(cfg_path, ckpt_override, device)

    split_id = _split_id(split)
    sampled = _sample_orientation_points(
        stack["orient_h5"],
        split_id=split_id,
        sample_count=sample_count,
        seed=seed,
    )

    p = sampled["p"]
    s = sampled["s"]
    quat = sampled["quat"]
    phi_target = sampled["phi"]
    label = sampled["label"]
    method = sampled["method"]

    model: OrientationSDFModel = stack["model"]
    ref_encoder: ReferenceQuaternionEncoder = stack["ref_encoder"]
    morph_emb = stack["morph_emb"]
    k_p = int(stack["k_p"])
    l_ref = float(stack["l_ref"])
    sign_thr = float(stack["sign_thr"])
    if sign_thr_override is not None:
        sign_thr = abs(float(sign_thr_override))

    pred = np.empty((p.shape[0],), dtype=np.float32)
    bs = max(1, int(infer_batch_size))

    with torch.no_grad():
        for i in range(0, p.shape[0], bs):
            j = min(p.shape[0], i + bs)

            p_t = torch.from_numpy(p[i:j]).to(device)
            s_t = torch.from_numpy(s[i:j]).to(device)
            q_t = torch.from_numpy(quat[i:j]).to(device)

            e_p = position_encoding(p_t / l_ref, k_p=k_p)
            s_in = torch.clamp(s_t, -1.0, 1.0).unsqueeze(-1)
            cond = torch.cat([morph_emb.expand(j - i, -1), e_p, s_in], dim=1)

            e_r = ref_encoder(q_t)
            _u_raw, u = model(e_r, cond, tau=float(tau))
            phi_raw = math.pi * u
            phi_pred = math.pi * torch.tanh(phi_raw / math.pi)
            pred[i:j] = phi_pred.detach().cpu().numpy().reshape(-1).astype(np.float32)

    mask_nb = method != 2
    if not np.any(mask_nb):
        raise RuntimeError("No non-boundary orientation samples selected; cannot compute sign/value metrics.")

    sign_acc = float(np.mean((pred[mask_nb] >= 0.0) == (label[mask_nb] > 0)))
    ape = _percent_error(pred[mask_nb], phi_target[mask_nb], eps=rel_eps)
    mape = float(np.mean(ape))
    medape = float(np.median(ape))

    mae_nb = float(np.mean(np.abs(pred[mask_nb] - phi_target[mask_nb])))

    # Aligned with training口径: focus on confidently outside/inside samples.
    mask_hard_out = mask_nb & (phi_target <= -sign_thr)
    mask_hard_in = mask_nb & (phi_target >= sign_thr)
    mask_value_eval = mask_nb & (np.abs(phi_target) >= sign_thr)

    hard_out_count = int(np.sum(mask_hard_out))
    hard_in_count = int(np.sum(mask_hard_in))
    value_eval_count = int(np.sum(mask_value_eval))

    hard_out_sign_acc = (
        float(np.mean((pred[mask_hard_out] >= 0.0) == (label[mask_hard_out] > 0)))
        if hard_out_count > 0
        else None
    )
    hard_in_sign_acc = (
        float(np.mean((pred[mask_hard_in] >= 0.0) == (label[mask_hard_in] > 0)))
        if hard_in_count > 0
        else None
    )
    outside_false_reachable_rate = (
        float(np.mean(pred[mask_hard_out] >= 0.0))
        if hard_out_count > 0
        else None
    )
    inside_false_unreachable_rate = (
        float(np.mean(pred[mask_hard_in] < 0.0))
        if hard_in_count > 0
        else None
    )
    mape_value_eval = (
        float(np.mean(_percent_error(pred[mask_value_eval], phi_target[mask_value_eval], eps=rel_eps)))
        if value_eval_count > 0
        else None
    )
    medape_value_eval = (
        float(np.median(_percent_error(pred[mask_value_eval], phi_target[mask_value_eval], eps=rel_eps)))
        if value_eval_count > 0
        else None
    )
    mae_value_eval = (
        float(np.mean(np.abs(pred[mask_value_eval] - phi_target[mask_value_eval])))
        if value_eval_count > 0
        else None
    )

    mask_b = method == 2
    bnd_mae = float(np.mean(np.abs(pred[mask_b] - phi_target[mask_b]))) if np.any(mask_b) else 0.0

    metrics: Dict[str, Any] = {
        "split": split,
        "split_id": split_id,
        "samples_total": int(p.shape[0]),
        "samples_non_boundary": int(np.sum(mask_nb)),
        "samples_boundary": int(np.sum(mask_b)),
        "sign_acc_non_boundary": sign_acc,
        "mape_non_boundary_percent": mape,
        "median_ape_non_boundary_percent": medape,
        "mae_non_boundary_rad": mae_nb,
        "boundary_mae_rad": bnd_mae,
        "sign_thr_rad": sign_thr,
        "hard_outside_count": hard_out_count,
        "hard_inside_count": hard_in_count,
        "value_eval_count": value_eval_count,
        "hard_outside_sign_acc": hard_out_sign_acc,
        "hard_inside_sign_acc": hard_in_sign_acc,
        "outside_false_reachable_rate": outside_false_reachable_rate,
        "inside_false_unreachable_rate": inside_false_unreachable_rate,
        "mape_value_eval_percent": mape_value_eval,
        "median_ape_value_eval_percent": medape_value_eval,
        "mae_value_eval_rad": mae_value_eval,
        "tau": float(tau),
        "checkpoint": str(stack["ckpt_path"]),
    }

    print(
        f"[orient][{split}] N={metrics['samples_total']} nb={metrics['samples_non_boundary']} "
        f"sign_acc={metrics['sign_acc_non_boundary']:.4f} "
        f"MAPE={metrics['mape_non_boundary_percent']:.2f}% "
        f"MAE(nb)={metrics['mae_non_boundary_rad']:.6f}rad "
        f"hard_out={metrics['hard_outside_count']} "
        f"hard_out_acc={metrics['hard_outside_sign_acc'] if metrics['hard_outside_sign_acc'] is not None else 'NA'} "
        f"ofr={metrics['outside_false_reachable_rate'] if metrics['outside_false_reachable_rate'] is not None else 'NA'}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast label-based validation: random samples from split (no eikonal, no train-batch sampling)."
    )
    parser.add_argument("--only", choices=["both", "pos", "orient"], default="both")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--pos-samples", type=int, default=None)
    parser.add_argument("--orient-samples", type=int, default=None)
    parser.add_argument("--infer-batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pos-ckpt", type=str, default=None)
    parser.add_argument("--orient-ckpt", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--orient-sign-thr", type=float, default=None)
    args = parser.parse_args()

    split = args.split if args.split is not None else SPLIT
    pos_samples = int(args.pos_samples if args.pos_samples is not None else POS_SAMPLES)
    orient_samples = int(args.orient_samples if args.orient_samples is not None else ORIENT_SAMPLES)
    infer_batch = int(args.infer_batch if args.infer_batch is not None else INFER_BATCH_SIZE)
    seed = int(args.seed if args.seed is not None else SEED)
    out_json = args.out_json if args.out_json is not None else OUT_JSON
    sign_thr_override = (
        float(args.orient_sign_thr) if args.orient_sign_thr is not None else ORIENT_SIGN_THR_OVERRIDE
    )

    pos_ckpt = args.pos_ckpt if args.pos_ckpt is not None else (POS_CKPT or None)
    orient_ckpt = args.orient_ckpt if args.orient_ckpt is not None else (ORIENT_CKPT or None)

    only = args.only
    run_pos = RUN_POSITION and only in ("both", "pos")
    run_orient = RUN_ORIENTATION and only in ("both", "orient")

    if not run_pos and not run_orient:
        raise RuntimeError("Both RUN_POSITION and RUN_ORIENTATION are disabled.")

    if args.device is not None:
        device = torch.device(str(args.device))
    else:
        # Keep default behavior consistent with your training configs.
        device = torch.device("cpu")

    results: Dict[str, Any] = {
        "config": {
            "split": split,
            "pos_samples": pos_samples,
            "orient_samples": orient_samples,
            "infer_batch": infer_batch,
            "seed": seed,
            "device": str(device),
            "rel_eps": REL_EPS,
            "orient_tau": ORIENT_TAU,
            "orient_sign_thr_override": sign_thr_override,
        }
    }

    if run_pos:
        results["position"] = validate_position_fast(
            cfg_path=POS_CONFIG,
            ckpt_override=pos_ckpt,
            split=split,
            sample_count=pos_samples,
            infer_batch_size=infer_batch,
            seed=seed,
            rel_eps=REL_EPS,
            device=device,
        )

    if run_orient:
        results["orientation"] = validate_orientation_fast(
            cfg_path=ORIENT_CONFIG,
            ckpt_override=orient_ckpt,
            split=split,
            sample_count=orient_samples,
            infer_batch_size=infer_batch,
            seed=seed + 17,
            rel_eps=REL_EPS,
            tau=ORIENT_TAU,
            sign_thr_override=sign_thr_override,
            device=device,
        )

    print("[validate-fast] summary")
    print(json.dumps(results, ensure_ascii=True, indent=2))

    out = Path(out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[validate-fast] wrote {out}")


if __name__ == "__main__":
    main()
