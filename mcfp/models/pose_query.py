from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import torch

from mcfp.data.io import load_morph_spec
from mcfp.data.pose_features import compute_morph_meta, compute_pose_features, denormalize_delta
from mcfp.models.backbone import TokenFusionBackbone
from mcfp.models.heads import MultiIndicatorHeads
from mcfp.models.morph_encoder import MorphologyEncoderConfig, MorphologyEncoderGNN
from mcfp.models.morph_graph import GraphData, build_link_graph
from mcfp.models.pose_encoder import PoseEncoder
from mcfp.models.stage1 import MCFPStage1
from mcfp.utils import se3
from mcfp.sim.pose_validation import build_robot_and_self_checker, validate_pose_with_ik


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read config values with dotted keys from dict/SimpleNamespace."""
    cur: Any = cfg
    for part in key.split("."):
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
    """Resolve a path relative to repo_root if needed."""
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _load_label_keys(stats_path: Path, label_keys: List[str] | None) -> List[str]:
    """Load label keys from stats file or explicit list."""
    if label_keys:
        return [str(k) for k in label_keys]
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    keys = stats.get("label_keys", None)
    if not keys:
        raise ValueError(f"[pose_query] stats file missing label_keys: {stats_path}")
    return [str(k) for k in keys]


def _quat_from_rpy(rpy: np.ndarray, degrees: bool) -> np.ndarray:
    """Convert roll-pitch-yaw (xyz) to quaternion [x, y, z, w]."""
    angles = np.asarray(rpy, dtype=np.float64).reshape(3)
    if degrees:
        angles = np.deg2rad(angles)
    roll, pitch, yaw = angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _parse_query_pose(cfg: Any) -> np.ndarray:
    """Parse pose from config in quaternion or RPY mode."""
    mode = str(_get(cfg, "query.pose_mode", "quat")).lower()
    pose_list = _get(cfg, "query.pose")
    if pose_list is None:
        raise ValueError("[pose_query] query.pose is required.")
    pose_arr = np.asarray(pose_list, dtype=np.float32).reshape(-1)
    if mode == "rpy":
        if pose_arr.size != 6:
            raise ValueError("[pose_query] query.pose must be [x,y,z,roll,pitch,yaw] for rpy mode.")
        pos = pose_arr[:3]
        rpy = pose_arr[3:]
        degrees = bool(_get(cfg, "query.rpy_degrees", True))
        quat = _quat_from_rpy(rpy, degrees=degrees)
        return np.concatenate([pos, quat], axis=0).astype(np.float32)
    if pose_arr.size != 7:
        raise ValueError("[pose_query] query.pose must be [x,y,z,qx,qy,qz,qw] for quat mode.")
    quat = pose_arr[3:7]
    norm = float(np.linalg.norm(quat))
    if norm > 1e-8:
        pose_arr[3:7] = quat / norm
    return pose_arr.astype(np.float32)


def _pose_error(target_pose: np.ndarray, solved_pose: np.ndarray) -> Tuple[float, float]:
    """Compute position/rotation error between two poses."""
    pos_err = float(np.linalg.norm(solved_pose[:3] - target_pose[:3]))
    ori_err = float(np.linalg.norm(se3.quat_delta_axis_angle(target_pose[3:], solved_pose[3:])))
    return pos_err, ori_err


def _collect_limit_violations(
    q: np.ndarray,
    limits: np.ndarray,
    names: List[str],
    eps: float,
) -> List[str]:
    """Collect joint limit violations for logging."""
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if limits.shape[0] == 0:
        return []
    n = min(q_arr.shape[0], limits.shape[0])
    violations: List[str] = []
    for i in range(n):
        low, high = float(limits[i, 0]), float(limits[i, 1])
        qi = float(q_arr[i])
        if qi < (low - eps) or qi > (high + eps):
            name = names[i] if i < len(names) else f"joint_{i}"
            violations.append(f"{name}: q={qi:.6f} lim=[{low:.6f},{high:.6f}]")
    return violations


def _build_heads_cfg_from_keys(
    label_keys: List[str],
    ws_name: str,
    ws_with_logits: bool,
    heads_cfg: Any,
) -> Dict[str, Any]:
    """Build head config list from label keys."""
    head_list = list(_get(heads_cfg, "heads", [])) if heads_cfg is not None else []
    if len(head_list) > 0:
        return {"heads": head_list}

    d_hidden = int(_get(heads_cfg, "default_hidden_dim", 256))
    n_layers = int(_get(heads_cfg, "default_num_layers", 2))
    drop = float(_get(heads_cfg, "default_dropout", 0.0))
    act_reg = str(_get(heads_cfg, "reg_out_activation", "identity"))
    act_ws = str(_get(heads_cfg, "ws_out_activation", "identity"))
    act_default = str(_get(heads_cfg, "default_out_activation", "identity"))

    built = []
    for name in label_keys:
        name = str(name)
        if name == ws_name:
            out_act = act_ws if ws_with_logits else act_default
        else:
            out_act = act_reg
        built.append(
            {
                "name": name,
                "hidden_dim": d_hidden,
                "num_layers": n_layers,
                "dropout": drop,
                "out_activation": out_act,
            }
        )
    return {"heads": built}


def _graph_to_device(g: GraphData, device: torch.device) -> GraphData:
    """Move graph tensors to target device."""
    return GraphData(
        x=g.x.to(device=device),
        edge_index=g.edge_index.to(device=device),
        edge_attr=(None if g.edge_attr is None else g.edge_attr.to(device=device)),
        batch=(None if g.batch is None else g.batch.to(device=device)),
        node_names=g.node_names,
        meta=g.meta,
    )


def _make_quiet_logger(name: str) -> logging.Logger:
    """Build a quiet logger to suppress verbose validation output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@dataclass(frozen=True)
class PoseQueryResult:
    """Container for pose query outputs."""
    g_ws: float
    reachable: bool
    delta_pos: np.ndarray
    delta_rot: np.ndarray
    adjusted_pose: np.ndarray


def build_stage1_model(
    train_cfg: Any,
    label_keys: List[str],
    morph_spec: Dict[str, Any],
    pose_dim: int,
    device: torch.device,
) -> MCFPStage1:
    """Build Stage-1 model to match training configuration."""
    g0 = build_link_graph(
        morph_spec,
        device=None,
        dtype=torch.float32,
        bidirectional=bool(_get(train_cfg, "model.morph_graph.bidirectional", True)),
        use_link_index_feature=bool(_get(train_cfg, "model.morph_graph.use_link_index_feature", True)),
    )
    node_in_dim = int(g0.x.shape[1])
    edge_in_dim = int(g0.edge_attr.shape[1]) if g0.edge_attr is not None else 0

    d_model = int(_get(train_cfg, "model.d_model", 256))
    me_cfg = MorphologyEncoderConfig(
        input_dim=node_in_dim,
        hidden_dim=d_model,
        num_layers=int(_get(train_cfg, "model.morph_encoder.num_layers", 3)),
        edge_dim=edge_in_dim,
        dropout=float(_get(train_cfg, "model.morph_encoder.dropout", 0.0)),
        use_layernorm=bool(_get(train_cfg, "model.morph_encoder.use_layernorm", True)),
    )
    morph_encoder = MorphologyEncoderGNN(me_cfg)

    pe_cfg = {
        "pose_dim": int(pose_dim),
        "emb_dim": d_model,
        "num_bands": int(_get(train_cfg, "model.pose_encoder.num_bands", 10)),
        "mlp_hidden": int(_get(train_cfg, "model.pose_encoder.mlp_hidden", 256)),
        "mlp_layers": int(_get(train_cfg, "model.pose_encoder.mlp_layers", 3)),
        "dropout": float(_get(train_cfg, "model.pose_encoder.dropout", 0.0)),
        "include_xyz_raw": bool(_get(train_cfg, "model.pose_encoder.include_xyz_raw", True)),
    }
    pose_encoder = PoseEncoder.from_cfg(pe_cfg)

    bb_cfg = {
        "d_model": d_model,
        "nhead": int(_get(train_cfg, "model.backbone.nhead", 8)),
        "num_layers": int(_get(train_cfg, "model.backbone.num_layers", 6)),
        "dim_feedforward": int(_get(train_cfg, "model.backbone.dim_feedforward", 1024)),
        "dropout": float(_get(train_cfg, "model.backbone.dropout", 0.1)),
        "max_nodes": _get(train_cfg, "model.backbone.max_nodes", None),
    }
    backbone = TokenFusionBackbone.from_cfg(bb_cfg)

    ws_name = str(_get(train_cfg, "loss.ws_name", "g_ws"))
    ws_with_logits = bool(_get(train_cfg, "loss.ws_with_logits", True))
    heads_cfg = _build_heads_cfg_from_keys(
        label_keys=label_keys,
        ws_name=ws_name,
        ws_with_logits=ws_with_logits,
        heads_cfg=_get(train_cfg, "model.heads", {}) or {},
    )
    heads = MultiIndicatorHeads.from_cfg(in_dim=d_model, cfg=heads_cfg)

    model = MCFPStage1(
        morph_encoder=morph_encoder,
        pose_encoder=pose_encoder,
        backbone=backbone,
        heads=heads,
    ).to(device=device)
    return model


def run_pose_query(cfg: Any, logger) -> PoseQueryResult:
    """Run a single pose query using a trained Stage-1 model."""
    repo_root = Path(_get(cfg, "paths.repo_root", ".")).resolve()
    train_cfg_path = _resolve_path(_get(cfg, "paths.train_config"), repo_root)
    checkpoint_path = _resolve_path(_get(cfg, "paths.checkpoint"), repo_root)
    morph_path = _resolve_path(_get(cfg, "paths.morph_json"), repo_root)

    from mcfp.utils.config import load_config

    train_cfg = load_config(train_cfg_path)
    stats_path = _resolve_path(_get(train_cfg, "paths.stats"), Path(_get(train_cfg, "paths.repo_root", repo_root)))
    label_keys = _load_label_keys(stats_path, list(_get(cfg, "data.label_keys", [])) or [])

    device_str = str(_get(cfg, "run.device", "cpu"))
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    morph_spec = load_morph_spec(morph_path)

    pose = _parse_query_pose(cfg).reshape(7)
    pose_cfg = train_cfg.data.pose_features
    delta_cfg = train_cfg.data.delta_norm

    morph_scale, aabb_min, aabb_max = compute_morph_meta(morph_spec)
    pose_feats = compute_pose_features(
        pose=pose,
        pose_cfg=pose_cfg,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        morph_scale=morph_scale,
    )
    pose_dim = int(pose_feats.shape[0])

    model = build_stage1_model(train_cfg, label_keys, morph_spec, pose_dim=pose_dim, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    graph = build_link_graph(
        morph_spec,
        device=None,
        dtype=torch.float32,
        bidirectional=bool(_get(train_cfg, "model.morph_graph.bidirectional", True)),
        use_link_index_feature=bool(_get(train_cfg, "model.morph_graph.use_link_index_feature", True)),
    )
    graph = _graph_to_device(graph, device)

    with torch.no_grad():
        batch = {
            "pose_feats": torch.as_tensor(pose_feats[None, :], dtype=torch.float32, device=device),
            "morph_graph": graph,
        }
        out = model(batch)
        preds = out.preds

    ws_name = str(_get(train_cfg, "loss.ws_name", "g_ws"))
    ws_with_logits = bool(_get(train_cfg, "loss.ws_with_logits", True))
    gws_raw = preds[ws_name].view(-1)[0].item()
    gws = float(torch.sigmoid(torch.tensor(gws_raw)).item()) if ws_with_logits else float(gws_raw)

    dp_norm = np.array(
        [
            float(preds.get("delta_pos_x").view(-1)[0].item()),
            float(preds.get("delta_pos_y").view(-1)[0].item()),
            float(preds.get("delta_pos_z").view(-1)[0].item()),
        ],
        dtype=np.float32,
    )
    dr_norm = np.array(
        [
            float(preds.get("delta_rot_x").view(-1)[0].item()),
            float(preds.get("delta_rot_y").view(-1)[0].item()),
            float(preds.get("delta_rot_z").view(-1)[0].item()),
        ],
        dtype=np.float32,
    )

    dp_real, dr_real = denormalize_delta(
        delta_pos_norm=dp_norm,
        delta_rot_norm=dr_norm,
        delta_cfg=delta_cfg,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        morph_scale=morph_scale,
    )

    threshold = float(_get(cfg, "query.gws_threshold", 0.8))
    reachable = gws >= threshold
    if reachable:
        adjusted_pose = pose.copy()
    else:
        pos_new = pose[:3] + dp_real
        quat_new = se3.apply_delta_quat(pose[3:], dr_real)
        adjusted_pose = np.concatenate([pos_new, quat_new], axis=0).astype(np.float32)

    logger.info(f"[pose_query] g_ws={gws:.6f} threshold={threshold:.3f} reachable={reachable}")
    if not reachable:
        logger.info(f"[pose_query] delta_pos={dp_real.tolist()} delta_rot={dr_real.tolist()}")
        logger.info(f"[pose_query] adjusted_pose={adjusted_pose.tolist()}")

    if bool(_get(cfg, "validation.enable", False)):
        verbose = bool(_get(cfg, "validation.verbose", False))
        val_cfg_path = _resolve_path(_get(cfg, "validation.data_gen_config"), repo_root)
        from mcfp.utils.config import load_config as _load_cfg

        val_cfg = _load_cfg(val_cfg_path)
        limits_cfg = _get(val_cfg, "limits", {}) or {}
        limit_eps = float(_get(limits_cfg, "eps", 1e-6))
        base_link = morph_spec.get("meta", {}).get("base_link", None)
        ee_link = morph_spec.get("meta", {}).get("ee_link", None)
        urdf_path = Path(morph_spec.get("meta", {}).get("urdf_path", ""))
        if not urdf_path.is_absolute():
            urdf_path = (repo_root / urdf_path).resolve()

        quiet_logger = _make_quiet_logger(f"{logger.name}.validation")
        robot, self_checker = build_robot_and_self_checker(
            urdf_path=urdf_path,
            base_link=base_link,
            end_effector_link=ee_link,
            cfg=val_cfg,
            logger=quiet_logger,
        )

        orig_val = validate_pose_with_ik(pose[:3], pose[3:], robot, self_checker, val_cfg)
        orig_reach = bool(orig_val["label"] > 0.5)
        logger.info(f"[pose_query] validate_original reachable={orig_reach}")
        if verbose:
            msg = (
                "[pose_query] original "
                f"ik={orig_val['ik_success']} limits={orig_val['within_limits']} "
                f"self_ok={orig_val['self_collision_free']} label={orig_val['label']}"
            )
            if orig_val.get("ik_success", False):
                pos_err, ori_err = _pose_error(pose, orig_val["pose"])
                msg += f" pos_err={pos_err:.6f} ori_err_deg={np.rad2deg(ori_err):.3f}"
            logger.info(msg)
            if orig_val.get("q", None) is not None:
                q_arr = np.asarray(orig_val["q"], dtype=np.float32).reshape(-1)
                logger.info(f"[pose_query] original q={q_arr.tolist()}")
                violations = _collect_limit_violations(
                    q=q_arr,
                    limits=robot.joint_limits,
                    names=robot.joint_names,
                    eps=limit_eps,
                )
                if violations:
                    logger.info(f"[pose_query] original limit_violations={violations}")

        adj_val = validate_pose_with_ik(adjusted_pose[:3], adjusted_pose[3:], robot, self_checker, val_cfg)
        adj_reach = bool(adj_val["label"] > 0.5)
        logger.info(f"[pose_query] validate_adjusted reachable={adj_reach}")
        if verbose:
            msg = (
                "[pose_query] adjusted "
                f"ik={adj_val['ik_success']} limits={adj_val['within_limits']} "
                f"self_ok={adj_val['self_collision_free']} label={adj_val['label']}"
            )
            if adj_val.get("ik_success", False):
                pos_err, ori_err = _pose_error(adjusted_pose, adj_val["pose"])
                msg += f" pos_err={pos_err:.6f} ori_err_deg={np.rad2deg(ori_err):.3f}"
            logger.info(msg)
            if adj_val.get("q", None) is not None:
                q_arr = np.asarray(adj_val["q"], dtype=np.float32).reshape(-1)
                logger.info(f"[pose_query] adjusted q={q_arr.tolist()}")
                violations = _collect_limit_violations(
                    q=q_arr,
                    limits=robot.joint_limits,
                    names=robot.joint_names,
                    eps=limit_eps,
                )
                if violations:
                    logger.info(f"[pose_query] adjusted limit_violations={violations}")

    return PoseQueryResult(
        g_ws=gws,
        reachable=reachable,
        delta_pos=dp_real,
        delta_rot=dr_real,
        adjusted_pose=adjusted_pose,
    )
