from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from mcfp.data.collate import PoseBatchCollator
from mcfp.data.datasets import DeltaFeatureConfig, PoseDeltaDataset, PoseFeatureConfig
from mcfp.data.io import read_jsonl, load_pose_samples
from mcfp.data.sampling import BalancedSamplingConfig, GroupedBalancedPoseBatchSampler
from mcfp.models.morph_graph import GraphData, build_link_graph
from mcfp.models.morph_encoder import MorphologyEncoderConfig, MorphologyEncoderGNN
from mcfp.models.pose_encoder import PoseEncoder
from mcfp.models.backbone import TokenFusionBackbone
from mcfp.models.heads import MultiIndicatorHeads
from mcfp.models.stage1 import MCFPStage1
from mcfp.train.losses import MultiTaskLoss
from mcfp.train.metrics import Stage1Metrics
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.utils.seed import set_seed, seed_worker


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


def _read_lines(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                lines.append(s)
    return lines


def _load_label_keys(cfg: Any, stats_path: Path) -> List[str]:
    user_keys = list(_get(cfg, "data.label_keys", [])) or []
    if len(user_keys) > 0:
        return [str(k) for k in user_keys]

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    keys = stats.get("label_keys", None)
    if not keys:
        raise ValueError(f"[train_stage1] stats file missing label_keys: {stats_path}")
    return [str(k) for k in keys]


def _split_indices_by_gws(
    labels: np.ndarray,
    val_ratio: float,
    rng: np.random.RandomState,
    stratify: bool,
) -> tuple[list[int], list[int]]:
    """Split indices into train/val with optional g_ws stratification."""
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    n = int(labels.shape[0])
    if n == 0:
        return [], []
    if val_ratio <= 0.0:
        return list(range(n)), []

    if not stratify:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(round(n * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx.tolist(), val_idx.tolist()

    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_val_pos = int(round(len(pos_idx) * val_ratio))
    n_val_neg = int(round(len(neg_idx) * val_ratio))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx.tolist(), val_idx.tolist()


def _make_pose_cfg(cfg: Any) -> PoseFeatureConfig:
    pf = _get(cfg, "data.pose_features", {}) or {}
    return PoseFeatureConfig(
        primary_pos=str(_get(pf, "primary_pos", "aabb_centered")),
        include_aabb_ratio=bool(_get(pf, "include_aabb_ratio", True)),
        include_aabb_centered=bool(_get(pf, "include_aabb_centered", True)),
        include_morph_scale=bool(_get(pf, "include_morph_scale", True)),
        include_raw_pos=bool(_get(pf, "include_raw_pos", False)),
        include_quat=bool(_get(pf, "include_quat", True)),
        quat_normalize=bool(_get(pf, "quat_normalize", False)),
        eps=float(_get(pf, "eps", 1e-8)),
    )


def _make_delta_cfg(cfg: Any) -> DeltaFeatureConfig:
    dc = _get(cfg, "data.delta_norm", {}) or {}
    return DeltaFeatureConfig(
        pos_norm=str(_get(dc, "pos", "aabb")),
        rot_norm=str(_get(dc, "rot", "pi")),
        eps=float(_get(dc, "eps", 1e-8)),
    )


def _graph_to_device(g: GraphData, device: torch.device) -> GraphData:
    return GraphData(
        x=g.x.to(device=device),
        edge_index=g.edge_index.to(device=device),
        edge_attr=(None if g.edge_attr is None else g.edge_attr.to(device=device)),
        batch=(None if g.batch is None else g.batch.to(device=device)),
        node_names=g.node_names,
        meta=g.meta,
    )


def _labels_tensor_to_dict(labels: torch.Tensor, label_keys: List[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for j, k in enumerate(label_keys):
        out[str(k)] = labels[:, j]
    return out


def _build_heads_cfg_from_keys(
    label_keys: List[str],
    ws_name: str,
    ws_with_logits: bool,
    heads_cfg: Dict[str, Any],
) -> Dict[str, Any]:
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


def _append_metrics_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """Append a row to metrics.csv, creating it with headers if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    row_aligned = {k: row.get(k, "") for k in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row_aligned)


def _format_eta(elapsed_s: float, steps_done: int, steps_total: int) -> str:
    if steps_done <= 0:
        return "eta=?"
    rate = elapsed_s / max(1, steps_done)
    remain = max(0, steps_total - steps_done) * rate
    mins = int(remain // 60)
    secs = int(remain % 60)
    return f"eta={mins:02d}m{secs:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="Run a short smoke test.")
    ap.add_argument("--smoke_steps", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger(name="mcfp.train.stage1", log_dir=_get(cfg, "logging.log_dir", "logs"))

    repo_root = Path(_get(cfg, "paths.repo_root")).resolve()
    run_dir = (repo_root / _get(cfg, "run.run_dir")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "metrics.csv"

    seed = int(_get(cfg, "run.seed", 42))
    set_seed(seed, deterministic=True)

    device_str = str(_get(cfg, "run.device", "cpu"))
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"[env] device={device}")

    manifest_path = Path(_get(cfg, "paths.manifest")).resolve()
    splits_dir = Path(_get(cfg, "paths.splits_dir")).resolve()
    stats_path = Path(_get(cfg, "paths.stats")).resolve()

    label_keys = _load_label_keys(cfg, stats_path)
    logger.info(f"[data] label_keys={label_keys}")

    manifest_records = read_jsonl(manifest_path)
    manifest_by_id = {str(r["variant_id"]): r for r in manifest_records}

    train_ids = _read_lines(splits_dir / "stage1_train.txt")
    val_ids = _read_lines(splits_dir / "stage1_val.txt")
    if len(train_ids) == 0:
        raise ValueError("[train_stage1] stage1_train.txt is empty.")

    split_cfg = _get(cfg, "data.split", {}) or {}
    split_mode = str(_get(split_cfg, "mode", "within_morph")).lower()
    val_ratio = float(_get(split_cfg, "val_ratio", 0.1))
    stratify_by_gws = bool(_get(split_cfg, "stratify_by_gws", True))
    split_seed = int(_get(split_cfg, "seed", seed))

    if split_mode not in ("within_morph", "by_variant"):
        raise ValueError(f"[train_stage1] Unsupported split mode: {split_mode}")

    pose_cfg = _make_pose_cfg(cfg)
    delta_cfg = _make_delta_cfg(cfg)

    if split_mode == "by_variant":
        if len(val_ids) == 0:
            logger.warning("[train_stage1] stage1_val.txt is empty; validation will be skipped.")
        train_variant_ids = list(train_ids)
        val_variant_ids = list(val_ids)
        train_indices_by_variant = None
        val_indices_by_variant = None
    else:
        if len(val_ids) > 0:
            logger.info("[train_stage1] split_mode=within_morph; stage1_val.txt will be ignored.")
        rng = np.random.RandomState(split_seed)
        train_indices_by_variant: Dict[str, List[int]] = {}
        val_indices_by_variant: Dict[str, List[int]] = {}
        for vid in train_ids:
            rec = manifest_by_id.get(vid, None)
            if rec is None:
                logger.warning(f"[train_stage1] Missing variant_id in manifest: {vid}")
                continue
            pose_path = (repo_root / rec["pose_path"]).resolve()
            data = load_pose_samples(pose_path)
            labels = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
            tr_idx, va_idx = _split_indices_by_gws(
                labels=labels,
                val_ratio=val_ratio,
                rng=rng,
                stratify=stratify_by_gws,
            )
            if len(tr_idx) > 0:
                train_indices_by_variant[vid] = tr_idx
            if len(va_idx) > 0:
                val_indices_by_variant[vid] = va_idx

        train_variant_ids = sorted(list(train_indices_by_variant.keys()))
        val_variant_ids = sorted(list(val_indices_by_variant.keys()))
        if len(train_variant_ids) == 0:
            raise ValueError("[train_stage1] Empty training split after within_morph split.")
        logger.info(
            f"[train_stage1] within_morph split: val_ratio={val_ratio} "
            f"stratify={stratify_by_gws} train_variants={len(train_variant_ids)} "
            f"val_variants={len(val_variant_ids)}"
        )
        for vid in train_variant_ids:
            tr_n = len(train_indices_by_variant.get(vid, []))
            va_n = len(val_indices_by_variant.get(vid, []))
            logger.info(f"[train_stage1] split_counts variant_id={vid} train={tr_n} val={va_n}")

    train_set = PoseDeltaDataset(
        repo_root=repo_root,
        manifest_records=manifest_records,
        variant_ids=train_variant_ids,
        label_keys=label_keys,
        pose_cfg=pose_cfg,
        delta_cfg=delta_cfg,
        sample_indices_by_variant=train_indices_by_variant,
        cache_pose=True,
        cache_specs=True,
    )

    val_set = None
    if len(val_variant_ids) > 0:
        val_set = PoseDeltaDataset(
            repo_root=repo_root,
            manifest_records=manifest_records,
            variant_ids=val_variant_ids,
            label_keys=label_keys,
            pose_cfg=pose_cfg,
            delta_cfg=delta_cfg,
            sample_indices_by_variant=val_indices_by_variant,
            cache_pose=True,
            cache_specs=True,
        )

    samp_cfg = BalancedSamplingConfig(
        batch_size=int(_get(cfg, "data.batch_size", 32)),
        ws_ratio=float(_get(cfg, "data.ws_ratio", 0.7)),
        seed=int(seed),
    )
    train_sampler = GroupedBalancedPoseBatchSampler(
        dataset=train_set,
        cfg=samp_cfg,
        repo_root=repo_root,
        manifest_by_id=manifest_by_id,
    )

    if val_set is not None:
        val_samp_cfg = BalancedSamplingConfig(
            batch_size=int(_get(cfg, "run.val_batch_size", 8)),
            ws_ratio=float(_get(cfg, "run.val_ws_ratio", _get(cfg, "data.ws_ratio", 0.7))),
            seed=int(seed) + 1,
        )
        val_sampler = GroupedBalancedPoseBatchSampler(
            dataset=val_set,
            cfg=val_samp_cfg,
            repo_root=repo_root,
            manifest_by_id=manifest_by_id,
        )
    else:
        val_sampler = None

    collate_fn = PoseBatchCollator(
        label_keys=label_keys,
        graph_bidirectional=bool(_get(cfg, "model.morph_graph.bidirectional", True)),
        graph_use_link_index=bool(_get(cfg, "model.morph_graph.use_link_index_feature", True)),
        strict_one_morph_per_batch=True,
        cache_graph=True,
    )

    num_workers = int(_get(cfg, "run.num_workers", 0))
    pin_memory = bool(_get(cfg, "run.pin_memory", False))

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )

    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    else:
        val_loader = None

    s0 = train_set[0]
    pose_in_dim = int(s0["pose_feats"].view(-1).shape[0])
    spec0 = s0["morph_spec"]

    g0 = build_link_graph(
        spec0,
        device=None,
        dtype=torch.float32,
        bidirectional=bool(_get(cfg, "model.morph_graph.bidirectional", True)),
        use_link_index_feature=bool(_get(cfg, "model.morph_graph.use_link_index_feature", True)),
    )
    node_in_dim = int(g0.x.shape[1])
    edge_in_dim = int(g0.edge_attr.shape[1]) if g0.edge_attr is not None else 0

    d_model = int(_get(cfg, "model.d_model", 256))
    logger.info(f"[model] pose_in_dim={pose_in_dim} node_in_dim={node_in_dim} d_model={d_model}")

    me_cfg = MorphologyEncoderConfig(
        input_dim=node_in_dim,
        hidden_dim=d_model,
        num_layers=int(_get(cfg, "model.morph_encoder.num_layers", 3)),
        edge_dim=edge_in_dim,
        dropout=float(_get(cfg, "model.morph_encoder.dropout", 0.0)),
        use_layernorm=bool(_get(cfg, "model.morph_encoder.use_layernorm", True)),
    )
    morph_encoder = MorphologyEncoderGNN(me_cfg)

    pe_cfg = {
        "pose_dim": pose_in_dim,
        "emb_dim": d_model,
        "num_bands": int(_get(cfg, "model.pose_encoder.num_bands", 10)),
        "mlp_hidden": int(_get(cfg, "model.pose_encoder.mlp_hidden", 256)),
        "mlp_layers": int(_get(cfg, "model.pose_encoder.mlp_layers", 3)),
        "dropout": float(_get(cfg, "model.pose_encoder.dropout", 0.0)),
        "include_xyz_raw": bool(_get(cfg, "model.pose_encoder.include_xyz_raw", True)),
    }
    pose_encoder = PoseEncoder.from_cfg(pe_cfg)

    bb_cfg = {
        "d_model": d_model,
        "nhead": int(_get(cfg, "model.backbone.nhead", 8)),
        "num_layers": int(_get(cfg, "model.backbone.num_layers", 6)),
        "dim_feedforward": int(_get(cfg, "model.backbone.dim_feedforward", 1024)),
        "dropout": float(_get(cfg, "model.backbone.dropout", 0.1)),
        "max_nodes": _get(cfg, "model.backbone.max_nodes", None),
    }
    backbone = TokenFusionBackbone.from_cfg(bb_cfg)

    ws_name = str(_get(cfg, "loss.ws_name", "g_ws"))
    ws_with_logits = bool(_get(cfg, "loss.ws_with_logits", True))

    heads_cfg = _build_heads_cfg_from_keys(
        label_keys=label_keys,
        ws_name=ws_name,
        ws_with_logits=ws_with_logits,
        heads_cfg=_get(cfg, "model.heads", {}) or {},
    )
    heads = MultiIndicatorHeads.from_cfg(in_dim=d_model, cfg=heads_cfg)

    model = MCFPStage1(
        morph_encoder=morph_encoder,
        pose_encoder=pose_encoder,
        backbone=backbone,
        heads=heads,
    ).to(device=device)

    head_weights = _get(cfg, "loss.head_weights", {}) or {}
    if not isinstance(head_weights, dict):
        head_weights = vars(head_weights)
    if len(head_weights) == 0:
        head_weights = {k: 1.0 for k in label_keys}

    loss_fn = MultiTaskLoss(
        head_weights=head_weights,
        ws_name=ws_name,
        ws_with_logits=ws_with_logits,
        reg_mask_key=str(_get(cfg, "loss.reg_mask_key", "delta_mask")),
        reg_mask_by_key=bool(_get(cfg, "loss.reg_mask_by_key", True)),
        mask_by_ws=bool(_get(cfg, "loss.mask_by_ws", True)),
        huber_delta=float(_get(cfg, "loss.huber_delta", 0.05)),
        ws_pos_weight=_get(cfg, "loss.ws_pos_weight", None),
        reg_clamp=bool(_get(cfg, "loss.reg_clamp", False)),
    ).to(device=device)

    metrics = Stage1Metrics(
        ws_name=ws_name,
        ws_is_logit=ws_with_logits,
        reg_mask_key=str(_get(cfg, "loss.reg_mask_key", "delta_mask")),
        reg_mask_by_key=bool(_get(cfg, "loss.reg_mask_by_key", True)),
        mask_by_ws=bool(_get(cfg, "loss.mask_by_ws", True)),
    )

    lr = float(_get(cfg, "optim.lr", 3e-4))
    wd = float(_get(cfg, "optim.weight_decay", 1e-2))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    use_amp = bool(_get(cfg, "optim.use_amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = float(_get(cfg, "optim.grad_clip_norm", 1.0))

    max_steps = int(_get(cfg, "run.max_steps", 20000))
    log_interval = int(_get(cfg, "run.log_interval", 50))
    val_interval = int(_get(cfg, "run.val_interval", 500))
    val_max_batches = int(_get(cfg, "run.val_max_batches", 50))
    save_interval = int(_get(cfg, "run.save_interval", 1000))
    keep_last_n = int(_get(cfg, "run.keep_last_n", 3))

    if args.smoke:
        max_steps = min(max_steps, int(args.smoke_steps))
        log_interval = max(1, min(50, log_interval))
        val_interval = max(1, min(50, max_steps))
        val_max_batches = min(val_max_batches, 10)
        save_interval = 0
        keep_last_n = 0
        logger.info(f"[smoke] enabled max_steps={max_steps} val_interval={val_interval}")

    metrics_fields = [
        "phase",
        "step",
        "max_steps",
        "lr",
        "loss",
        "loss/raw_sum",
        "loss/weighted_no_reg",
        "loss/total",
    ]
    for k in label_keys:
        metrics_fields.append(f"loss/{k}")
        if k != ws_name:
            metrics_fields.append(f"mask/{k}_den")
    metrics_fields.extend(["ws_acc", "ws_precision", "ws_recall", "ws_f1"])
    for k in label_keys:
        if k == ws_name:
            continue
        metrics_fields.append(f"{k}_mae")
        metrics_fields.append(f"{k}_rmse")

    logger.info("[train] start")
    step = 0
    model.train()
    start_time = time.time()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            batch["pose_feats"] = batch["pose_feats"].to(device=device, dtype=torch.float32)
            batch["labels"] = batch["labels"].to(device=device, dtype=torch.float32)
            batch["delta_mask"] = batch["delta_mask"].to(device=device, dtype=torch.float32)
            batch["ws_mask"] = batch["ws_mask"].to(device=device, dtype=torch.float32)
            batch["morph_graph"] = _graph_to_device(batch["morph_graph"], device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)
                loss_out = loss_fn(out.preds, batch)
                loss = loss_out.total

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optim)
            scaler.update()

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                pct = 100.0 * float(step + 1) / float(max_steps)
                lr_now = float(optim.param_groups[0].get("lr", lr))
                eta = _format_eta(elapsed, step + 1, max_steps)
                ws_ratio = float(batch["ws_mask"].mean().item()) if "ws_mask" in batch else 0.0
                logger.info(
                    f"[train] step={step+1}/{max_steps} ({pct:5.1f}%) "
                    f"lr={lr_now:.3e} loss={float(loss.item()):.6f} "
                    f"ws_ratio={ws_ratio:.3f} {eta}"
                )
                row = {
                    "phase": "train",
                    "step": step + 1,
                    "max_steps": max_steps,
                    "lr": lr_now,
                    "loss": float(loss.item()),
                }
                for k, v in loss_out.stats.items():
                    row[k] = v
                _append_metrics_csv(metrics_csv, row, metrics_fields)

            if val_loader is not None and val_interval > 0 and (step + 1) % val_interval == 0:
                model.eval()
                metrics.reset()
                val_losses: List[float] = []
                val_raw_sums: List[float] = []
                val_weighted_no_reg: List[float] = []
                val_stats_sum: Dict[str, float] = {}
                val_batches = 0
                ws_pos = 0.0
                ws_total = 0.0

                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= val_max_batches:
                            break
                        vb["pose_feats"] = vb["pose_feats"].to(device=device, dtype=torch.float32)
                        vb["labels"] = vb["labels"].to(device=device, dtype=torch.float32)
                        vb["delta_mask"] = vb["delta_mask"].to(device=device, dtype=torch.float32)
                        vb["ws_mask"] = vb["ws_mask"].to(device=device, dtype=torch.float32)
                        vb["morph_graph"] = _graph_to_device(vb["morph_graph"], device)

                        vout = model(vb)
                        vloss_out = loss_fn(vout.preds, vb)
                        val_losses.append(float(vloss_out.total.item()))
                        val_batches += 1
                        for k, v in vloss_out.stats.items():
                            val_stats_sum[k] = val_stats_sum.get(k, 0.0) + float(v)
                        if "loss/raw_sum" in vloss_out.stats:
                            val_raw_sums.append(float(vloss_out.stats["loss/raw_sum"]))
                        if "loss/weighted_no_reg" in vloss_out.stats:
                            val_weighted_no_reg.append(float(vloss_out.stats["loss/weighted_no_reg"]))
                        ws_pos += float(vb["ws_mask"].sum().item())
                        ws_total += float(vb["ws_mask"].numel())

                        labels_dict = _labels_tensor_to_dict(vb["labels"], vb["label_keys"])
                        metrics.update(vout.preds, labels_dict, batch=vb)

                m = metrics.compute()
                mean_val = sum(val_losses) / max(1, len(val_losses))
                mean_raw_sum = sum(val_raw_sums) / max(1, len(val_raw_sums))
                mean_weighted_no_reg = sum(val_weighted_no_reg) / max(1, len(val_weighted_no_reg))
                val_stats_avg = {}
                if val_batches > 0:
                    for k, v in val_stats_sum.items():
                        val_stats_avg[k] = v / float(val_batches)
                ws_ratio = ws_pos / max(1.0, ws_total)
                elapsed = time.time() - start_time
                pct = 100.0 * float(step + 1) / float(max_steps)
                eta = _format_eta(elapsed, step + 1, max_steps)
                logger.info(
                    f"[val]   step={step+1}/{max_steps} ({pct:5.1f}%) "
                    f"loss={mean_val:.6f} ws_ratio={ws_ratio:.3f} {eta} metrics={m.scalars}"
                )
                row = {
                    "phase": "val",
                    "step": step + 1,
                    "max_steps": max_steps,
                    "loss": mean_val,
                    "loss/raw_sum": mean_raw_sum,
                    "loss/weighted_no_reg": mean_weighted_no_reg,
                    "loss/total": mean_val,
                }
                for k, v in val_stats_avg.items():
                    if k not in row:
                        row[k] = v
                for k, v in m.scalars.items():
                    row[k] = v
                _append_metrics_csv(metrics_csv, row, metrics_fields)

                model.train()

            if save_interval > 0 and (step + 1) % save_interval == 0:
                ckpt_dir = run_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                path = ckpt_dir / f"step_{step+1:07d}.pt"
                obj = {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": None if scaler is None else scaler.state_dict(),
                }
                torch.save(obj, path)

                if keep_last_n > 0:
                    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
                    if len(all_ckpts) > keep_last_n:
                        for p in all_ckpts[:-keep_last_n]:
                            try:
                                p.unlink()
                            except Exception:
                                pass
                logger.info(f"[ckpt] saved step={step+1}")

            step += 1

    logger.info("[train] done")


if __name__ == "__main__":
    main()
