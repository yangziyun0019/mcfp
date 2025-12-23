# scripts/train_stage1.py

from __future__ import annotations

import argparse
import csv
import json
import logging
import inspect
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from mcfp.data.io import read_jsonl
from mcfp.data.datasets import Stage1Dataset, PoseFeatureConfig
from mcfp.data.sampling import GroupedBalancedBatchSampler, BalancedSamplingConfig

from mcfp.models.morph_graph import GraphData, build_link_graph
from mcfp.models.morph_encoder import MorphologyEncoderGNN, MorphologyEncoderConfig
from mcfp.models.pose_encoder import PoseEncoder
from mcfp.models.backbone import TokenFusionBackbone
from mcfp.models.heads import MultiIndicatorHeads
from mcfp.models.stage1 import MCFPStage1

from mcfp.train.losses import MultiTaskLoss
from mcfp.train.metrics import Stage1Metrics

from mcfp.utils.seed import set_seed, seed_worker


# -----------------------------
# utils
# -----------------------------

def _get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _read_lines(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                lines.append(s)
    return lines


def _load_label_keys(cfg: Dict[str, Any], stats_path: Path) -> List[str]:
    user_keys = list(_get(cfg, "data.label_keys", [])) or []
    if len(user_keys) > 0:
        return [str(k) for k in user_keys]

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    keys = stats.get("label_keys", None)
    if not keys:
        raise ValueError(f"[train_stage1] stats file missing label_keys: {stats_path}")
    return [str(k) for k in keys]


def _make_pose_cfg(cfg: Dict[str, Any]) -> PoseFeatureConfig:
    """
    Build PoseFeatureConfig without mutating fields.
    Works with frozen dataclass.

    It will:
      - prefer new names: include_aabb_ratio/include_aabb_centered/include_morph_scale
      - fallback to old names: use_aabb_ratio/use_aabb_centered/use_morph_scale
    """
    pf = _get(cfg, "data.pose_features", {}) or {}

    # Normalize possible key variants into a canonical dict first
    norm: Dict[str, Any] = {}
    # new names
    if "include_aabb_ratio" in pf:
        norm["include_aabb_ratio"] = bool(pf["include_aabb_ratio"])
    if "include_aabb_centered" in pf:
        norm["include_aabb_centered"] = bool(pf["include_aabb_centered"])
    if "include_morph_scale" in pf:
        norm["include_morph_scale"] = bool(pf["include_morph_scale"])
    if "grid_round_decimals" in pf:
        norm["grid_round_decimals"] = int(pf["grid_round_decimals"])

    # old names fallback (only fill if new ones are absent)
    if "use_aabb_ratio" in pf and "include_aabb_ratio" not in norm:
        norm["use_aabb_ratio"] = bool(pf["use_aabb_ratio"])
    if "use_aabb_centered" in pf and "include_aabb_centered" not in norm:
        norm["use_aabb_centered"] = bool(pf["use_aabb_centered"])
    if "use_morph_scale" in pf and "include_morph_scale" not in norm:
        norm["use_morph_scale"] = bool(pf["use_morph_scale"])
    if "grid_round_decimals" in pf:
        norm["grid_round_decimals"] = int(pf["grid_round_decimals"])

    # Filter kwargs by PoseFeatureConfig signature
    import inspect
    sig = inspect.signature(PoseFeatureConfig)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in norm.items() if k in allowed}

    return PoseFeatureConfig(**kwargs)



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
    head_list = list(_get(heads_cfg, "heads", [])) or []
    if len(head_list) > 0:
        return {"heads": head_list}

    d_hidden = int(_get(heads_cfg, "default_hidden_dim", 256))
    n_layers = int(_get(heads_cfg, "default_num_layers", 2))
    drop = float(_get(heads_cfg, "default_dropout", 0.0))
    act_reg = str(_get(heads_cfg, "default_out_activation", "sigmoid"))
    act_ws = str(_get(heads_cfg, "ws_out_activation", "identity"))

    built = []
    for name in label_keys:
        name = str(name)
        if name == ws_name:
            out_act = act_ws if ws_with_logits else act_reg
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


def _make_grouped_collate(
    label_keys: List[str],
    graph_cfg: Dict[str, Any],
) :
    """
    Contract v1 batch:
      - pose_feats: Tensor[B,F] (F usually 9)
      - labels: Tensor[B,K]
      - label_keys: List[str]
      - ws_mask: Tensor[B] float32
      - morph_graph: GraphData
      - variant_id: str
    """
    bidir = bool(_get(graph_cfg, "bidirectional", True))
    use_idx = bool(_get(graph_cfg, "use_link_index_feature", True))

    graph_cache: Dict[str, GraphData] = {}

    def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(items) == 0:
            raise ValueError("[collate] empty batch")

        vid0 = str(items[0].get("variant_id"))
        for it in items:
            if str(it.get("variant_id")) != vid0:
                raise ValueError(f"[collate] batch has multiple variant_id: {vid0} vs {it.get('variant_id')}")

        # pose feats (support both 'pose_feats' and legacy 'point_feats')
        pose_list = []
        for it in items:
            x = it.get("pose_feats", None)
            if x is None:
                x = it.get("point_feats", None)
            if x is None:
                raise KeyError("[collate] missing pose_feats/point_feats in sample")
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            pose_list.append(x.view(-1))
        pose_feats = torch.stack(pose_list, dim=0).to(dtype=torch.float32)

        # labels (expect Tensor[K] per sample; also tolerate dict labels)
        y0 = items[0].get("labels", None)
        if y0 is None:
            raise KeyError("[collate] missing labels in sample")

        if isinstance(y0, torch.Tensor):
            lab_list = []
            for it in items:
                y = it["labels"]
                if not isinstance(y, torch.Tensor):
                    y = torch.as_tensor(y, dtype=torch.float32)
                lab_list.append(y.view(-1))
            labels = torch.stack(lab_list, dim=0).to(dtype=torch.float32)
        elif isinstance(y0, dict):
            # dict -> tensor [B,K] aligned by label_keys
            rows = []
            for it in items:
                row = [float(it["labels"][k]) for k in label_keys]
                rows.append(row)
            labels = torch.as_tensor(rows, dtype=torch.float32)
        else:
            raise TypeError(f"[collate] unsupported labels type: {type(y0)}")

        # ws_mask (float/bool tolerated)
        ws_list = []
        for it in items:
            w = it.get("ws_mask", None)
            if w is None:
                ws_list.append(0.0)
            else:
                if isinstance(w, torch.Tensor):
                    wv = float(w.view(-1)[0].item())
                else:
                    wv = float(w)
                ws_list.append(1.0 if wv > 0.5 else 0.0)
        ws_mask = torch.as_tensor(ws_list, dtype=torch.float32)

        # morph_graph (one per batch)
        if vid0 in graph_cache:
            g = graph_cache[vid0]
        else:
            spec = items[0].get("morph_spec", None)
            if spec is None:
                raise KeyError("[collate] missing morph_spec in sample")
            g0 = build_link_graph(
                spec,
                device=None,
                dtype=torch.float32,
                bidirectional=bidir,
                use_link_index_feature=use_idx,
            )
            # Optional: ensure g.batch exists for pooling/debug
            if g0.batch is None:
                g = GraphData(
                    x=g0.x,
                    edge_index=g0.edge_index,
                    edge_attr=g0.edge_attr,
                    batch=torch.zeros((g0.x.shape[0],), dtype=torch.long),
                    node_names=g0.node_names,
                    meta=g0.meta,
                )
            else:
                g = g0
            graph_cache[vid0] = g

        return {
            "variant_id": vid0,
            "pose_feats": pose_feats,
            "labels": labels,
            "label_keys": list(label_keys),
            "ws_mask": ws_mask,
            "morph_graph": g,
        }

    return _collate


def _setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _append_metrics_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """Append a row to metrics.csv, creating it with headers if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    row_aligned = {k: row.get(k, "") for k in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as f:
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


def _log_coverage_summary(
    logger: logging.Logger,
    total_variants: int,
    seen_variants: set,
    family_to_variants: Dict[str, set],
    family_seen: Dict[str, set],
    family_sample_counts: Dict[str, int],
    total_samples: int,
) -> None:
    """Log coverage summary for variants and families after training."""
    seen_count = len(seen_variants)
    seen_ratio = float(seen_count) / float(max(1, total_variants))
    logger.info(f"[coverage] variants_seen={seen_count}/{total_variants} ({seen_ratio:.3f})")

    fam_names = sorted(family_to_variants.keys())
    for fam in fam_names:
        total_f = len(family_to_variants.get(fam, set()))
        seen_f = len(family_seen.get(fam, set()))
        cov = float(seen_f) / float(max(1, total_f))
        share = float(family_sample_counts.get(fam, 0)) / float(max(1, total_samples))
        logger.info(
            f"[coverage] family={fam} variants_seen={seen_f}/{total_f} ({cov:.3f}) "
            f"sample_share={share:.3f}"
        )


def _save_checkpoint(
    run_dir: Path,
    step: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    keep_last_n: int,
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"

    obj = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    torch.save(obj, path)

    # cleanup
    if keep_last_n > 0:
        all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if len(all_ckpts) > keep_last_n:
            for p in all_ckpts[:-keep_last_n]:
                try:
                    p.unlink()
                except Exception:
                    pass


# -----------------------------
# main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true", help="Run a short smoke test.")
    ap.add_argument("--smoke_steps", type=int, default=50)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(_get(cfg, "paths.repo_root")).resolve()
    run_dir = (repo_root / _get(cfg, "run.run_dir")).resolve()
    logger = _setup_logger(run_dir)
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
    K = len(label_keys)
    logger.info(f"[data] K={K}, label_keys[:8]={label_keys[:8]}")

    # Load manifest + splits
    manifest_records = read_jsonl(manifest_path)
    manifest_by_id = {str(r["variant_id"]): r for r in manifest_records}

    train_ids = _read_lines(splits_dir / "stage1_train.txt")
    val_ids = _read_lines(splits_dir / "stage1_val.txt")
    test_ids = _read_lines(splits_dir / "stage1_test.txt") if (splits_dir / "stage1_test.txt").exists() else []

    pose_cfg = _make_pose_cfg(cfg)

    train_set = Stage1Dataset(
        repo_root=repo_root,
        manifest_records=manifest_records,
        variant_ids=train_ids,
        label_keys=label_keys,
        pose_cfg=pose_cfg,
        cache_maps=True,
        cache_specs=True,
    )

    val_set = Stage1Dataset(
        repo_root=repo_root,
        manifest_records=manifest_records,
        variant_ids=val_ids,
        label_keys=label_keys,
        pose_cfg=pose_cfg,
        cache_maps=True,
        cache_specs=True,
    )

    # Coverage bookkeeping (train variants only)
    train_family_by_id: Dict[str, str] = {}
    family_to_variants: Dict[str, set] = {}
    for vid in train_ids:
        rec = train_set.records_by_id.get(vid, {})
        fam = str(rec.get("family", "unknown"))
        train_family_by_id[vid] = fam
        family_to_variants.setdefault(fam, set()).add(vid)

    # Batch sampler (grouped by variant_id)
    samp_norm = {
        "batch_size": int(_get(cfg, "data.batch_size", 8)),
        "ws_ratio": float(_get(cfg, "data.ws_ratio", 0.8)),
        "seed": int(seed),
    }

    sig = inspect.signature(BalancedSamplingConfig)
    allowed = set(sig.parameters.keys())
    samp_kwargs = {k: v for k, v in samp_norm.items() if k in allowed}

    samp_cfg = BalancedSamplingConfig(**samp_kwargs)

    train_sampler = GroupedBalancedBatchSampler(
        dataset=train_set,
        cfg=samp_cfg,
        repo_root=repo_root,
        manifest_by_id=manifest_by_id,
    )

    collate_fn = _make_grouped_collate(
        label_keys=label_keys,
        graph_cfg=_get(cfg, "model.morph_graph", {}) or {},
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

    # For val: sampled loader to control ws/non-ws ratio (default to data.ws_ratio).
    val_ws_ratio = float(_get(cfg, "run.val_ws_ratio", _get(cfg, "data.ws_ratio", 0.8)))
    val_batch_size = int(_get(cfg, "run.val_batch_size", 1))
    val_samp_cfg = BalancedSamplingConfig(
        batch_size=val_batch_size,
        ws_ratio=val_ws_ratio,
        seed=int(seed) + 1,
    )
    val_sampler = GroupedBalancedBatchSampler(
        dataset=val_set,
        cfg=val_samp_cfg,
        repo_root=repo_root,
        manifest_by_id=manifest_by_id,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    # Infer dims from one dataset sample
    s0 = train_set[0]
    pf0 = s0.get("pose_feats", s0.get("point_feats"))
    if pf0 is None:
        raise KeyError("[train_stage1] dataset sample missing pose_feats/point_feats")
    pose_in_dim = int(pf0.view(-1).shape[0])

    spec0 = s0.get("morph_spec", None)
    if spec0 is None:
        raise KeyError("[train_stage1] dataset sample missing morph_spec")

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
    logger.info(f"[model] pose_in_dim={pose_in_dim}, node_in_dim={node_in_dim}, d_model={d_model}")

    # Build model (explicit wiring; no guessing)
    me_cfg = MorphologyEncoderConfig(
        input_dim=node_in_dim,
        hidden_dim=d_model,  # enforce same as backbone token dim
        num_layers=int(_get(cfg, "model.morph_encoder.num_layers", 3)),
        edge_dim=edge_in_dim,
        dropout=float(_get(cfg, "model.morph_encoder.dropout", 0.0)),
        use_layernorm=bool(_get(cfg, "model.morph_encoder.use_layernorm", True)),
    )
    morph_encoder = MorphologyEncoderGNN(me_cfg)

    pe_cfg = dict(_get(cfg, "model.pose_encoder", {}) or {})
    pe_cfg["pose_dim"] = pose_in_dim
    pe_cfg["emb_dim"] = d_model
    pose_encoder = PoseEncoder.from_cfg(pe_cfg)

    bb_cfg = dict(_get(cfg, "model.backbone", {}) or {})
    bb_cfg["d_model"] = d_model
    backbone = TokenFusionBackbone.from_cfg(bb_cfg)

    ws_name = str(_get(cfg, "loss.ws_name", "g_ws"))
    ws_with_logits = bool(_get(cfg, "loss.ws_with_logits", True))
    metrics_fields: List[str] = [
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
    for k in label_keys:
        if k == ws_name:
            continue
        metrics_fields.append(f"mask/{k}_den")
    metrics_fields.extend(["ws_acc", "ws_precision", "ws_recall", "ws_f1"])
    for k in label_keys:
        if k == ws_name:
            continue
        metrics_fields.append(f"{k}_mae")
        metrics_fields.append(f"{k}_rmse")
    heads_cfg = _build_heads_cfg_from_keys(
        label_keys=label_keys,
        ws_name=ws_name,
        ws_with_logits=ws_with_logits,
        heads_cfg=dict(_get(cfg, "model.heads", {}) or {}),
    )
    heads = MultiIndicatorHeads.from_cfg(in_dim=d_model, cfg=heads_cfg)

    model = MCFPStage1(
        morph_encoder=morph_encoder,
        pose_encoder=pose_encoder,
        backbone=backbone,
        heads=heads,
    ).to(device=device)

    # Loss + metrics
    head_weights = dict(_get(cfg, "loss.head_weights", {}) or {})
    if len(head_weights) == 0:
        head_weights = {k: 1.0 for k in label_keys}

    loss_fn = MultiTaskLoss(
        head_weights=head_weights,
        ws_name=ws_name,
        ws_with_logits=ws_with_logits,
        mask_by_ws=bool(_get(cfg, "loss.mask_by_ws", True)),
        huber_delta=float(_get(cfg, "loss.huber_delta", 0.05)),
        ws_pos_weight=_get(cfg, "loss.ws_pos_weight", None),
    ).to(device=device)

    metrics = Stage1Metrics(
        ws_name=ws_name,
        ws_is_logit=ws_with_logits,
        mask_by_ws=bool(_get(cfg, "loss.mask_by_ws", True)),
    )

    # Optim
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
        max_steps = min(max_steps, 1000)
        log_interval = 100
        val_interval = max(1, min(50, max_steps))
        val_max_batches = min(val_max_batches, 20)
        save_interval = 0
        keep_last_n = 0
        logger.info(f"[smoke] enabled max_steps={max_steps} val_interval={val_interval}")

    logger.info("[train] start")

    step = 0
    model.train()
    start_time = time.time()

    seen_variants: set = set()
    family_seen: Dict[str, set] = {k: set() for k in family_to_variants.keys()}
    family_sample_counts: Dict[str, int] = {k: 0 for k in family_to_variants.keys()}
    total_samples = 0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            # Move batch to device
            batch["pose_feats"] = batch["pose_feats"].to(device=device, dtype=torch.float32)
            batch["labels"] = batch["labels"].to(device=device, dtype=torch.float32)
            batch["ws_mask"] = batch["ws_mask"].to(device=device, dtype=torch.float32)
            batch["morph_graph"] = _graph_to_device(batch["morph_graph"], device)

            # Coverage tracking
            vid = str(batch.get("variant_id", ""))
            fam = train_family_by_id.get(vid, "unknown")
            seen_variants.add(vid)
            family_seen.setdefault(fam, set()).add(vid)
            bs = int(batch["pose_feats"].shape[0])
            family_sample_counts[fam] = family_sample_counts.get(fam, 0) + bs
            total_samples += bs

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)  # Stage1Output(preds=..., z=...)
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
                    f"[train] step={step+1}/{max_steps} ({pct:5.1f}%) lr={lr_now:.3e} "
                    f"loss={float(loss.item()):.6f} ws_ratio={ws_ratio:.3f} {eta}"
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

            if (step + 1) % val_interval == 0:
                model.eval()
                metrics.reset()
                val_losses: List[float] = []
                val_raw_sums: List[float] = []
                val_weighted_no_reg: List[float] = []
                ws_pos = 0.0
                ws_total = 0.0

                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= val_max_batches:
                            break
                        vb["pose_feats"] = vb["pose_feats"].to(device=device, dtype=torch.float32)
                        vb["labels"] = vb["labels"].to(device=device, dtype=torch.float32)
                        vb["ws_mask"] = vb["ws_mask"].to(device=device, dtype=torch.float32)
                        vb["morph_graph"] = _graph_to_device(vb["morph_graph"], device)

                        vout = model(vb)
                        vloss_out = loss_fn(vout.preds, vb)
                        val_losses.append(float(vloss_out.total.item()))
                        if "loss/raw_sum" in vloss_out.stats:
                            val_raw_sums.append(float(vloss_out.stats["loss/raw_sum"]))
                        if "loss/weighted_no_reg" in vloss_out.stats:
                            val_weighted_no_reg.append(float(vloss_out.stats["loss/weighted_no_reg"]))
                        ws_pos += float(vb["ws_mask"].sum().item())
                        ws_total += float(vb["ws_mask"].numel())

                        # metrics expects labels dict
                        labels_dict = _labels_tensor_to_dict(vb["labels"], vb["label_keys"])
                        metrics.update(vout.preds, labels_dict)

                m = metrics.compute()
                mean_val = sum(val_losses) / max(1, len(val_losses))
                mean_raw_sum = sum(val_raw_sums) / max(1, len(val_raw_sums))
                mean_weighted_no_reg = sum(val_weighted_no_reg) / max(1, len(val_weighted_no_reg))
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
                }
                for k, v in m.scalars.items():
                    row[k] = v
                _append_metrics_csv(metrics_csv, row, metrics_fields)

                model.train()

            if save_interval > 0 and (step + 1) % save_interval == 0:
                _save_checkpoint(run_dir, step + 1, model, optim, scaler if use_amp else None, keep_last_n)
                logger.info(f"[ckpt] saved step={step+1}")

            step += 1

    logger.info("[train] done")

    _log_coverage_summary(
        logger=logger,
        total_variants=len(train_ids),
        seen_variants=seen_variants,
        family_to_variants=family_to_variants,
        family_seen=family_seen,
        family_sample_counts=family_sample_counts,
        total_samples=total_samples,
    )


if __name__ == "__main__":
    main()
