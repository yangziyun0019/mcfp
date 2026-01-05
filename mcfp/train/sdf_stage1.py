from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import shutil
import contextlib
import time

import numpy as np
import torch
from torch import nn

from mcfp.data.datasets import BatchRatios, MorphDatasetEntry, MultiMorphSDFDataset
from mcfp.models.morph_encoder import build_morph_graph_from_json
from mcfp.models.stage1 import MCFPStage1
from mcfp.utils.seed import set_seed


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read config value from dict-like or object-like cfg."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    """Read nested config value using a dotted key path."""
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
    """Resolve path relative to repo_root if needed."""
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _to_device(graph, device: torch.device):
    """Move GraphData to target device."""
    return graph.__class__(
        x=graph.x.to(device),
        edge_index=graph.edge_index.to(device),
        edge_attr=graph.edge_attr.to(device) if graph.edge_attr is not None else None,
        batch=graph.batch.to(device) if graph.batch is not None else None,
        node_names=graph.node_names,
        meta=graph.meta,
    )


def _compute_eikonal_loss(
    model: nn.Module,
    w_eik: torch.Tensor,
    morph_graph,
) -> torch.Tensor:
    """Compute Eikonal loss on w_eik."""
    if w_eik.numel() == 0:
        return torch.zeros((), device=w_eik.device)
    w_eik = w_eik.detach().requires_grad_(True)
    s = model(w_eik, morph_graph=morph_graph)
    grad = torch.autograd.grad(s.sum(), w_eik, create_graph=True)[0]
    grad_norm = torch.linalg.norm(grad, dim=-1)
    return torch.mean((grad_norm - 1.0) ** 2)


def _compute_loss_terms(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    morph_graph,
    *,
    margin_nb: float,
    margin_far: float,
    use_eikonal: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute individual loss terms for one batch."""
    w_b = batch["w_b"]
    w_in = batch["w_in"]
    w_out = batch["w_out"]
    w_rot_far = batch["w_rot_far"]
    w_pos_far = batch["w_pos_far"]
    w_eik = batch["w_eik"]

    s_b = model(w_b, morph_graph=morph_graph) if w_b.numel() > 0 else torch.zeros((), device=w_b.device)
    s_in = model(w_in, morph_graph=morph_graph) if w_in.numel() > 0 else torch.zeros((), device=w_b.device)
    s_out = model(w_out, morph_graph=morph_graph) if w_out.numel() > 0 else torch.zeros((), device=w_b.device)

    l0 = torch.mean(torch.abs(s_b)) if s_b.numel() > 0 else torch.zeros((), device=w_b.device)
    lnb_in = torch.mean(torch.relu(margin_nb - s_in)) if w_in.numel() > 0 else torch.zeros((), device=w_b.device)
    lnb_out = torch.mean(torch.relu(margin_nb + s_out)) if w_out.numel() > 0 else torch.zeros((), device=w_b.device)
    lnb = lnb_in + lnb_out

    w_far = torch.cat([w_rot_far, w_pos_far], dim=0)
    if w_far.numel() == 0:
        lfar = torch.zeros((), device=w_b.device)
    else:
        s_far = model(w_far, morph_graph=morph_graph)
        lfar = torch.mean(torch.relu(s_far + margin_far))

    leik = _compute_eikonal_loss(model, w_eik, morph_graph=morph_graph) if use_eikonal else torch.zeros((), device=w_b.device)

    metrics = {
        "l0": float(l0.detach().cpu()),
        "lnb": float(lnb.detach().cpu()),
        "lfar": float(lfar.detach().cpu()),
        "leik": float(leik.detach().cpu()),
    }
    return (l0, lnb, lfar, leik), metrics


def train_sdf_stage1(
    cfg: Any,
    logger,
    *,
    config_path: Optional[str | Path] = None,
) -> None:
    """Train Stage-1 implicit field with SDF data."""
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    run_dir = _resolve_path(_get_path(cfg, "paths.run_dir", "runs/sdf_stage1/exp001"), repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        dst = run_dir / "config.yaml"
        shutil.copyfile(config_path, dst)

    metrics_file = _get_path(cfg, "run.metrics_file", "metrics.csv")
    metrics_path = Path(metrics_file)
    if not metrics_path.is_absolute():
        metrics_path = run_dir / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fields = [
        "step",
        "split",
        "loss",
        "l0",
        "lnb",
        "lfar",
        "leik",
        "margin_nb",
        "margin_far",
        "elapsed_sec",
    ]
    metrics_exists = metrics_path.is_file() and metrics_path.stat().st_size > 0
    metrics_f = metrics_path.open("a", encoding="utf-8", newline="")
    metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
    if not metrics_exists:
        metrics_writer.writeheader()
        metrics_f.flush()

    seed = int(_get_path(cfg, "run.seed", 42))
    deterministic = bool(_get_path(cfg, "run.deterministic", True))
    device_str = str(_get_path(cfg, "run.device", "cpu"))
    max_steps = int(_get_path(cfg, "run.max_steps", 10000))
    log_interval = int(_get_path(cfg, "run.log_interval", 50))
    val_interval = int(_get_path(cfg, "run.val_interval", 500))
    val_batches = int(_get_path(cfg, "run.val_batches", 10))
    save_interval = int(_get_path(cfg, "run.save_interval", 1000))

    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("[sdf_stage1] CUDA unavailable, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    set_seed(int(seed), deterministic=deterministic)
    rng = np.random.default_rng(int(seed))

    data_cfg = _get(cfg, "data", None)
    batch_size = int(_get(data_cfg, "batch_size", 2048))
    morph_sampling = str(_get(data_cfg, "morph_sampling", "uniform"))
    require_boundary = bool(_get(data_cfg, "require_boundary", True))

    ratios_cfg = _get(data_cfg, "ratios", None)
    ratios = BatchRatios(
        boundary=float(_get(ratios_cfg, "boundary", 0.70)),
        rot_far=float(_get(ratios_cfg, "rot_far", 0.20)),
        pos_far=float(_get(ratios_cfg, "pos_far", 0.05)),
        eikonal=float(_get(ratios_cfg, "eikonal", 0.05)),
    )

    eik_cfg = _get(data_cfg, "eikonal", None)
    use_delta_sigma = bool(_get(eik_cfg, "use_delta_from_meta", True))
    sigma_scale = float(_get(eik_cfg, "sigma_scale", 0.5))
    sigma_abs = float(_get(eik_cfg, "sigma", 0.0025))
    cover_ratio = float(_get(eik_cfg, "cover_ratio", 0.05))
    cover_scale = float(_get(eik_cfg, "cover_scale", 1.2))

    entries_cfg = list(_get(data_cfg, "datasets", []))
    if not entries_cfg:
        raise ValueError("[sdf_stage1] data.datasets must be non-empty.")
    entries: List[MorphDatasetEntry] = []
    for item in entries_cfg:
        ds_root = _resolve_path(_get(item, "dataset_root", None), repo_root)
        spec_path = _resolve_path(_get(item, "spec_path", None), repo_root)
        entries.append(MorphDatasetEntry(dataset_root=ds_root, spec_path=spec_path))

    dataset = MultiMorphSDFDataset(entries, require_boundary=require_boundary)

    graphs = []
    for ds, entry in zip(dataset.datasets, dataset.entries):
        graph = build_morph_graph_from_json(entry.spec_path, l_ref=ds.l_ref, device=device)
        graphs.append(_to_device(graph, device))

    node_feat_dim = graphs[0].x.shape[1]

    model_cfg = _get(cfg, "model", None)
    morph_cfg = _get(model_cfg, "morph_encoder", None)
    pose_cfg = _get(model_cfg, "pose_encoder", None)
    backbone_cfg = _get(model_cfg, "backbone", None)

    model = MCFPStage1(
        node_feat_dim=int(node_feat_dim),
        morph_dim=int(_get(morph_cfg, "out_dim", 128)),
        pose_dim=int(_get(pose_cfg, "out_dim", 128)),
        pose_hidden_dims=list(_get(pose_cfg, "hidden_dims", [128, 128])),
        pose_fourier_dim=int(_get(pose_cfg, "fourier_dim", 0)),
        pose_fourier_scale=float(_get(pose_cfg, "fourier_scale", 10.0)),
        d_model=int(_get(morph_cfg, "d_model", 128)),
        n_layers=int(_get(morph_cfg, "n_layers", 4)),
        n_heads=int(_get(morph_cfg, "n_heads", 8)),
        d_ff=int(_get(morph_cfg, "d_ff", 512)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
        backbone_hidden_dim=int(_get(backbone_cfg, "hidden_dim", 128)),
        backbone_num_layers=int(_get(backbone_cfg, "num_layers", 4)),
        backbone_out_dim=int(_get(backbone_cfg, "out_dim", 128)),
        w0_first=float(_get(backbone_cfg, "w0_first", 30.0)),
        w0=float(_get(backbone_cfg, "w0", 1.0)),
    ).to(device)

    optim_cfg = _get(cfg, "optim", None)
    lr = float(_get(optim_cfg, "lr", 3e-4))
    weight_decay = float(_get(optim_cfg, "weight_decay", 0.01))
    grad_clip = float(_get(optim_cfg, "grad_clip_norm", 1.0))
    use_amp = bool(_get(optim_cfg, "use_amp", True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    loss_cfg = _get(cfg, "loss", None)
    margin_nb = _get(loss_cfg, "margin_nb", None)
    margin_nb_scale = float(_get(loss_cfg, "margin_nb_scale", 0.5))
    margin_far = float(_get(loss_cfg, "margin_far", 0.02))
    beta = float(_get(loss_cfg, "beta", 0.5))
    gamma = float(_get(loss_cfg, "gamma", 0.1))

    logger.info(f"[sdf_stage1] run_dir={run_dir}")
    logger.info(f"[sdf_stage1] device={device}, seed={seed}")
    logger.info(f"[sdf_stage1] batch_size={batch_size}, ratios={asdict(ratios)}")
    logger.info(f"[sdf_stage1] metrics_path={metrics_path}")

    step = 0
    start_time = time.time()
    try:
        while step < max_steps:
            model.train()
            batch = dataset.sample_batch(
                batch_size=batch_size,
                ratios=ratios,
                rng=rng,
                eikonal_sigma=None if use_delta_sigma else sigma_abs,
                eikonal_sigma_scale=sigma_scale,
                eikonal_cover_ratio=cover_ratio,
                cover_scale=cover_scale,
                morph_sampling=morph_sampling,
            )
            morph_id = int(batch["morph_id"])
            graph = graphs[morph_id]
            delta = float(batch["meta"].get("delta", 0.0))
            if margin_nb is None:
                if delta <= 0.0:
                    raise ValueError("[sdf_stage1] margin_nb is None and meta.delta <= 0.")
                m_nb = margin_nb_scale * delta
            else:
                m_nb = float(margin_nb)

            w_b = batch["w_b"].to(device)
            w_in = batch["w_in"].to(device)
            w_out = batch["w_out"].to(device)
            w_rot_far = batch["w_rot_far"].to(device)
            w_pos_far = batch["w_pos_far"].to(device)
            w_eik = batch["w_eik"].to(device)

            batch_t = {
                "w_b": w_b,
                "w_in": w_in,
                "w_out": w_out,
                "w_rot_far": w_rot_far,
                "w_pos_far": w_pos_far,
                "w_eik": w_eik,
            }

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(enabled=scaler.is_enabled())
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                (l0, lnb, lfar, leik), metrics = _compute_loss_terms(
                    model,
                    batch_t,
                    graph,
                    margin_nb=float(m_nb),
                    margin_far=float(margin_far),
                )
                loss = l0 + lnb + beta * lfar + gamma * leik

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            loss_val = float(loss.detach().cpu())
            if log_interval > 0 and (step % log_interval == 0):
                logger.info(
                    "[sdf_stage1] step=%d loss=%.6f l0=%.6f lnb=%.6f lfar=%.6f leik=%.6f",
                    step,
                    loss_val,
                    metrics["l0"],
                    metrics["lnb"],
                    metrics["lfar"],
                    metrics["leik"],
                )
                metrics_writer.writerow(
                    {
                        "step": step,
                        "split": "train",
                        "loss": loss_val,
                        "l0": metrics["l0"],
                        "lnb": metrics["lnb"],
                        "lfar": metrics["lfar"],
                        "leik": metrics["leik"],
                        "margin_nb": float(m_nb),
                        "margin_far": float(margin_far),
                        "elapsed_sec": time.time() - start_time,
                    }
                )
                metrics_f.flush()

            if val_interval > 0 and (step % val_interval == 0) and val_batches > 0:
                model.eval()
                val_losses: List[float] = []
                val_terms = {"l0": 0.0, "lnb": 0.0, "lfar": 0.0, "leik": 0.0}
                with torch.no_grad():
                    for _ in range(val_batches):
                        batch = dataset.sample_batch(
                            batch_size=batch_size,
                            ratios=ratios,
                            rng=rng,
                            eikonal_sigma=None if use_delta_sigma else sigma_abs,
                            eikonal_sigma_scale=sigma_scale,
                            eikonal_cover_ratio=cover_ratio,
                            cover_scale=cover_scale,
                            morph_sampling=morph_sampling,
                        )
                        morph_id = int(batch["morph_id"])
                        graph = graphs[morph_id]
                        delta = float(batch["meta"].get("delta", 0.0))
                        if margin_nb is None:
                            m_nb = margin_nb_scale * delta if delta > 0 else float(margin_far)
                        else:
                            m_nb = float(margin_nb)

                        batch_t = {
                            "w_b": batch["w_b"].to(device),
                            "w_in": batch["w_in"].to(device),
                            "w_out": batch["w_out"].to(device),
                            "w_rot_far": batch["w_rot_far"].to(device),
                            "w_pos_far": batch["w_pos_far"].to(device),
                            "w_eik": batch["w_eik"].to(device),
                        }
                        (l0, lnb, lfar, leik), metrics = _compute_loss_terms(
                            model,
                            batch_t,
                            graph,
                            margin_nb=float(m_nb),
                            margin_far=float(margin_far),
                            use_eikonal=False,
                        )
                        loss = l0 + lnb + beta * lfar
                        val_losses.append(float(loss))
                        for k in val_terms:
                            val_terms[k] += float(metrics[k])
                val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                for k in val_terms:
                    val_terms[k] = val_terms[k] / float(max(val_batches, 1))
                logger.info("[sdf_stage1] val step=%d loss=%.6f", step, val_loss)
                metrics_writer.writerow(
                    {
                        "step": step,
                        "split": "val",
                        "loss": val_loss,
                        "l0": val_terms["l0"],
                        "lnb": val_terms["lnb"],
                        "lfar": val_terms["lfar"],
                        "leik": val_terms["leik"],
                        "margin_nb": float(m_nb),
                        "margin_far": float(margin_far),
                        "elapsed_sec": time.time() - start_time,
                    }
                )
                metrics_f.flush()

            if save_interval > 0 and (step % save_interval == 0):
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, run_dir / f"checkpoint_{step:06d}.pt")

            step += 1

        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, run_dir / "checkpoint_final.pt")
    finally:
        metrics_f.close()
