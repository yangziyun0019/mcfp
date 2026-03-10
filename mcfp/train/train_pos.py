"""Train the Position-SDF model on prepared voxel datasets.

This module contains the full optimization and validation loop for position training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import csv
import time
import queue as pyqueue

import numpy as np
import torch
from torch import nn

from mcfp.data.pos_dataset import PositionDataset
from mcfp.data.morph_spec_io import load_morphology_spec
from mcfp.models.encodings import position_encoding
from mcfp.models.morph_encoder import MorphologyEncoder
from mcfp.models.pos_sdf import PositionSDFModel
from mcfp.utils.prefetch import Prefetcher


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


def _pos_worker(
    queue,
    stop_event,
    seed: int,
    pos_path: str,
    split_voxel: str | None,
    cache_grid: bool,
    batch_size: int,
    ratios: tuple,
    split_id: int,
    s_bucket_weights: tuple | None,
    s_bucket_tau: float | None,
    sign_balance: float,
) -> None:
    rng = np.random.default_rng(int(seed))
    dataset = PositionDataset(Path(pos_path), Path(split_voxel) if split_voxel else None, cache_grid=cache_grid)
    while not stop_event.is_set():
        batch = dataset.sample_batch(
            rng,
            batch_size,
            ratios,
            split_id=split_id,
            s_bucket_weights=s_bucket_weights,
            s_bucket_tau=s_bucket_tau,
            sign_balance=sign_balance,
        )
        while not stop_event.is_set():
            try:
                queue.put(batch, timeout=0.5)
                break
            except pyqueue.Full:
                continue


def train_pos(cfg: Any, logger) -> None:
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    run_dir = _resolve_path(_get_path(cfg, "paths.run_dir", "runs/pos_sdf/exp001"), repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))
    seed = int(_get_path(cfg, "run.seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Data
    pos_path = _resolve_path(_get_path(cfg, "paths.position_h5"), repo_root)
    split_voxel_cfg = _get_path(cfg, "paths.splits_voxel", None)
    split_voxel = _resolve_path(split_voxel_cfg, repo_root) if split_voxel_cfg else None
    dataset = PositionDataset(pos_path, split_voxel, cache_grid=bool(_get_path(cfg, "data.cache_grid", True)))

    # Morphology
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    # Models
    k_p = int(_get_path(cfg, "model.position.k_p", 10))
    morph_cfg = _get_path(cfg, "model.morph", None)
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)

    pos_model = PositionSDFModel(
        in_dim=3 + 2 * 3 * k_p,
        hidden_dim=int(_get_path(cfg, "model.position.hidden_dim", 256)),
        num_layers=int(_get_path(cfg, "model.position.num_layers", 5)),
        w0_first=float(_get_path(cfg, "model.position.w0_first", 30.0)),
        w0=float(_get_path(cfg, "model.position.w0", 1.0)),
        cond_dim=int(_get(morph_cfg, "d_model", 256)),
    ).to(device)

    params = list(morph_encoder.parameters()) + list(pos_model.parameters())
    optimizer = torch.optim.Adam(params, lr=float(_get_path(cfg, "optim.lr", 1e-4)))

    # Training settings
    max_steps = int(_get_path(cfg, "run.max_steps", 200000))
    log_interval = int(_get_path(cfg, "run.log_interval", 100))
    val_interval = int(_get_path(cfg, "run.val_interval", 10000))
    val_batches = int(_get_path(cfg, "run.val_batches", 5))
    save_interval = int(_get_path(cfg, "run.save_interval", 10000))
    overfit_batch = bool(_get_path(cfg, "run.overfit_batch", False))
    overfit_steps = int(_get_path(cfg, "run.overfit_steps", 2000))
    if overfit_batch:
        max_steps = min(max_steps, max(1, overfit_steps))

    batch_size = int(_get_path(cfg, "data.batch_size", 8192))
    ratios = tuple(_get_path(cfg, "data.ratios", [0.35, 0.65]))
    s_bucket_weights_cfg = _get_path(cfg, "data.s_bucket_weights", None)
    s_bucket_weights = tuple(s_bucket_weights_cfg) if s_bucket_weights_cfg is not None else None
    if s_bucket_weights is None:
        legacy = _get_path(cfg, "data.inner_tier_ratios", None)
        if legacy is not None:
            s_bucket_weights = tuple(legacy)
    s_bucket_tau = _get_path(cfg, "data.s_bucket_tau", None)
    if s_bucket_tau is not None:
        s_bucket_tau = float(s_bucket_tau)
    sign_balance = float(_get_path(cfg, "data.sign_balance", 0.5))
    w_sign = float(_get_path(cfg, "loss.sign_weight", 1.0))
    w_reg = float(_get_path(cfg, "loss.reg_weight", 1.0))
    w_bnd = float(_get_path(cfg, "loss.boundary_weight", 1.0))
    w_eik = float(_get_path(cfg, "loss.eikonal_weight", 0.0))
    reg_warmup = int(_get_path(cfg, "loss.reg_warmup_steps", 0))
    eik_warmup = int(_get_path(cfg, "loss.eikonal_warmup_steps", 0))
    k_eq_start = float(_get_path(cfg, "loss.k_eq_start", 1.0))
    k_eq_end = float(_get_path(cfg, "loss.k_eq_end", k_eq_start))
    k_eq_warmup = int(_get_path(cfg, "loss.k_eq_warmup_steps", 0))
    reg_weights = _get_path(cfg, "loss.reg_weights", [1.0])
    sign_bucket_weights = _get_path(cfg, "loss.sign_bucket_weights", None)
    if sign_bucket_weights is None:
        sign_bucket_weights = _get_path(cfg, "loss.sign_weights", None)
    sign_w_pos = float(_get_path(cfg, "loss.sign_weight_pos", 1.0))
    sign_w_neg = float(_get_path(cfg, "loss.sign_weight_neg", 2.0))
    k_m = _get_path(cfg, "loss.sign_k_m", None)
    if k_m is None:
        k_m = float(_get_path(cfg, "loss.sign_k_minus", 2.0))
    k_m = float(k_m)
    trunc_k = float(_get_path(cfg, "loss.trunc_k", 40.0))
    num_workers = int(_get_path(cfg, "data.num_workers", 0))
    prefetch_batches = int(_get_path(cfg, "data.prefetch_batches", max(2, num_workers * 2)))

    def _align_weights(weights: list[float] | tuple[float, ...], target: int) -> list[float]:
        w = list(weights)
        if not w:
            return [1.0] * target
        if len(w) < target:
            w.extend([w[-1]] * (target - len(w)))
        elif len(w) > target:
            w = w[:target]
        return w

    num_tiers = int(dataset.num_s) + 1
    reg_weights = _align_weights(list(reg_weights), num_tiers)
    if sign_bucket_weights is not None:
        sign_bucket_weights = tuple(_align_weights(list(sign_bucket_weights), num_tiers))
    if s_bucket_weights is not None:
        s_bucket_weights = tuple(_align_weights(list(s_bucket_weights), int(dataset.num_s)))

    metrics_path = run_dir / "metrics.csv"
    metrics_f = metrics_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(metrics_f)
    writer.writerow(["step", "loss", "l_sign", "l_reg", "l_bnd", "l_eik", "elapsed_sec"])

    prefetcher = None
    if num_workers > 0 and not overfit_batch:
        prefetcher = Prefetcher(
            _pos_worker,
            worker_args=(
                str(pos_path),
                str(split_voxel) if split_voxel is not None else None,
                bool(_get_path(cfg, "data.cache_grid", True)),
                batch_size,
                ratios,
                0,
                s_bucket_weights,
                s_bucket_tau,
                sign_balance,
            ),
            num_workers=num_workers,
            maxsize=prefetch_batches,
            seed=seed,
        )


    def _evaluate(split_id: int, batches: int) -> Dict[str, float]:
        sign_list = []
        reg_list = []
        bnd_list = []
        eik_list = []
        for _ in range(batches):
            batch = dataset.sample_batch(
                rng,
                batch_size,
                ratios,
                split_id=split_id,
                s_bucket_weights=s_bucket_weights,
                s_bucket_tau=s_bucket_tau,
                sign_balance=sign_balance,
            )
            p = torch.from_numpy(batch["p"]).to(device)
            s = torch.from_numpy(batch["s"]).to(device)
            source = torch.from_numpy(batch["source"]).to(device)
            tier = torch.from_numpy(batch.get("tier", np.zeros_like(batch["s"], dtype=np.uint8))).to(device)

            s_trunc = trunc_k * float(dataset.voxel_size)
            s_clip = torch.clamp(s, -s_trunc, s_trunc)
            p_norm = (p / l_ref).requires_grad_(True)
            y = s_clip / l_ref
            e_p = position_encoding(p_norm, k_p=k_p)
            morph_emb, _ = morph_encoder(morph_spec, device)
            s_pred = pos_model(e_p, morph_emb)

            mask_bnd = source == 0
            mask_nb = source == 1
            reg_w = torch.as_tensor(reg_weights, device=device, dtype=torch.float32)
            tier_idx = torch.clamp(tier.long(), 0, reg_w.numel() - 1)
            w_reg = reg_w[tier_idx]

            if torch.any(mask_nb):
                y_s = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
                margin = float(k_m) * float(dataset.voxel_size / l_ref)
                hinge = torch.relu(margin - y_s * s_pred)
                w_y = torch.where(y_s > 0, torch.full_like(y, sign_w_pos), torch.full_like(y, sign_w_neg))
                if sign_bucket_weights is not None:
                    w_sb = torch.as_tensor(sign_bucket_weights, device=device, dtype=torch.float32)
                    w_y = w_y * w_sb[tier_idx]
                w = w_y[mask_nb]
                sign_list.append((hinge[mask_nb] * w).sum().div(w.sum().clamp_min(1e-12)).item())

                reg_loss = torch.nn.functional.smooth_l1_loss(s_pred, y, reduction="none")
                w = w_reg[mask_nb]
                reg_list.append((reg_loss[mask_nb] * w).sum().div(w.sum().clamp_min(1e-12)).item())

            if torch.any(mask_bnd):
                bnd_loss = torch.nn.functional.smooth_l1_loss(s_pred, torch.zeros_like(s_pred), reduction="none")
                bnd_list.append(torch.mean(bnd_loss[mask_bnd]).item())

            v_norm = float(dataset.voxel_size / l_ref)
            k_eq_eval = float(k_eq_end)
            s_eq = k_eq_eval * v_norm
            grad = torch.autograd.grad(s_pred.sum(), p_norm, create_graph=False)[0]
            grad_norm = torch.linalg.norm(grad, dim=1)
            mask_eq = torch.abs(y) <= s_eq
            mask_far = ~mask_eq
            l_eq = torch.mean((grad_norm[mask_eq] - 1.0) ** 2) if torch.any(mask_eq) else torch.zeros((), device=device)
            l_far = torch.mean(torch.relu(grad_norm[mask_far] - 1.0) ** 2) if torch.any(mask_far) else torch.zeros((), device=device)
            eik_list.append((l_eq + l_far).item())
        return {
            "sign": float(np.mean(sign_list)) if sign_list else 0.0,
            "reg": float(np.mean(reg_list)) if reg_list else 0.0,
            "bnd": float(np.mean(bnd_list)) if bnd_list else 0.0,
            "eik": float(np.mean(eik_list)) if eik_list else 0.0,
        }

    fixed_batch = None
    if overfit_batch:
        fixed_batch = dataset.sample_batch(
            rng,
            batch_size,
            ratios,
            split_id=0,
            s_bucket_weights=s_bucket_weights,
            s_bucket_tau=s_bucket_tau,
            sign_balance=sign_balance,
        )
    start_time = time.time()
    sample_time_acc = 0.0
    compute_time_acc = 0.0
    step_time_acc = 0.0
    acc_steps = 0
    try:
        for step in range(1, max_steps + 1):
            t0 = time.time()
            if fixed_batch is not None:
                batch = fixed_batch
            elif prefetcher is not None:
                batch = prefetcher.get()
            else:
                batch = dataset.sample_batch(
                    rng,
                    batch_size,
                    ratios,
                    split_id=0,
                    s_bucket_weights=s_bucket_weights,
                    s_bucket_tau=s_bucket_tau,
                    sign_balance=sign_balance,
                )
            t1 = time.time()
            p = torch.from_numpy(batch["p"]).to(device)
            s = torch.from_numpy(batch["s"]).to(device)
            source = torch.from_numpy(batch["source"]).to(device)
            tag = torch.from_numpy(batch["tag"]).to(device)
            tier = torch.from_numpy(batch.get("tier", np.zeros_like(batch["s"], dtype=np.uint8))).to(device)

            s_trunc = trunc_k * float(dataset.voxel_size)
            s_clip = torch.clamp(s, -s_trunc, s_trunc)
            p_norm = p / l_ref
            y = s_clip / l_ref

            p_norm = p_norm.requires_grad_(True)
            e_p = position_encoding(p_norm, k_p=k_p)
            morph_emb, _ = morph_encoder(morph_spec, device)
            s_pred = pos_model(e_p, morph_emb)

            # Losses
            mask_bnd = source == 0
            mask_nb = source == 1
            reg_w = torch.as_tensor(reg_weights, device=device, dtype=torch.float32)
            tier_idx = torch.clamp(tier.long(), 0, reg_w.numel() - 1)

            # Non-boundary regression
            l_reg = torch.zeros((), device=device)
            if torch.any(mask_nb):
                reg_loss = torch.nn.functional.smooth_l1_loss(s_pred, y, reduction="none")
                w_reg_t = reg_w[tier_idx]
                w = w_reg_t[mask_nb]
                l_reg = torch.sum(reg_loss[mask_nb] * w) / torch.clamp(w.sum(), min=1e-12)

            # Boundary regression (to 0)
            l_bnd = torch.zeros((), device=device)
            if torch.any(mask_bnd):
                bnd_loss = torch.nn.functional.smooth_l1_loss(s_pred, torch.zeros_like(s_pred), reduction="none")
                l_bnd = torch.mean(bnd_loss[mask_bnd])

            # Sign hinge (non-boundary only, outside-heavy)
            l_sign = torch.zeros((), device=device)
            if torch.any(mask_nb):
                y_s = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
                v_norm = float(dataset.voxel_size / l_ref)
                margin = float(k_m) * v_norm
                hinge = torch.relu(margin - y_s * s_pred)
                w_y = torch.where(y_s > 0, torch.full_like(y, sign_w_pos), torch.full_like(y, sign_w_neg))
                if sign_bucket_weights is not None:
                    w_sb = torch.as_tensor(sign_bucket_weights, device=device, dtype=torch.float32)
                    w_y = w_y * w_sb[tier_idx]
                w = w_y[mask_nb]
                l_sign = torch.sum(hinge[mask_nb] * w) / torch.clamp(w.sum(), min=1e-12)

            # Eikonal: near-boundary equality + far-field inequality
            v_norm = float(dataset.voxel_size / l_ref)
            if eik_warmup > 0:
                w_eik_step = w_eik * min(float(step) / float(eik_warmup), 1.0)
            else:
                w_eik_step = w_eik

            if reg_warmup > 0:
                w_reg_step = w_reg * min(float(step) / float(reg_warmup), 1.0)
            else:
                w_reg_step = w_reg

            if k_eq_warmup > 0:
                t_eq = min(float(step) / float(k_eq_warmup), 1.0)
                k_eq = float(k_eq_start) + (float(k_eq_end) - float(k_eq_start)) * t_eq
            else:
                k_eq = float(k_eq_end)

            l_eik = torch.zeros((), device=device)
            if w_eik_step > 0:
                grad = torch.autograd.grad(s_pred.sum(), p_norm, create_graph=True)[0]
                grad_norm = torch.linalg.norm(grad, dim=1)
                s_eq = k_eq * v_norm
                mask_eq = torch.abs(y) <= s_eq
                mask_far = ~mask_eq
                l_eq = torch.mean((grad_norm[mask_eq] - 1.0) ** 2) if torch.any(mask_eq) else torch.zeros((), device=device)
                l_far = torch.mean(torch.relu(grad_norm[mask_far] - 1.0) ** 2) if torch.any(mask_far) else torch.zeros((), device=device)
                l_eik = l_eq + l_far

            loss = w_sign * l_sign + w_reg_step * l_reg + w_bnd * l_bnd + w_eik_step * l_eik

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2 = time.time()

            sample_time_acc += (t1 - t0)
            compute_time_acc += (t2 - t1)
            step_time_acc += (t2 - t0)
            acc_steps += 1

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                avg_sample = sample_time_acc / max(acc_steps, 1)
                avg_compute = compute_time_acc / max(acc_steps, 1)
                avg_step = step_time_acc / max(acc_steps, 1)
                with torch.no_grad():
                    pred_pos_ratio = float(torch.mean((s_pred >= 0).float()).item())
                    pred_mean = float(torch.mean(s_pred).item())
                logger.info(
                    f"[pos] step={step} loss={loss.item():.6f} "
                    f"l_sign={l_sign.item():.6f} l_reg={l_reg.item():.6f} l_bnd={l_bnd.item():.6f} "
                    f"l_eik={l_eik.item():.6f} w_reg={w_reg_step:.3f} w_eik={w_eik_step:.3f} k_eq={k_eq:.2f} "
                    f"pred_pos_ratio={pred_pos_ratio:.3f} pred_mean={pred_mean:.4f} "
                    f"t_sample={avg_sample:.4f}s t_compute={avg_compute:.4f}s t_step={avg_step:.4f}s"
                )
                writer.writerow([step, loss.item(), l_sign.item(), l_reg.item(), l_bnd.item(), l_eik.item(), elapsed])
                metrics_f.flush()
                sample_time_acc = 0.0
                compute_time_acc = 0.0
                step_time_acc = 0.0
                acc_steps = 0


            if step % val_interval == 0:
                pos_model.eval()
                with torch.enable_grad():
                    metrics = _evaluate(split_id=1, batches=val_batches)
                pos_model.train()
                logger.info(
                    f"[pos][val] step={step} sign={metrics['sign']:.6f} reg={metrics['reg']:.6f} "
                    f"bnd={metrics['bnd']:.6f} eik={metrics['eik']:.6f}"
                )

            if step % save_interval == 0:
                ckpt = {
                    "step": step,
                    "morph_encoder": morph_encoder.state_dict(),
                    "pos_model": pos_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, run_dir / f"checkpoint_{step:07d}.pt")
    finally:
        if prefetcher is not None:
            prefetcher.close()

    metrics_f.close()
    torch.save({
        "step": max_steps,
        "morph_encoder": morph_encoder.state_dict(),
        "pos_model": pos_model.state_dict(),
    }, run_dir / "checkpoint_final.pt")
