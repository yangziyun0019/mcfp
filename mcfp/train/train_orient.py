"""Train the Orientation-SDF model on prepared anchor-conditioned datasets.

This module contains the full optimization and validation loop for orientation training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import csv
import time
import math
import queue as pyqueue

import numpy as np
import torch
from torch import nn
import h5py

from mcfp.data.orient_dataset import OrientationDataset
from mcfp.data.morph_spec_io import load_morphology_spec
from mcfp.models.encodings import position_encoding, ReferenceQuaternionEncoder, sample_ref_quaternions
from mcfp.models.morph_encoder import MorphologyEncoder
from mcfp.models.orient_sdf import OrientationSDFModel
from mcfp.models.pos_sdf import PositionSDFModel
from mcfp.utils.quat import exp_quat, quat_mul
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


def _load_q_ref_from_h5(path: Path) -> Optional[np.ndarray]:
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


def _load_q_ref(cfg: Any, orient_path: Path, repo_root: Path, k_r: int) -> np.ndarray:
    q_ref = None
    use_dataset = bool(_get_path(cfg, "data.use_dataset_q_ref", True))
    if use_dataset:
        q_ref = _load_q_ref_from_h5(orient_path)

    q_ref_path = _get_path(cfg, "data.q_ref_path", None)
    if q_ref is None and q_ref_path:
        q_ref_path = _resolve_path(q_ref_path, repo_root)
        if q_ref_path.suffix in (".npy", ".npz"):
            data = np.load(q_ref_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "q_ref" in data:
                    q_ref = np.asarray(data["q_ref"], dtype=np.float32)
                else:
                    # fall back to first array
                    q_ref = np.asarray(data[list(data.files)[0]], dtype=np.float32)
            else:
                q_ref = np.asarray(data, dtype=np.float32)
        elif q_ref_path.suffix in (".h5", ".hdf5"):
            q_ref = _load_q_ref_from_h5(q_ref_path)

    if q_ref is None:
        seed = int(_get_path(cfg, "data.q_ref_seed", 20260126))
        q_ref = sample_ref_quaternions(k_r, seed=seed).cpu().numpy().astype(np.float32)
    return q_ref


def _orient_worker(
    queue,
    stop_event,
    seed: int,
    orient_path: str,
    anchors: int,
    samples_per_anchor: int,
    ratio_bd: float,
    ratio_sh: float,
    ratio_gl: float,
    bucket_cfg: Dict[str, float],
    q_ref: Optional[np.ndarray],
    split_id: int,
) -> None:
    rng = np.random.default_rng(int(seed))
    dataset = OrientationDataset(Path(orient_path), q_ref=q_ref)
    try:
        while not stop_event.is_set():
            batch = dataset.sample_batch(
                rng,
                anchors,
                samples_per_anchor,
                ratio_bd,
                ratio_sh,
                ratio_gl,
                n_phi=int(bucket_cfg["n_phi"]),
                tau_phi=float(bucket_cfg["tau_phi"]),
                q_min=int(bucket_cfg["q_min"]),
                r0=float(bucket_cfg["r0"]),
                r1=float(bucket_cfg["r1"]),
                phi_min=float(bucket_cfg["phi_min"]),
                split_id=split_id,
            )
            while not stop_event.is_set():
                try:
                    queue.put(batch, timeout=0.5)
                    break
                except pyqueue.Full:
                    continue
    finally:
        dataset.close()


def train_orient(cfg: Any, logger) -> None:
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    run_dir = _resolve_path(_get_path(cfg, "paths.run_dir", "runs/orient_sdf/exp001"), repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))
    seed = int(_get_path(cfg, "run.seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Data
    orient_path = _resolve_path(_get_path(cfg, "paths.orientation_h5"), repo_root)

    # Morphology
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    # Models
    k_p = int(_get_path(cfg, "model.position.k_p", 10))
    k_r = int(_get_path(cfg, "model.orientation.k_r", 64))
    q_ref = _load_q_ref(cfg, orient_path, repo_root, k_r)
    k_r = int(q_ref.shape[0])
    ref_encoder = ReferenceQuaternionEncoder(q_ref=torch.from_numpy(q_ref)).to(device)

    dataset = OrientationDataset(orient_path, q_ref=q_ref)

    morph_cfg = _get_path(cfg, "model.morph", None)
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)

    orient_model = OrientationSDFModel(
        orient_in_dim=2 * k_r,
        context_dim=int(_get_path(cfg, "model.orientation.context_dim", 128)),
        hidden_dim=int(_get_path(cfg, "model.orientation.hidden_dim", 256)),
        num_layers=int(_get_path(cfg, "model.orientation.num_layers", 4)),
        experts=int(_get_path(cfg, "model.orientation.experts", 4)),
        cond_in_dim=int(_get(morph_cfg, "d_model", 256)) + (3 + 2 * 3 * k_p) + 1,
    ).to(device)

    # Optional position model for B-2
    pos_ckpt = _get_path(cfg, "paths.position_checkpoint", None)
    pos_model = None
    if pos_ckpt:
        pos_model = PositionSDFModel(
            in_dim=3 + 2 * 3 * k_p,
            hidden_dim=int(_get_path(cfg, "model.position.hidden_dim", 256)),
            num_layers=int(_get_path(cfg, "model.position.num_layers", 5)),
            w0_first=float(_get_path(cfg, "model.position.w0_first", 30.0)),
            w0=float(_get_path(cfg, "model.position.w0", 1.0)),
            cond_dim=int(_get(morph_cfg, "d_model", 256)),
        ).to(device)
        ckpt = torch.load(_resolve_path(pos_ckpt, repo_root), map_location=device)
        pos_model.load_state_dict(ckpt["pos_model"])
        pos_model.eval()

    optimizer = torch.optim.Adam(
        list(morph_encoder.parameters()) + list(orient_model.parameters()),
        lr=float(_get_path(cfg, "optim.lr", 2e-4)),
    )

    # Training settings
    steps_b1 = int(_get_path(cfg, "run.steps_b1", 50000))
    steps_b2 = int(_get_path(cfg, "run.steps_b2", 10000))
    log_interval = int(_get_path(cfg, "run.log_interval", 200))
    save_interval = int(_get_path(cfg, "run.save_interval", 2000))
    val_interval = int(_get_path(cfg, "run.val_interval", 2000))
    val_batches = int(_get_path(cfg, "run.val_batches", 2))

    anchors = int(_get_path(cfg, "data.anchors_per_step", _get_path(cfg, "data.anchors", 64)))
    samples_per_anchor = int(_get_path(cfg, "data.samples_per_anchor", 64))
    ratio_bd = float(_get_path(cfg, "data.ratio_bd", 0.2))
    ratio_sh = float(_get_path(cfg, "data.ratio_sh", 0.4))
    ratio_gl = float(_get_path(cfg, "data.ratio_gl", 0.4))
    bucket_cfg = {
        "n_phi": int(_get_path(cfg, "data.n_phi", 10)),
        "tau_phi": float(_get_path(cfg, "data.tau_phi", 2.5)),
        "q_min": int(_get_path(cfg, "data.q_min", 8)),
        "r0": float(_get_path(cfg, "data.r0", 0.5)),
        "r1": float(_get_path(cfg, "data.r1", 0.8)),
        "phi_min": float(_get_path(cfg, "data.phi_min", 1e-3)),
    }
    num_workers = int(_get_path(cfg, "data.num_workers", 0))
    prefetch_batches = int(_get_path(cfg, "data.prefetch_batches", max(2, num_workers * 2)))

    tau0 = float(_get_path(cfg, "model.orientation.tau0", 0.2))
    tau1 = float(_get_path(cfg, "model.orientation.tau1", 0.05))
    tau_steps = int(_get_path(cfg, "model.orientation.tau_steps", steps_b1))

    sign_thr = float(_get_path(cfg, "loss.sign_thr", 0.35))
    sign_margin = float(_get_path(cfg, "loss.sign_margin", 0.10))
    w_eik = float(_get_path(cfg, "loss.eikonal_weight", 0.1))
    eik_warm_frac = float(_get_path(cfg, "loss.eikonal_warmup_frac", 0.15))
    eik_ramp_frac = float(_get_path(cfg, "loss.eikonal_ramp_frac", 0.55))
    phi_eq_start = float(_get_path(cfg, "loss.phi_eq_start", bucket_cfg["phi_min"]))
    phi_eq_end = float(_get_path(cfg, "loss.phi_eq_end", 0.5))
    w_sign = float(_get_path(cfg, "loss.sign_weight", 0.5))
    w_bal = float(_get_path(cfg, "loss.balance_weight", 0.1))
    ent_max = float(_get_path(cfg, "loss.entropy_weight", 0.01))
    w_bd = float(_get_path(cfg, "loss.weight_boundary", 2.0))
    w_gl = float(_get_path(cfg, "loss.weight_global", 0.25))

    metrics_path = run_dir / "metrics.csv"
    metrics_f = metrics_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(metrics_f)
    writer.writerow(["step", "loss", "l_phi", "l_eik", "l_sign", "l_bal", "l_ent", "elapsed_sec"])

    prefetcher = None
    if num_workers > 0:
        prefetcher = Prefetcher(
            _orient_worker,
            worker_args=(
                str(orient_path),
                anchors,
                samples_per_anchor,
                ratio_bd,
                ratio_sh,
                ratio_gl,
                bucket_cfg,
                q_ref,
                0,
            ),
            num_workers=num_workers,
            maxsize=prefetch_batches,
            seed=seed,
        )

    def _tau(step: int) -> float:
        if tau_steps <= 0:
            return tau1
        ratio = min(1.0, step / tau_steps)
        return tau0 * ((tau1 / tau0) ** ratio)

    def _entropy_weight(step: int, total: int) -> float:
        warm = int(total * 0.2)
        if step <= warm:
            return 0.0
        ratio = min(1.0, (step - warm) / max(total - warm, 1))
        return ent_max * ratio

    def _eik_weight(step: int, total: int) -> float:
        start = int(total * eik_warm_frac)
        end = int(total * (eik_warm_frac + eik_ramp_frac))
        if step <= start:
            return 0.0
        if step >= end:
            return w_eik
        ratio = (step - start) / max(end - start, 1)
        return w_eik * ratio

    def _phi_eq(step: int, total: int) -> float:
        start = int(total * eik_warm_frac)
        end = int(total * (eik_warm_frac + eik_ramp_frac))
        if step <= start:
            return phi_eq_start
        if step >= end:
            return phi_eq_end
        ratio = (step - start) / max(end - start, 1)
        return phi_eq_start + (phi_eq_end - phi_eq_start) * ratio


    def _evaluate(batches: int, step: int) -> Dict[str, float]:
        mae_list = []
        bnd_list = []
        sign_list = []
        eik_list = []
        for _ in range(batches):
            batch = dataset.sample_batch(
                rng,
                anchors,
                samples_per_anchor,
                ratio_bd,
                ratio_sh,
                ratio_gl,
                n_phi=bucket_cfg["n_phi"],
                tau_phi=bucket_cfg["tau_phi"],
                q_min=bucket_cfg["q_min"],
                r0=bucket_cfg["r0"],
                r1=bucket_cfg["r1"],
                phi_min=bucket_cfg["phi_min"],
                split_id=1,
            )
            p = torch.from_numpy(batch["p"]).to(device)
            s = torch.from_numpy(batch["s"]).to(device)
            quat = torch.from_numpy(batch["quat"]).to(device)
            phi = torch.from_numpy(batch["phi"]).to(device)
            method = torch.from_numpy(batch["method"]).to(device)
            label = torch.from_numpy(batch["label"]).to(device)

            p_norm = p / l_ref
            e_p = position_encoding(p_norm, k_p=k_p)
            morph_emb, _ = morph_encoder(morph_spec, device)
            s_in = torch.clamp(s, -1.0, 1.0)

            cond = torch.cat([morph_emb.expand(p.shape[0], -1), e_p, s_in.unsqueeze(-1)], dim=1)
            e_r = ref_encoder(quat)
            u_raw, u = orient_model(e_r, cond, tau=_tau(step))
            phi_raw = math.pi * u
            phi_pred = math.pi * torch.tanh(phi_raw / math.pi)

            mae_list.append(torch.mean(torch.abs(phi_pred - phi)).item())
            if torch.any(method == 2):
                bnd_list.append(torch.mean(torch.abs(phi_pred[method == 2])).item())
            mask = method != 2
            if torch.any(mask):
                sign_acc = torch.mean(((phi_pred[mask] >= 0) == (label[mask] > 0)).float()).item()
                sign_list.append(sign_acc)

            # Eikonal
            mask_strong = (method == 0) | (method == 1) | (method == 2) | (method == 3)
            if torch.any(mask_strong):
                delta = torch.zeros((quat.shape[0], 3), device=device, requires_grad=True)
                q_delta = exp_quat(delta)
                q_new = quat_mul(q_delta, quat)
                e_r2 = ref_encoder(q_new)
                u_raw2, u2 = orient_model(e_r2, cond, tau=_tau(step))
                phi_raw2 = math.pi * u2
                g = torch.autograd.grad(phi_raw2[mask_strong].sum(), delta, create_graph=True)[0]
                grad_norm = torch.linalg.norm(g[mask_strong], dim=1)
                eik_list.append(torch.mean(torch.abs(grad_norm - 1.0)).item())

        return {
            "mae": float(np.mean(mae_list)) if mae_list else 0.0,
            "bnd": float(np.mean(bnd_list)) if bnd_list else 0.0,
            "sign": float(np.mean(sign_list)) if sign_list else 0.0,
            "eik": float(np.mean(eik_list)) if eik_list else 0.0,
        }

    start_time = time.time()
    total_steps = steps_b1 + steps_b2

    try:
        for step in range(1, total_steps + 1):
            phase = "B1" if step <= steps_b1 else "B2"
            tau = _tau(step)

            if prefetcher is not None:
                batch = prefetcher.get()
            else:
                batch = dataset.sample_batch(
                    rng,
                    anchors,
                    samples_per_anchor,
                    ratio_bd,
                    ratio_sh,
                    ratio_gl,
                    n_phi=bucket_cfg["n_phi"],
                    tau_phi=bucket_cfg["tau_phi"],
                    q_min=bucket_cfg["q_min"],
                    r0=bucket_cfg["r0"],
                    r1=bucket_cfg["r1"],
                    phi_min=bucket_cfg["phi_min"],
                    split_id=0,
                )
            p = torch.from_numpy(batch["p"]).to(device)
            s = torch.from_numpy(batch["s"]).to(device)
            quat = torch.from_numpy(batch["quat"]).to(device)
            phi = torch.from_numpy(batch["phi"]).to(device)
            method = torch.from_numpy(batch["method"]).to(device)
            anchor_id = torch.from_numpy(batch["anchor_id"]).to(device)

            p_norm = p / l_ref
            e_p = position_encoding(p_norm, k_p=k_p)
            morph_emb, _ = morph_encoder(morph_spec, device)

            # scheduled teacher forcing for s_in
            if phase == "B1" or pos_model is None:
                s_in = s
            else:
                with torch.no_grad():
                    e_p_pos = position_encoding(p_norm, k_p=k_p)
                    s_pred = pos_model(e_p_pos, morph_emb)
                t = step - steps_b1
                alpha = max(0.0, 1.0 - t / max(steps_b2, 1))
                noise = torch.randn_like(s) * 0.01
                s_in = torch.clamp(alpha * s + (1 - alpha) * s_pred + noise, -1.0, 1.0)

            s_in = torch.clamp(s_in, -1.0, 1.0)

            cond = torch.cat([morph_emb.expand(p.shape[0], -1), e_p, s_in.unsqueeze(-1)], dim=1)
            e_r = ref_encoder(quat)
            u_raw, u = orient_model(e_r, cond, tau=tau)
            phi_raw = math.pi * u
            phi_pred = math.pi * torch.tanh(phi_raw / math.pi)

            # Losses
            weight = torch.ones_like(phi)
            weight = torch.where(method == 2, torch.full_like(weight, w_bd), weight)
            weight = torch.where(method == 3, torch.full_like(weight, w_gl), weight)
            l_phi = torch.mean(weight * torch.nn.functional.smooth_l1_loss(phi_pred, phi, reduction="none"))

            # Eikonal on strong methods
            mask_strong = (method == 0) | (method == 1) | (method == 2) | (method == 3)
            if torch.any(mask_strong):
                delta = torch.zeros((quat.shape[0], 3), device=device, requires_grad=True)
                q_delta = exp_quat(delta)
                q_new = quat_mul(q_delta, quat)
                e_r2 = ref_encoder(q_new)
                u_raw2, u2 = orient_model(e_r2, cond, tau=tau)
                phi_raw2 = math.pi * u2
                g = torch.autograd.grad(phi_raw2[mask_strong].sum(), delta, create_graph=True)[0]
                grad_norm = torch.linalg.norm(g[mask_strong], dim=1)
                phi_eq = _phi_eq(step, total_steps)
                phi_abs = torch.abs(phi[mask_strong])
                near_mask = phi_abs <= phi_eq
                far_mask = ~near_mask
                l_eik = torch.zeros((), device=device)
                if torch.any(near_mask):
                    l_eik = l_eik + torch.mean((grad_norm[near_mask] - 1.0) ** 2)
                if torch.any(far_mask):
                    l_eik = l_eik + torch.mean(torch.relu(grad_norm[far_mask] - 1.0) ** 2)
            else:
                l_eik = torch.zeros((), device=device)

            sign_thr_pos = abs(sign_thr)
            mask_sign = phi <= -sign_thr_pos
            if torch.any(mask_sign):
                l_sign = torch.mean(torch.relu(phi_pred[mask_sign] + sign_margin))
            else:
                l_sign = torch.zeros((), device=device)

            # MoE balance + entropy
            alpha = torch.softmax(u_raw / tau, dim=1)
            # anchor-level average
            unique_anchor = torch.unique(anchor_id)
            alpha_anchor = []
            for a in unique_anchor:
                alpha_anchor.append(torch.mean(alpha[anchor_id == a], dim=0))
            alpha_anchor = torch.stack(alpha_anchor, dim=0)
            alpha_mean = torch.mean(alpha_anchor, dim=0)
            l_bal = torch.mean((alpha_mean - 1.0 / alpha_mean.numel()) ** 2)
            l_ent = torch.mean(-torch.sum(alpha * torch.log(alpha + 1e-8), dim=1))

            ent_w = _entropy_weight(step, total_steps)
            eik_w = _eik_weight(step, total_steps)
            loss = l_phi + eik_w * l_eik + w_sign * l_sign + w_bal * l_bal + ent_w * l_ent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"[orient] step={step} loss={loss.item():.6f} l_phi={l_phi.item():.6f} "
                    f"l_eik={l_eik.item():.6f} l_sign={l_sign.item():.6f} l_bal={l_bal.item():.6f} l_ent={l_ent.item():.6f} "
                    f"w_eik={eik_w:.4f} phi_eq={_phi_eq(step, total_steps):.3f}"
                )
                writer.writerow([step, loss.item(), l_phi.item(), l_eik.item(), l_sign.item(), l_bal.item(), l_ent.item(), elapsed])
                metrics_f.flush()


            if step % val_interval == 0:
                orient_model.eval()
                with torch.enable_grad():
                    metrics = _evaluate(val_batches, step)
                orient_model.train()
                logger.info(
                    f"[orient][val] step={step} mae={metrics['mae']:.6f} bnd={metrics['bnd']:.6f} "
                    f"sign={metrics['sign']:.4f} eik={metrics['eik']:.6f}"
                )

            if step % save_interval == 0:
                ckpt = {
                    "step": step,
                    "morph_encoder": morph_encoder.state_dict(),
                    "orient_model": orient_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, run_dir / f"checkpoint_{step:07d}.pt")
    finally:
        if prefetcher is not None:
            prefetcher.close()

    metrics_f.close()
    torch.save({
        "step": total_steps,
        "morph_encoder": morph_encoder.state_dict(),
        "orient_model": orient_model.state_dict(),
    }, run_dir / "checkpoint_final.pt")
