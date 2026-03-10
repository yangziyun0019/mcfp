"""Sample anchor-conditioned orientation batches from prepared orientation datasets.

This loader implements the batching strategy used by Orientation-SDF training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import h5py


class OrientationDataset:
    def __init__(self, path: Path, q_ref: Optional[np.ndarray] = None) -> None:
        self.path = Path(path)
        self.h5 = None
        self.q_ref = None if q_ref is None else np.asarray(q_ref, dtype=np.float32)

        with h5py.File(self.path, "r") as h5:
            self.anchor_pos = np.asarray(h5["/anchors/pos"], dtype=np.float32)
            self.anchor_s = np.asarray(h5["/anchors/s_v"], dtype=np.float32)
            self.anchor_s_norm = np.asarray(h5["/anchors/s_v_norm"], dtype=np.float32) if "/anchors/s_v_norm" in h5 else None
            self.anchor_split = np.asarray(h5["/anchors/split"], dtype=np.uint8) if "/anchors/split" in h5 else None
            self.anchor_voxel = np.asarray(h5["/anchors/voxel_id"], dtype=np.uint64)

            self.anchor_start = np.asarray(h5["/csr/anchor_start"], dtype=np.uint64)
            self.sample_index = np.asarray(h5["/csr/sample_index"], dtype=np.uint64)

        per_anchor = np.diff(self.anchor_start)
        self.valid_anchors = np.where(per_anchor > 0)[0]

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def sample_batch(
        self,
        rng: np.random.Generator,
        anchors: int,
        samples_per_anchor: int,
        ratio_bd: float = 0.2,
        ratio_sh: float = 0.4,
        ratio_gl: float = 0.4,
        *,
        n_phi: int = 10,
        tau_phi: float = 2.5,
        q_min: int = 8,
        r0: float = 0.5,
        r1: float = 0.8,
        phi_min: float = 1e-3,
        split_id: int = 0,
    ) -> Dict[str, np.ndarray]:
        self._open()
        h5 = self.h5

        valid = self.valid_anchors
        if self.anchor_split is not None:
            valid = valid[self.anchor_split[valid] == split_id]
        if valid.size == 0:
            raise RuntimeError("No anchors available for the requested split")

        anchor_ids = rng.choice(valid, size=anchors, replace=anchors > valid.size)

        # Prepare outputs
        p_list = []
        s_list = []
        anchor_list = []
        quat_list = []
        phi_list = []
        label_list = []
        method_list = []

        sample_quat = h5["/samples/quat"]
        sample_phi = h5["/samples/phi"]
        sample_label = h5["/samples/label"] if "/samples/label" in h5 else None
        sample_method = h5["/samples/method"] if "/samples/method" in h5 else None
        sample_bucket_r = h5["/samples/bucket_r"] if "/samples/bucket_r" in h5 else None

        def _choice(arr: np.ndarray, n: int) -> np.ndarray:
            if n <= 0 or arr.size == 0:
                return np.zeros((0,), dtype=np.int64)
            return rng.choice(arr, size=n, replace=n > arr.size)

        def _normalize_label(label: np.ndarray) -> np.ndarray:
            uniq = np.unique(label)
            if set(uniq.tolist()) <= {0, 1, 2}:
                # legacy: 1 reachable, 0 unreachable, 2 boundary
                out = np.where(label == 0, -1, 1).astype(np.int8)
            elif set(uniq.tolist()) <= {-1, 1}:
                out = label.astype(np.int8)
            else:
                out = np.where(label >= 0, 1, -1).astype(np.int8)
            return out

        def _bucket_phi(abs_phi: np.ndarray) -> np.ndarray:
            a_min = max(float(phi_min), 1e-6)
            a_max = np.pi
            eps = 1e-6
            u = np.log(abs_phi / a_min + eps) / np.log(a_max / a_min + eps)
            u = np.clip(u, 0.0, 1.0)
            b = 1 + np.floor(n_phi * u).astype(np.int64)
            b = np.clip(b, 1, n_phi)
            return b

        def _bucket_ref(quat: np.ndarray, bucket_pre: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if bucket_pre is not None:
                return bucket_pre.astype(np.int64)
            if self.q_ref is None:
                return None
            q = quat.astype(np.float32)
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)
            dots = np.abs(q @ self.q_ref.T)
            return 1 + np.argmax(dots, axis=1)

        def _sample_by_buckets(
            idx_pool: np.ndarray,
            phi_vals: np.ndarray,
            label_vals: np.ndarray,
            quat_vals: np.ndarray,
            bucket_pre: Optional[np.ndarray],
            target: int,
        ) -> np.ndarray:
            if target <= 0 or idx_pool.size == 0:
                return np.zeros((0,), dtype=np.int64)
            abs_phi = np.abs(phi_vals)
            b_phi = _bucket_phi(abs_phi)
            b_ref = _bucket_ref(quat_vals, bucket_pre)
            # active buckets
            active = []
            bucket_map = {}
            for k in range(1, n_phi + 1):
                mask = b_phi == k
                if np.any(mask):
                    active.append(k)
                    bucket_map[k] = mask
            if not active:
                return _choice(idx_pool, target)

            if target < len(active):
                weights = np.array([np.exp(-(k - 1) / max(tau_phi, 1e-6)) for k in active], dtype=np.float32)
                weights = weights / np.sum(weights)
                chosen = rng.choice(np.array(active), size=target, replace=False, p=weights)
                selected = []
                for k in chosen:
                    sel_idx = np.where(bucket_map[k])[0]
                    picked = _choice(sel_idx, 1)
                    if picked.size:
                        selected.append(idx_pool[picked[0]])
                return np.asarray(selected, dtype=np.int64)

            weights = np.array([np.exp(-(k - 1) / max(tau_phi, 1e-6)) for k in active], dtype=np.float32)
            weights = weights / np.sum(weights)
            raw = weights * target
            q_min_eff = min(int(q_min), max(1, target // len(active)))
            quota = np.maximum(q_min_eff, np.round(raw).astype(np.int64))
            total = int(np.sum(quota))
            if total != target:
                diff = target - total
                order = np.argsort(weights)[::-1] if diff > 0 else np.argsort(weights)
                i = 0
                while diff != 0 and i < order.size * 5:
                    k_idx = order[i % order.size]
                    if diff > 0:
                        quota[k_idx] += 1
                        diff -= 1
                    else:
                        if quota[k_idx] > 1:
                            quota[k_idx] -= 1
                            diff += 1
                    i += 1

            selected = []
            for k_idx, k in enumerate(active):
                qk = int(quota[k_idx])
                if qk <= 0:
                    continue
                mask_k = bucket_map[k]
                idx_k = np.where(mask_k)[0]
                if idx_k.size == 0:
                    continue
                # sign split
                lbl_k = label_vals[idx_k]
                neg_mask = lbl_k < 0
                pos_mask = ~neg_mask
                n_neg = int(np.round((r0 + (r1 - r0) * (k - 1) / max(n_phi - 1, 1)) * qk))
                n_pos = qk - n_neg
                idx_neg = idx_k[neg_mask]
                idx_pos = idx_k[pos_mask]
                if idx_neg.size == 0:
                    n_pos = qk
                    n_neg = 0
                if idx_pos.size == 0:
                    n_neg = qk
                    n_pos = 0

                def _sample_by_ref(idx_sub: np.ndarray, n_take: int) -> Sequence[int]:
                    if n_take <= 0 or idx_sub.size == 0:
                        return []
                    if b_ref is None:
                        return _choice(idx_sub, n_take).tolist()
                    b_sub = b_ref[idx_sub]
                    buckets = np.unique(b_sub)
                    if buckets.size == 0:
                        return _choice(idx_sub, n_take).tolist()
                    picks = []
                    if n_take <= buckets.size:
                        chosen = rng.choice(buckets, size=n_take, replace=False)
                        for b in chosen:
                            cand = idx_sub[b_sub == b]
                            pick = _choice(cand, 1)
                            if pick.size:
                                picks.append(pick[0])
                        return picks
                    base = n_take // buckets.size
                    rem = n_take % buckets.size
                    for b in buckets:
                        cand = idx_sub[b_sub == b]
                        if cand.size == 0:
                            continue
                        picks.extend(_choice(cand, base).tolist())
                    if rem > 0:
                        chosen = rng.choice(buckets, size=rem, replace=True)
                        for b in chosen:
                            cand = idx_sub[b_sub == b]
                            if cand.size == 0:
                                continue
                            picks.extend(_choice(cand, 1).tolist())
                    return picks

                selected.extend(_sample_by_ref(idx_pos, n_pos))
                selected.extend(_sample_by_ref(idx_neg, n_neg))

            if len(selected) < target:
                # fill from pool if not enough
                remaining = target - len(selected)
                selected.extend(_choice(np.arange(idx_pool.size), remaining).tolist())

            return np.asarray(idx_pool[selected[:target]], dtype=np.int64)

        for a in anchor_ids:
            start = int(self.anchor_start[a])
            end = int(self.anchor_start[a + 1])
            if end <= start:
                continue
            ids = self.sample_index[start:end]

            # Load method for filtering
            if sample_method is not None:
                methods = sample_method[ids]
                bd_idx = np.where(methods == 2)[0]
                sh_idx = np.where((methods == 0) | (methods == 1))[0]
                gl_idx = np.where(methods == 3)[0]
            else:
                methods = None
                bd_idx = np.zeros((0,), dtype=np.int64)
                sh_idx = np.arange(ids.shape[0])
                gl_idx = np.zeros((0,), dtype=np.int64)

            k_total = int(samples_per_anchor)
            k_bd = int(np.round(ratio_bd * k_total))
            k_sh = int(np.round(ratio_sh * k_total))
            k_gl = max(0, k_total - k_bd - k_sh)

            sel_bd = _choice(bd_idx, k_bd)
            sel_sh = _choice(sh_idx, k_sh)
            # fill missing from global
            k_gl = max(0, k_total - (sel_bd.size + sel_sh.size))

            if gl_idx.size > 0 and k_gl > 0:
                phi_gl = sample_phi[ids[gl_idx]]
                label_gl = _normalize_label(sample_label[ids[gl_idx]]) if sample_label is not None else np.ones_like(phi_gl, dtype=np.int8)
                quat_gl = sample_quat[ids[gl_idx]]
                bucket_gl = sample_bucket_r[ids[gl_idx]] if sample_bucket_r is not None else None
                sel_gl_rel = _sample_by_buckets(gl_idx, phi_gl, label_gl, quat_gl, bucket_gl, k_gl)
            else:
                sel_gl_rel = np.zeros((0,), dtype=np.int64)
                if k_gl > 0:
                    # fallback to shell/boundary if no global samples
                    pool = np.concatenate([sh_idx, bd_idx])
                    if pool.size > 0:
                        sel_gl_rel = _choice(pool, k_gl)

            idx = np.concatenate([sel_bd, sel_sh, sel_gl_rel])
            if idx.size == 0:
                continue
            sel = ids[idx]
            # h5py fancy indexing requires strictly increasing order (no duplicates).
            # Use unique+inverse to preserve sampling multiplicity.
            sel_unique, inv = np.unique(sel, return_inverse=True)

            quat = sample_quat[sel_unique][inv]
            phi = sample_phi[sel_unique][inv]
            label = sample_label[sel_unique][inv] if sample_label is not None else np.ones((sel.shape[0],), dtype=np.int8)
            method = sample_method[sel_unique][inv] if sample_method is not None else np.zeros((sel.shape[0],), dtype=np.uint8)

            if sample_label is not None:
                label = _normalize_label(label)
                phi = np.where(label > 0, np.abs(phi), -np.abs(phi))
                if sample_method is not None:
                    phi = np.where(method == 2, 0.0, phi)

            # broadcast anchor features
            p = np.repeat(self.anchor_pos[a][None, :], sel.shape[0], axis=0)
            if self.anchor_s_norm is not None:
                s = np.repeat(self.anchor_s_norm[a][None], sel.shape[0], axis=0)
            else:
                s = np.repeat(self.anchor_s[a][None], sel.shape[0], axis=0)
            s = np.abs(s)

            p_list.append(p)
            s_list.append(s)
            anchor_list.append(np.full((sel.shape[0],), a, dtype=np.int64))
            quat_list.append(quat)
            phi_list.append(phi)
            label_list.append(label)
            method_list.append(method)

        if not p_list:
            raise RuntimeError("No samples collected for orientation batch")

        return {
            "p": np.vstack(p_list).astype(np.float32),
            "s": np.concatenate(s_list).astype(np.float32),
            "quat": np.vstack(quat_list).astype(np.float32),
            "phi": np.concatenate(phi_list).astype(np.float32),
            "label": np.concatenate(label_list).astype(np.int8),
            "method": np.concatenate(method_list).astype(np.uint8),
            "anchor_id": np.concatenate(anchor_list).astype(np.int64),
        }
