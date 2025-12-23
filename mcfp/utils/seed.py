# mcfp/utils/seed.py

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class RNGState:
    """Container for reproducible RNG state snapshots.

    Fields
    ------
    py : Python 'random' module state.
    np : NumPy RNG state.
    torch_cpu : PyTorch CPU RNG state.
    torch_cuda : PyTorch CUDA RNG states (one per device), or None if CUDA unavailable.
    """

    py: object
    np: tuple
    torch_cpu: torch.Tensor
    torch_cuda: Optional[list[torch.Tensor]] = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python/NumPy/PyTorch (CPU & CUDA).

    Args:
        seed: Global seed.
        deterministic: If True, enable deterministic behavior in PyTorch backend.
            This improves reproducibility at the cost of potential performance.

    Notes:
        - Determinism is best-effort. Some ops may still be non-deterministic depending on
          CUDA/PyTorch versions and specific kernels used.
        - If you need strict determinism, combine:
            (1) deterministic=True
            (2) fixed software versions
            (3) fixed hardware + driver
    """
    seed = int(seed)

    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _configure_torch_determinism(deterministic=deterministic)


def _configure_torch_determinism(deterministic: bool) -> None:
    """Configure PyTorch backend flags for (non-)deterministic execution."""
    # cuDNN flags
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)

    # PyTorch deterministic algorithms flag (may raise on unsupported ops)
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        # Some PyTorch builds do not support this API fully.
        pass

    # For cuBLAS determinism (recommended by PyTorch docs).
    # Must be set before CUDA context initialization to be fully effective.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    """Initialize RNG states for a DataLoader worker process.

    Args:
        worker_id: Worker index assigned by PyTorch DataLoader.

    Notes:
        - PyTorch sets a base seed for each worker via torch.initial_seed().
          We derive a 32-bit seed from it and seed Python/NumPy/PyTorch accordingly.
        - This function should be passed to DataLoader(worker_init_fn=seed_worker).
    """
    # torch.initial_seed() is different across workers when DataLoader uses a Generator.
    worker_seed = int(torch.initial_seed()) % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def get_rng_state() -> RNGState:
    """Capture current RNG states for reproducible resume."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu = torch.get_rng_state()

    torch_cuda: Optional[list[torch.Tensor]] = None
    if torch.cuda.is_available():
        torch_cuda = torch.cuda.get_rng_state_all()

    return RNGState(
        py=py_state,
        np=np_state,
        torch_cpu=torch_cpu,
        torch_cuda=torch_cuda,
    )


def set_rng_state(state: RNGState) -> None:
    """Restore RNG states from a previously captured snapshot.

    Args:
        state: RNGState returned by get_rng_state().

    Notes:
        - For CUDA, restoration requires the same number of visible devices.
        - If CUDA is unavailable, torch_cuda is ignored.
    """
    random.setstate(state.py)
    np.random.set_state(state.np)
    torch.set_rng_state(state.torch_cpu)

    if torch.cuda.is_available() and state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def rng_state_to_dict(state: RNGState) -> Dict[str, Any]:
    """Serialize RNGState into a checkpoint-friendly dict."""
    out: Dict[str, Any] = {
        "py": state.py,
        "np": state.np,
        "torch_cpu": state.torch_cpu,
        "torch_cuda": state.torch_cuda,
    }
    return out


def rng_state_from_dict(d: Dict[str, Any]) -> RNGState:
    """Deserialize RNGState from a checkpoint dict."""
    return RNGState(
        py=d["py"],
        np=d["np"],
        torch_cpu=d["torch_cpu"],
        torch_cuda=d.get("torch_cuda", None),
    )
