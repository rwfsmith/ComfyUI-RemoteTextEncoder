"""
Device detection for CUDA and ROCm (HIP) backends.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def detect_device(preferred: str = "auto") -> torch.device:
    """
    Detect and return the best available compute device.

    preferred: "auto" | "cuda" | "rocm" | "cpu"
      - "rocm" is treated as a CUDA device index under PyTorch's HIP build.
    """
    if preferred == "cpu":
        logger.info("Device forced to CPU by configuration.")
        return torch.device("cpu")

    if preferred in ("cuda", "auto"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            backend = "ROCm/HIP" if _is_rocm() else "CUDA"
            logger.info(
                "Using %s device: %s (VRAM: %.1f GiB)",
                backend,
                props.name,
                props.total_memory / 1024**3,
            )
            return device

    if preferred == "rocm":
        # ROCm uses the CUDA runtime under PyTorch; same check applies.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "Using ROCm/HIP device: %s",
                torch.cuda.get_device_name(device),
            )
            return device
        logger.warning("ROCm requested but no CUDA-compatible (HIP) device found; falling back to CPU.")
        return torch.device("cpu")

    logger.warning("No GPU device available; falling back to CPU.")
    return torch.device("cpu")


def _is_rocm() -> bool:
    """Return True if the PyTorch build targets ROCm/HIP."""
    try:
        return torch.version.hip is not None
    except AttributeError:
        return False


def device_info() -> dict:
    """Return a human-readable dict describing the active device."""
    if not torch.cuda.is_available():
        return {"backend": "cpu", "name": "CPU", "vram_gib": None}

    props = torch.cuda.get_device_properties(0)
    return {
        "backend": "rocm" if _is_rocm() else "cuda",
        "name": props.name,
        "vram_gib": round(props.total_memory / 1024**3, 2),
        "cuda_capability": f"{props.major}.{props.minor}" if not _is_rocm() else "N/A",
    }
