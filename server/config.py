"""
Configuration for the Remote Text Encoder server.

Values can be overridden via environment variables or command-line flags.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    # ── Network ────────────────────────────────────────────────────────────────
    host: str = os.environ.get("RTE_HOST", "0.0.0.0")
    port: int = int(os.environ.get("RTE_PORT", "8288"))

    # ── Device ─────────────────────────────────────────────────────────────────
    # "auto" | "cuda" | "rocm" | "cpu"
    device: str = os.environ.get("RTE_DEVICE", "auto")

    # Torch dtype for model weights: "fp32" | "fp16" | "bf16" | "fp8e4m3" | "fp8e5m2" | "fp8"
    # fp8 variants store weights in FP8 and run compute in bf16 via autocast.
    # fp8 is an alias for fp8e4m3 (recommended).
    dtype: str = os.environ.get("RTE_DTYPE", "fp16")

    # ── Model cache ────────────────────────────────────────────────────────────
    # Directory where downloaded HF models are cached
    model_cache_dir: Optional[str] = os.environ.get("RTE_MODEL_CACHE_DIR", None)

    # Directory that is scanned for local models on startup and for /v1/models.
    # Sub-directories that contain a config.json are treated as model directories.
    # .safetensors files at the top level are treated as single-file models.
    # Defaults to a 'models' folder next to server.py.
    models_dir: str = os.environ.get(
        "RTE_MODELS_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    )

    # Keep models resident after first load (True) or unload when idle (False)
    keep_models_loaded: bool = os.environ.get("RTE_KEEP_MODELS_LOADED", "1") == "1"

    # Seconds of inactivity before unloading a model (0 = never unload)
    model_ttl_seconds: int = int(os.environ.get("RTE_MODEL_TTL", "300"))

    # ── Encoding ───────────────────────────────────────────────────────────────
    # Maximum token length accepted from clients
    max_token_length: int = int(os.environ.get("RTE_MAX_TOKEN_LENGTH", "77"))

    # ── Security ───────────────────────────────────────────────────────────────
    # Optional bearer token clients must send in Authorization header
    api_key: Optional[str] = os.environ.get("RTE_API_KEY", None)

    # Comma-separated list of allowed CORS origins ("*" to allow all)
    cors_origins: list = field(
        default_factory=lambda: os.environ.get("RTE_CORS_ORIGINS", "*").split(",")
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = os.environ.get("RTE_LOG_LEVEL", "INFO")


# Singleton used by the rest of the application
config = ServerConfig()
