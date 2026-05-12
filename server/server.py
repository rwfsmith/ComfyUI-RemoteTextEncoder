"""
Remote Text Encoder Server for ComfyUI
=======================================

Exposes a REST API that ComfyUI (or any client) can call to run CLIP / T5 / Gemma
text-encoding on a machine with a CUDA or ROCm GPU.

Endpoints
---------
GET  /                       Health-check / server info
GET  /info                   Device & configuration details
GET  /models                 List currently loaded models
POST /encode                 Encode text with a specified model
DELETE /models/{model_id}    Unload a specific model from VRAM
DELETE /models               Unload all models
POST /comfy/encode           ComfyUI-compatible single-encoder endpoint
POST /comfy/encode/ltxv      LTX-Video 2.3 dual (T5 + Gemma) encode endpoint
POST /comfy/encode/gemma_raw Gemma all-layers encode (for local projection)

OpenAI-compatible
-----------------
GET  /v1/models              List available models (OpenAI format)
POST /v1/embeddings          Create embeddings (OpenAI format)
"""

import argparse
import base64
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import ServerConfig, config
from device import detect_device, device_info
from model_manager import KNOWN_MODELS, ModelManager, scan_local_models

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rte.server")

# ── App bootstrap ─────────────────────────────────────────────────────────────

device = detect_device(config.device)
manager = ModelManager(config, device)

app = FastAPI(
    title="Remote Text Encoder",
    description="CLIP / T5 text-encoder server for ComfyUI (CUDA & ROCm)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth dependency ───────────────────────────────────────────────────────────


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not config.api_key:
        return  # auth disabled
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


AUTH = Depends(verify_api_key)

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class EncodeRequest(BaseModel):
    model: str = Field(
        ...,
        description="Hugging Face repo-id or local path of the text encoder, "
                    "e.g. 'openai/clip-vit-large-patch14'",
        examples=["openai/clip-vit-large-patch14"],
    )
    texts: List[str] = Field(
        ...,
        description="List of prompt strings to encode.",
        examples=[["a photo of a cat", "a photo of a dog"]],
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum token length (defaults to server config value).",
    )
    return_pooled: bool = Field(
        default=False,
        description="Also return pooled / mean-pooled embeddings.",
    )


class EncodeResponse(BaseModel):
    model: str
    shape: List[int]
    dtype: str = "float32"
    # Base64-encoded raw bytes of the float32 numpy array (C-contiguous)
    embeddings_b64: str
    pooled_b64: Optional[str] = None


class ServerInfoResponse(BaseModel):
    version: str
    device: Dict[str, Any]
    config: Dict[str, Any]


# ── OpenAI-compatible schemas ───────────────────────────────────────────────────────────────────


class OAIModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "remote-text-encoder"
    # Extra metadata not in the official spec but useful for clients
    family: Optional[str] = None
    loaded: bool = False
    # Full path or HF repo-id that the server can load; may differ from id
    path: str = ""


class OAIModelList(BaseModel):
    object: str = "list"
    data: List[OAIModelObject]


class OAIEmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class OAIEmbeddingsRequest(BaseModel):
    model: str = Field(
        ...,
        description="HF repo-id or local path of the text encoder.",
        examples=["openai/clip-vit-large-patch14"],
    )
    input: Union[str, List[str]] = Field(
        ...,
        description="String or list of strings to embed.",
    )
    # Optional – ignored for compatibility with callers that pass them
    encoding_format: Optional[str] = Field(default="float")
    dimensions: Optional[int] = None
    user: Optional[str] = None


class OAIEmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[OAIEmbeddingObject]
    model: str
    usage: Dict[str, int]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _arr_to_b64(arr: np.ndarray) -> str:
    """Encode a numpy array to a base64 string (raw bytes, float32, C-order)."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _short_name(model_id: str) -> str:
    """
    Return a short display name for a model_id.

    - Full path to a .safetensors file  → filename without extension
    - Full path to a directory           → directory name
    - HF repo-id                         → returned unchanged
    """
    from pathlib import Path as _Path
    p = _Path(model_id)
    if p.suffix.lower() == ".safetensors":
        return p.stem
    if p.is_dir() or (p.parent != _Path(".") and not "/" in model_id and not model_id.startswith("http")):
        return p.name
    return model_id


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "Remote Text Encoder"}


@app.get("/info", response_model=ServerInfoResponse, tags=["health"])
async def info(_: None = AUTH):
    return ServerInfoResponse(
        version="1.0.0",
        device=device_info(),
        config={
            "dtype": config.dtype,
            "max_token_length": config.max_token_length,
            "keep_models_loaded": config.keep_models_loaded,
            "model_ttl_seconds": config.model_ttl_seconds,
            "auth_enabled": bool(config.api_key),
        },
    )


@app.get("/models", tags=["models"])
async def list_models(_: None = AUTH):
    return {"loaded": manager.loaded_models()}


@app.delete("/models/{model_id:path}", tags=["models"])
async def unload_model(model_id: str, _: None = AUTH):
    found = manager.unload(model_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' is not loaded.")
    return {"unloaded": model_id}


@app.delete("/models", tags=["models"])
async def unload_all_models(_: None = AUTH):
    manager.unload_all()
    return {"status": "all models unloaded"}


@app.post("/encode", response_model=EncodeResponse, tags=["encode"])
async def encode(req: EncodeRequest, _: None = AUTH):
    """
    Encode a list of texts using the specified model.

    Returns base64-encoded raw float32 bytes for the embeddings tensor
    with shape [batch, tokens, hidden_dim].

    **Decoding on the client side (Python):**
    ```python
    import base64, numpy as np
    data = base64.b64decode(response["embeddings_b64"])
    embeddings = np.frombuffer(data, dtype=np.float32).reshape(response["shape"])
    ```
    """
    if not req.texts:
        raise HTTPException(status_code=422, detail="'texts' must not be empty.")

    try:
        result = manager.encode(
            model_id=req.model,
            texts=req.texts,
            max_length=req.max_length,
            return_pooled=req.return_pooled,
        )
    except Exception as exc:
        logger.exception("Encoding failed for model %s", req.model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    embeddings: np.ndarray = result["embeddings"]
    pooled_b64 = _arr_to_b64(result["pooled"]) if "pooled" in result else None

    return EncodeResponse(
        model=req.model,
        shape=list(embeddings.shape),
        embeddings_b64=_arr_to_b64(embeddings),
        pooled_b64=pooled_b64,
    )


# ── ComfyUI-compatible batch endpoint ────────────────────────────────────────
# ComfyUI's remote CLIP node (CLIPTextEncodeRemote) POSTs JSON of the form:
#   { "model_name": "...", "text": "...", "clip_skip": N }
# and expects:
#   { "cond": [[B64], {"pooled_output": [B64]}] }
# This endpoint bridges that protocol.


class ComfyEncodeRequest(BaseModel):
    model_name: str
    text: str
    clip_skip: int = Field(default=1, ge=1, le=12)


@app.post("/comfy/encode", tags=["comfy"])
async def comfy_encode(req: ComfyEncodeRequest, _: None = AUTH):
    """
    ComfyUI-compatible text-encode endpoint.
    Compatible with the CLIPTextEncodeRemote custom node protocol.
    """
    try:
        result = manager.encode(
            model_id=req.model_name,
            texts=[req.text],
            return_pooled=True,
        )
    except Exception as exc:
        logger.exception("ComfyUI encode failed for model %s", req.model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    embeddings: np.ndarray = result["embeddings"]     # [1, T, D]
    pooled: np.ndarray = result.get("pooled", embeddings.mean(axis=1))  # [1, D]

    return {
        "cond": [
            _arr_to_b64(embeddings[0]),  # [T, D]  – strip batch dim
            {"pooled_output": _arr_to_b64(pooled[0])},  # [D]
        ],
        "shape": list(embeddings.shape[1:]),
        "pooled_shape": list(pooled.shape[1:]),
    }


# ── LTX-Video 2.3 dual-encoder endpoint ──────────────────────────────────────
# LTX-Video 2.3 conditions its DiT on two encoders simultaneously:
#   • T5-XXL       – primary sequence conditioning, 4096-dim
#   • Gemma-3-12B  – secondary conditioning, concatenated via cross-attention
# Both sets of embeddings are returned in a single round-trip to avoid two
# separate HTTP requests when encoding paired prompts.


class LTXVEncodeRequest(BaseModel):
    t5_model: str = Field(
        default="google/t5-v1_1-xxl",
        description="HF repo-id or path of the T5 encoder on the server.",
        examples=["google/t5-v1_1-xxl", "Lightricks/ltx-video-2b-v0.9.5"],
    )
    gemma_model: str = Field(
        default="google/gemma-3-12b",
        description="HF repo-id or .safetensors path of the Gemma-3-12B encoder on the server.",
        examples=["google/gemma-3-12b", "/models/gemma3-12b-fp8.safetensors"],
    )
    text: str = Field(..., description="The prompt to encode with both models.")
    t5_max_length: int = Field(
        default=256,
        ge=16,
        le=4096,
        description="Max token length for T5 (LTX-Video uses 256 by default).",
    )
    gemma_max_length: int = Field(
        default=512,
        ge=16,
        le=8192,
        description="Max token length for Gemma (LTX-Video uses 512 by default).",
    )


@app.post("/comfy/encode/ltxv", tags=["comfy"])
async def comfy_encode_ltxv(req: LTXVEncodeRequest, _: None = AUTH):
    """
    LTX-Video 2.3 dual-encoder conditioning endpoint.

    Runs T5-XXL and Gemma-3-12B encoding in a single request.  The response
    carries both sets of embeddings as base64-encoded float32 tensors.

    **Response structure:**
    ```json
    {
      "t5":    {"embeddings_b64": "...", "shape": [1, T5_tokens, 4096],
                "pooled_b64": "...",    "pooled_shape": [1, 4096]},
      "gemma": {"embeddings_b64": "...", "shape": [1, gemma_tokens, D],
                "pooled_b64": "...",    "pooled_shape": [1, D]}
    }
    ```

    **ComfyUI conditioning layout produced by the companion node:**
    ```
    [[t5_cond [1,T5,4096], {
        "pooled_output":       t5_pooled  [1, 4096],
        "gemma_embeds":        gemma_cond [1, Tg, D],
        "gemma_pooled":        gemma_pool [1, D],
    }]]
    ```
    """
    try:
        result = manager.encode_dual(
            t5_model_id=req.t5_model,
            gemma_model_id=req.gemma_model,
            texts=[req.text],
            t5_max_length=req.t5_max_length,
            gemma_max_length=req.gemma_max_length,
        )
    except Exception as exc:
        logger.exception("LTXV dual encode failed (t5=%s, gemma=%s)", req.t5_model, req.gemma_model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    t5_emb: np.ndarray = result["t5_embeddings"]      # [1, T5, 4096]
    t5_pool: np.ndarray = result["t5_pooled"]          # [1, 4096]
    g_emb: np.ndarray = result["gemma_embeddings"]     # [1, Tg, D]
    g_pool: np.ndarray = result["gemma_pooled"]        # [1, D]

    return {
        "t5": {
            "embeddings_b64": _arr_to_b64(t5_emb),
            "shape": list(t5_emb.shape),
            "pooled_b64": _arr_to_b64(t5_pool),
            "pooled_shape": list(t5_pool.shape),
        },
        "gemma": {
            "embeddings_b64": _arr_to_b64(g_emb),
            "shape": list(g_emb.shape),
            "pooled_b64": _arr_to_b64(g_pool),
            "pooled_shape": list(g_pool.shape),
        },
    }


# ── Gemma raw all-layers endpoint (for local projection) ─────────────────────
# LTX-Video 2.3 uses a learned text_embedding_projection applied *after* Gemma.
# This endpoint returns ALL hidden states so the ComfyUI node can apply the
# projection locally (from a .safetensors file on the client machine).


class GemmaRawEncodeRequest(BaseModel):
    model_name: str = Field(
        ...,
        description="HF repo-id or .safetensors path of the Gemma-3-12B model on the server.",
    )
    text: str = Field(..., description="The prompt to encode.")
    max_length: int = Field(
        default=1024,
        ge=16,
        le=8192,
        description="Max token length (Gemma tokenizer pads to at least this length).",
    )


@app.post("/comfy/encode/gemma_raw", tags=["comfy"])
async def comfy_encode_gemma_raw(req: GemmaRawEncodeRequest, _: None = AUTH):
    """
    Encode text with Gemma and return ALL hidden states stacked.

    **Shape:** ``[1, num_layers+1, T_nonpadded, hidden_size]`` in **float16**.
    For Gemma-3-12B: ``[1, 49, T, 3840]``.

    Intended for the ``LTXVRemoteCLIPLoader`` node which applies the
    ``ltx-2.3_text_projection_bf16.safetensors`` projection locally.

    **Response:**
    ```json
    {
      "all_hidden_b64": "<base64 float16 bytes>",
      "all_hidden_shape": [1, 49, T, 3840],
      "pooled_b64": "<base64 float32 bytes>",
      "pooled_shape": [3840],
      "dtype": "float16"
    }
    ```
    """
    try:
        result = manager.encode_gemma_raw(
            model_id=req.model_name,
            texts=[req.text],
            max_length=req.max_length,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gemma raw encode failed for model %s", req.model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    all_hidden: np.ndarray = result["all_hidden"]   # [1, L, T, D] as int16 view of bfloat16
    pooled: np.ndarray = result["pooled"]           # [1, D] float32

    # Raw bytes are a bfloat16 tensor viewed as int16; client reverses with view(bfloat16)
    all_hidden_b64 = base64.b64encode(
        np.ascontiguousarray(all_hidden).tobytes()
    ).decode("ascii")

    return {
        "all_hidden_b64": all_hidden_b64,
        "all_hidden_shape": list(all_hidden.shape),
        "pooled_b64": _arr_to_b64(pooled[0]),   # [D]
        "pooled_shape": list(pooled.shape[1:]),
        "dtype": "bfloat16",
    }


# ── OpenAI-compatible endpoints ───────────────────────────────────────────────
# These allow tools that speak the OpenAI API (LangChain, LlamaIndex, etc.)
# to use this server as a drop-in embedding provider, and also power
# the ComfyUI node's live model-discovery dropdown.


@app.get("/v1/models", response_model=OAIModelList, tags=["openai"])
async def oai_list_models(_: None = AUTH):
    """
    List available text-encoder models.

    Only models found in the server\'s ``models/`` directory are returned,
    plus any models currently loaded in VRAM that aren\'t in that directory.

    ``id``   – short display name (filename without extension, or dir name).
    ``path`` – full absolute path the server uses to load the model.
    """
    loaded_map: dict[str, str] = {}
    for full_path in manager.loaded_models():
        short = _short_name(full_path)
        loaded_map[short] = full_path

    now = int(time.time())
    entries: dict[str, OAIModelObject] = {}

    # 1 – local models found in models_dir
    for full_path, family in scan_local_models(config.models_dir):
        short = _short_name(full_path)
        entries[short] = OAIModelObject(
            id=short,
            path=full_path,
            created=now,
            family=family,
            loaded=(short in loaded_map),
        )

    # 2 – any loaded models not covered by the directory scan
    for short, full_path in loaded_map.items():
        if short not in entries:
            entries[short] = OAIModelObject(
                id=short,
                path=full_path,
                created=now,
                loaded=True,
            )

    return OAIModelList(data=sorted(entries.values(), key=lambda m: m.id.lower()))


@app.post("/v1/embeddings", response_model=OAIEmbeddingsResponse, tags=["openai"])
async def oai_create_embeddings(req: OAIEmbeddingsRequest, _: None = AUTH):
    """
    Create embeddings in OpenAI format.

    The ``embedding`` field in each data object is the **mean-pooled** float32
    vector (1-D, length = hidden_dim) to match the OpenAI embedding convention.
    Use ``POST /encode`` with ``return_pooled=true`` if you need the full
    token-level sequence tensor.

    **Compatible with:**
    - ``openai`` Python SDK (set ``base_url`` + ``api_key``)
    - LangChain ``OpenAIEmbeddings``
    - LlamaIndex ``OpenAIEmbedding``
    """
    texts: List[str] = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(status_code=422, detail="'input' must not be empty.")

    try:
        result = manager.encode(
            model_id=req.model,
            texts=texts,
            return_pooled=True,
        )
    except Exception as exc:
        logger.exception("OAI embeddings failed for model %s", req.model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    embeddings: np.ndarray = result["embeddings"]      # [B, T, D]
    # Prefer the explicit pooled vector; fall back to mean-pool
    if "pooled" in result:
        pooled: np.ndarray = result["pooled"]          # [B, D]
    else:
        pooled = embeddings.mean(axis=1)               # [B, D]

    # Optionally truncate to requested dimensions
    if req.dimensions and req.dimensions < pooled.shape[-1]:
        pooled = pooled[:, : req.dimensions]

    token_count = embeddings.shape[1]
    data = [
        OAIEmbeddingObject(embedding=pooled[i].tolist(), index=i)
        for i in range(len(texts))
    ]

    return OAIEmbeddingsResponse(
        data=data,
        model=req.model,
        usage={
            "prompt_tokens": token_count * len(texts),
            "total_tokens": token_count * len(texts),
        },
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Remote Text Encoder Server for ComfyUI")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument(
        "--device",
        default=config.device,
        choices=["auto", "cuda", "rocm", "cpu"],
        help="Compute device to use.",
    )
    parser.add_argument(
        "--dtype",
        default=config.dtype,
        choices=["fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2", "fp8"],
        help="Model weight dtype.  fp8* variants store weights compressed and compute in bf16.",
    )
    parser.add_argument("--api-key", default=config.api_key, help="Optional bearer API key.")
    parser.add_argument(
        "--models-dir",
        default=config.models_dir,
        help="Directory scanned for local models (sub-dirs with config.json + top-level .safetensors). "
             "Defaults to a 'models' folder next to server.py.",
    )
    parser.add_argument(
        "--no-keep-loaded",
        action="store_true",
        help="Unload models after TTL expires instead of keeping them in VRAM.",
    )
    parser.add_argument(
        "--model-ttl",
        type=int,
        default=config.model_ttl_seconds,
        help="Seconds before idle models are evicted (0 = never).",
    )
    parser.add_argument("--log-level", default=config.log_level)
    args = parser.parse_args()

    # Apply CLI overrides back to the singleton config
    config.host = args.host
    config.port = args.port
    config.device = args.device
    config.dtype = args.dtype
    config.api_key = args.api_key
    config.models_dir = args.models_dir
    config.keep_models_loaded = not args.no_keep_loaded
    config.model_ttl_seconds = args.model_ttl
    config.log_level = args.log_level.upper()

    logging.getLogger().setLevel(config.log_level)
    return config


if __name__ == "__main__":
    cfg = _parse_args()
    logger.info(
        "Starting Remote Text Encoder  host=%s  port=%d  device=%s  dtype=%s",
        cfg.host, cfg.port, cfg.device, cfg.dtype,
    )
    uvicorn.run(
        "server:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
        reload=False,
    )
