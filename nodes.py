"""
ComfyUI Remote Text Encoder – Node implementations
===================================================

Node graph (mirrors the built-in SD / SDXL / LTX-Video workflow):

  ┌─────────────────────┐
  │  RemoteCLIPLoader   │  server_url, model_name, api_key
  └────────┬────────────┘
           │ REMOTE_CLIP
           ▼
  ┌──────────────────────────┐
  │  CLIPTextEncodeRemote    │  text, clip_skip, max_length
  └────────┬─────────────────┘
           │ CONDITIONING
           ▼
     KSampler / etc.

For SDXL (dual encoder):

  RemoteDualCLIPLoader ──► CLIPTextEncodeCoupleRemote ──► CONDITIONING
  (model 1 = clip-l, model 2 = clip-g)

For LTX-Video 2.3 (T5-XXL + Gemma-3-12B):

  RemoteDualCLIPLoader ──► LTXVTextEncodeRemote ──► CONDITIONING
  (model 1 = T5-XXL, model 2 = Gemma-3-12B)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import folder_paths
from safetensors.torch import load_file as _safetensors_load_file

from . import server_config
from .server_config import (
    MODEL_PLACEHOLDER,
    refresh_models,
    _cached_models_map,
)

logger = logging.getLogger("comfyui.remote_text_encoder")

# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_ltxv_projection() -> str:
    """
    Search ComfyUI's ``text_encoders`` folder(s) for an LTX-V 2.x projection
    safetensors file and return the first match found.

    Matches filenames that contain both ``projection`` and
    ``ltx`` (case-insensitive), e.g.:
      • ltx-2.3_text_projection_bf16.safetensors
      • ltxv_text_projection_fp16.safetensors

    Falls back to an empty string if nothing is found or if ``folder_paths``
    is not importable (i.e. running outside ComfyUI).
    """
    try:
        import folder_paths  # only available inside a ComfyUI process
        search_dirs = folder_paths.get_folder_paths("text_encoders")
    except Exception:
        return ""

    import glob
    import os
    for d in search_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            name_lower = os.path.basename(path).lower()
            if "projection" in name_lower and "ltx" in name_lower:
                return path
    return ""


# ── Types ─────────────────────────────────────────────────────────────────────

# Keep as an internal alias so old serialised workflows that reference the
# string don't break, but loaders now advertise the standard "CLIP" socket.
REMOTE_CLIP_TYPE = "CLIP"


# ── Connection handle ─────────────────────────────────────────────────────────

@dataclass
class RemoteCLIPConnection:
    """Holds everything needed to call the Remote Text Encoder server."""
    server_url: str          # e.g. "http://192.168.1.10:8288"
    model_name: str          # HF repo-id or local path on the server
    api_key: Optional[str]   # Bearer token, or None
    clip_skip: int           # Passed as a hint; the server-side skip is 1-based
    timeout: int             # HTTP request timeout in seconds

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def encode(
        self,
        text: str,
        *,
        max_length: int = 77,
        return_pooled: bool = True,
    ) -> dict[str, Any]:
        """
        Call POST /comfy/encode and return the raw response dict.
        Raises requests.HTTPError on server-side failures.
        """
        url = self.server_url.rstrip("/") + "/comfy/encode"
        payload = {
            "model_name": self.model_name,
            "text": text,
            "clip_skip": self.clip_skip,
        }
        resp = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def encode_batch(
        self,
        texts: list[str],
        *,
        max_length: int = 77,
        return_pooled: bool = True,
    ) -> dict[str, Any]:
        """
        Call POST /encode for a batch of texts.
        Returns {"embeddings_b64": ..., "pooled_b64": ..., "shape": [...]}
        """
        url = self.server_url.rstrip("/") + "/encode"
        payload = {
            "model": self.model_name,
            "texts": texts,
            "max_length": max_length,
            "return_pooled": return_pooled,
        }
        resp = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def encode_ltxv(
        self,
        gemma_model_name: str,
        text: str,
        *,
        t5_max_length: int = 256,
        gemma_max_length: int = 512,
    ) -> dict[str, Any]:
        """
        Call POST /comfy/encode/ltxv – encodes with T5 (self.model_name) and
        Gemma (gemma_model_name) in one round-trip.  Both models must be loaded
        on the same server instance.
        """
        url = self.server_url.rstrip("/") + "/comfy/encode/ltxv"
        payload = {
            "t5_model": self.model_name,
            "gemma_model": gemma_model_name,
            "text": text,
            "t5_max_length": t5_max_length,
            "gemma_max_length": gemma_max_length,
        }
        resp = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def tokenize(self, text: str) -> dict:
        """
        ComfyUI CLIP duck-type interface.
        Stores the text so encode_from_tokens can forward it to the server.
        """
        return {"text": text}

    def encode_from_tokens(
        self,
        tokens: dict,
        return_pooled: bool = True,
        return_dict: bool = False,
    ):
        """
        ComfyUI CLIP duck-type interface.
        Calls the server and returns (cond [1,T,D], pooled [1,D]) so this
        object can be wired directly into any standard ComfyUI encode node.
        """
        data = self.encode(tokens["text"], return_pooled=return_pooled)
        cond = _b64_to_tensor(data["cond"][0], data["shape"]).unsqueeze(0)
        if return_pooled:
            pooled = _b64_to_tensor(
                data["cond"][1]["pooled_output"], data["pooled_shape"]
            ).unsqueeze(0)
            return cond, pooled
        return cond

    def ping(self) -> bool:
        """Quick health-check – returns True if the server responds."""
        try:
            r = requests.get(self.server_url.rstrip("/") + "/", timeout=5, headers=self._headers)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def cond_stage_model(self):
        raise RuntimeError(
            "RemoteCLIPConnection cannot be used with LoRA loaders. "
            "LoRA patching modifies local model weights and is incompatible with a "
            "remote CLIP connection. Apply LoRAs on the server side, or use a local "
            "CLIP model for LoRA loading before feeding into a Remote encode node."
        )


@dataclass
class RemoteDualCLIPConnection:
    """
    Dual-encoder CLIP object.  Mirrors what ComfyUI's DualCLIPLoader produces:
    two models loaded internally, one CLIP socket out.

    Implements the ComfyUI CLIP duck-type (tokenize / encode_from_tokens) so it
    can be dropped into any standard encode node.  Our custom nodes
    (CLIPTextEncodeCoupleRemote, LTXVTextEncodeRemote) cast it back to this
    type when they need to call the two encoders separately.
    """
    clip1: RemoteCLIPConnection   # CLIP-L (SDXL) or T5-XXL (LTX-V)
    clip2: RemoteCLIPConnection   # CLIP-G (SDXL) or Gemma-3-12B (LTX-V)

    def tokenize(self, text: str) -> dict:
        return {"text": text}

    def encode_from_tokens(
        self,
        tokens: dict,
        return_pooled: bool = True,
        return_dict: bool = False,
    ):
        """
        Default encode: SDXL-style dual encode with the same text for both
        encoders.  Suitable for standard CLIPTextEncode nodes.
        """
        text = tokens["text"]
        data_l = self.clip1.encode(text, return_pooled=False)
        data_g = self.clip2.encode(text, return_pooled=True)
        emb_l = _b64_to_tensor(data_l["cond"][0], data_l["shape"])
        emb_g = _b64_to_tensor(data_g["cond"][0], data_g["shape"])
        t_l, t_g = emb_l.shape[0], emb_g.shape[0]
        if t_l < t_g:
            emb_l = torch.cat([emb_l, torch.zeros(t_g - t_l, emb_l.shape[1])], dim=0)
        elif t_g < t_l:
            emb_g = torch.cat([emb_g, torch.zeros(t_l - t_g, emb_g.shape[1])], dim=0)
        cond = torch.cat([emb_l, emb_g], dim=-1).unsqueeze(0)
        pooled = _b64_to_tensor(
            data_g["cond"][1]["pooled_output"], data_g["pooled_shape"]
        ).unsqueeze(0)
        if return_pooled:
            return cond, pooled
        return cond

    @property
    def cond_stage_model(self):
        raise RuntimeError(
            "RemoteDualCLIPConnection cannot be used with LoRA loaders. "
            "LoRA patching modifies local model weights and is incompatible with a "
            "remote CLIP connection. Apply LoRAs on the server side, or use a local "
            "CLIP model for LoRA loading before feeding into a Remote encode node."
        )


class _DualLinearProjection(nn.Module):
    """
    Local replica of ComfyUI's ``DualLinearProjection`` (lt.py).

    Takes the raw all-hidden tensor [B, L, T, D] and returns
    [B, T, out_video + out_audio] (e.g. [B, T, 6144] for the 2.3 release).
    """
    def __init__(self, in_dim: int, out_dim_video: int, out_dim_audio: int):
        super().__init__()
        self.video_aggregate_embed = nn.Linear(in_dim, out_dim_video, bias=True)
        self.audio_aggregate_embed = nn.Linear(in_dim, out_dim_audio, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        source_dim = x.shape[-1]               # D (= 3840)
        x = x.movedim(1, -1)                   # [B, T, D, L]
        # RMS-normalise over the D dimension, then flatten D×L
        x = (x * torch.rsqrt(torch.mean(x ** 2, dim=2, keepdim=True) + 1e-6)
             ).flatten(start_dim=2)            # [B, T, D*L]
        video = self.video_aggregate_embed(
            x * math.sqrt(self.video_aggregate_embed.out_features / source_dim)
        )
        audio = self.audio_aggregate_embed(
            x * math.sqrt(self.audio_aggregate_embed.out_features / source_dim)
        )
        return torch.cat((video, audio), dim=-1)


@dataclass
class LTXVRemoteCLIPConnection:
    """
    Hybrid encoder for LTX-Video 2.x (Gemma-3-12B on server + local projection).

    Gemma runs remotely on the GPU server and returns ALL hidden states
    (embedding layer + 48 transformer layers = 49 total, shape [1, 49, T, 3840]).
    The projection module (loaded from a local ``.safetensors`` file) is applied
    here.  Two projection types are supported:

    ``single_linear``
        ``Linear(3840×49 → 3840, bias=False)``
        Key: ``text_embedding_projection.weight``
        Output: [B, T, 3840]

    ``dual_linear``
        ``DualLinearProjection(3840×49 → video_dim + audio_dim)``
        Keys: ``text_embedding_projection.video_aggregate_embed.*``
              ``text_embedding_projection.audio_aggregate_embed.*``
        Output: [B, T, video_dim + audio_dim]  (e.g. [B, T, 6144])
    """
    server_url: str
    model_name: str
    api_key: Optional[str]
    timeout: int
    gemma_max_length: int
    projection: nn.Module            # single_linear → nn.Linear  |  dual_linear → _DualLinearProjection
    projection_type: str             # "single_linear" or "dual_linear"

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def tokenize(self, text: str) -> dict:
        return {"text": text}

    def encode_ltxv(self, text: str) -> tuple:
        """
        Encode *text* via the server then apply the local projection.

        Returns
        -------
        cond   : torch.Tensor [1, T, 3840] float32
        pooled : torch.Tensor [1, 3840]   float32
        extra  : {"unprocessed_ltxav_embeds": True} – consumed by av_model.py
        """
        url = self.server_url.rstrip("/") + "/comfy/encode/gemma_raw"
        payload = {
            "model_name": self.model_name,
            "text": text,
            "max_length": self.gemma_max_length,
        }
        resp = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # ── Decode all-hidden tensor [1, L, T, D] ───────────────────────────
        # Server sends bfloat16 bytes viewed as int16 (numpy has no bf16 dtype).
        # Reverse: read as int16, reinterpret as bfloat16 via torch view, cast to float32.
        shape = data["all_hidden_shape"]
        raw = base64.b64decode(data["all_hidden_b64"])
        wire_dtype = data.get("dtype", "float16")
        if wire_dtype == "bfloat16":
            all_hidden = torch.from_numpy(
                np.frombuffer(raw, dtype=np.int16).reshape(shape).copy()
            ).view(torch.bfloat16).float()  # [B, L, T, D]
        else:
            all_hidden = torch.from_numpy(
                np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
            ).float()  # [B, L, T, D] – legacy float16 path

        # ── Decode pooled [D] ────────────────────────────────────────────────
        p_raw = base64.b64decode(data["pooled_b64"])
        p_shape = data["pooled_shape"]
        pooled = torch.from_numpy(
            np.frombuffer(p_raw, dtype=np.float32).reshape(p_shape).copy()
        ).unsqueeze(0)  # [1, D]

        # ── Apply projection (mirrors lt.py LTXAVTEModel.encode_token_weights) ─
        proj_device = next(self.projection.parameters()).device
        out = all_hidden.to(proj_device)    # [B, L, T, D]

        if self.projection_type == "single_linear":
            # movedim(1,-1) → [B, T, D, L]
            # range-normalise to [-8, 8] over dims (T, D)
            # reshape → [B, T, D*L] = [B, T, 3840*49]
            # Linear(188160 → 3840)
            out = out.movedim(1, -1)            # [B, T, D, L]
            mean = out.mean(dim=(1, 2), keepdim=True)
            rng = out.amax(dim=(1, 2), keepdim=True) - out.amin(dim=(1, 2), keepdim=True)
            out = 8.0 * (out - mean) / (rng + 1e-6)
            out = out.reshape(out.shape[0], out.shape[1], -1)   # [B, T, D*L]
            out = self.projection(out)                           # [B, T, 3840]
        else:
            # dual_linear: DualLinearProjection does its own movedim + RMS norm
            # input: [B, L, T, D] → output: [B, T, video_dim + audio_dim]
            out = self.projection(out)

        out = out.float().cpu()                              # ensure float32 on CPU

        return out, pooled, {"unprocessed_ltxav_embeds": True}

    @property
    def cond_stage_model(self):
        raise RuntimeError(
            "LTXVRemoteCLIPConnection cannot be used with LoRA loaders. "
            "The projection weights are loaded locally, but LoRA patching requires "
            "the full model graph. Apply LoRAs on the server side instead."
        )


# ── Decode helpers ────────────────────────────────────────────────────────────

def _b64_to_tensor(b64: str, shape: list[int]) -> torch.Tensor:
    """Decode base64 raw float32 bytes → torch.Tensor on CPU."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(arr)


def _comfy_encode_response_to_conditioning(
    data: dict[str, Any],
) -> list[tuple[torch.Tensor, dict]]:
    """
    Convert a /comfy/encode response into ComfyUI CONDITIONING format:
      [[cond_tensor [T, D], {"pooled_output": pooled_tensor [D]}]]
    """
    emb = _b64_to_tensor(data["cond"][0], data["shape"])          # [T, D]
    pooled = _b64_to_tensor(
        data["cond"][1]["pooled_output"], data["pooled_shape"]
    )                                                               # [D]

    # ComfyUI expects cond as [1, T, D]
    cond = emb.unsqueeze(0)
    return [[cond, {"pooled_output": pooled.unsqueeze(0)}]]


def _batch_encode_to_conditioning(
    data: dict[str, Any],
) -> list[tuple[torch.Tensor, dict]]:
    """
    Convert a /encode batch response (single text item) into
    ComfyUI CONDITIONING format.
    """
    shape: list[int] = data["shape"]          # [B, T, D]
    emb = _b64_to_tensor(data["embeddings_b64"], shape)  # [B, T, D]

    result = []
    for i in range(shape[0]):
        extra: dict[str, Any] = {}
        if data.get("pooled_b64"):
            pooled_shape = [shape[0], shape[2]]
            pooled_full = _b64_to_tensor(data["pooled_b64"], pooled_shape)
            extra["pooled_output"] = pooled_full[i].unsqueeze(0)  # [1, D]
        result.append([emb[i].unsqueeze(0), extra])               # [1, T, D]
    return result


# ── Node: RemoteCLIPLoader ────────────────────────────────────────────────────

class RemoteCLIPLoader:
    """
    Replacement for the built-in CLIPLoader / DualCLIPLoader nodes.

    The ``model_name`` dropdown is populated live from the server’s
    GET /v1/models endpoint using the URL stored in ``rte_config.json``.
    Run the **Refresh Remote Models** utility node or press F5 in ComfyUI
    to update the list after loading a new model on the server.

    If a model you need is not in the dropdown yet, type its name into
    the ``custom_model`` field – this overrides the dropdown selection.

    Outputs a REMOTE_CLIP handle that stores the server connection
    parameters.  No network call is made at load time.
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        models_map = refresh_models(server_config._cache_server_url)
        model_names = list(models_map.keys())
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8288",
                        "multiline": False,
                        "tooltip": "Base URL of the Remote Text Encoder server. "
                                   "Changing this and pressing F5 refreshes the model list.",
                    },
                ),
                "model_name": (
                    model_names,
                    {
                        "tooltip": "Select a model discovered from the server via GET /v1/models. "
                                   "Use 'custom_model' to enter a name not in this list.",
                    },
                ),
                "clip_skip": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "Number of CLIP layers to skip (1 = no skip).",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 5,
                        "max": 600,
                        "step": 5,
                        "tooltip": "HTTP request timeout in seconds.",
                    },
                ),
            },
            "optional": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Bearer API key if the server requires authentication.",
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "If non-empty, overrides the model_name dropdown. "
                                   "Use this for models not yet in the discovery list.",
                    },
                ),
            },
        }

    def load(
        self,
        server_url: str,
        model_name: str,
        clip_skip: int,
        timeout: int,
        api_key: str = "",
        custom_model: str = "",
    ) -> tuple[RemoteCLIPConnection]:
        actual_model = custom_model.strip() if custom_model.strip() else model_name
        if actual_model == MODEL_PLACEHOLDER:
            raise ValueError(
                "No model selected. Either pick one from the dropdown or enter a name in 'custom_model'."
            )

        # Resolve short display name → full path (if we have a cached map)
        if not custom_model.strip():
            from .server_config import _cached_models_map
            if actual_model in _cached_models_map and _cached_models_map[actual_model]:
                actual_model = _cached_models_map[actual_model]

        # Remember the URL so the next R-refresh queries the right server
        refresh_models(server_url.strip(), api_key.strip(), timeout)

        conn = RemoteCLIPConnection(
            server_url=server_url.strip(),
            model_name=actual_model,
            api_key=api_key.strip() or None,
            clip_skip=clip_skip,
            timeout=timeout,
        )
        logger.info(
            "RemoteCLIPLoader: server=%s  model=%s  clip_skip=%d",
            conn.server_url,
            conn.model_name,
            conn.clip_skip,
        )
        return (conn,)

# ── Node: RemoteDualCLIPLoader ──────────────────────────────────────────────

class RemoteDualCLIPLoader:
    """
    Loads two text encoders from the same server and outputs a single CLIP
    object — exactly like ComfyUI's built-in DualCLIPLoader.

    Wire the output to:
      • CLIPTextEncodeRemote      – encodes the same text with both models (SDXL default)
      • CLIPTextEncodeCoupleRemote – SDXL with separate text_l / text_g prompts
      • LTXVTextEncodeRemote       – LTX-Video T5 + Gemma encoding
      • Any standard ComfyUI CLIPTextEncode node
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        models_map = refresh_models(server_config._cache_server_url or "http://localhost:8288")
        model_names = list(models_map.keys())
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8288",
                        "multiline": False,
                        "tooltip": "Base URL of the Remote Text Encoder server.",
                    },
                ),
                "model_name_1": (
                    model_names,
                    {"tooltip": "First encoder  (e.g. CLIP-L for SDXL, T5-XXL for LTX-Video)."},
                ),
                "model_name_2": (
                    model_names,
                    {"tooltip": "Second encoder (e.g. CLIP-G for SDXL, Gemma-3-12B for LTX-Video)."},
                ),
                "clip_skip": (
                    "INT",
                    {"default": 1, "min": 1, "max": 12, "step": 1,
                     "tooltip": "CLIP layer skip applied to both encoders."},
                ),
                "timeout": (
                    "INT",
                    {"default": 60, "min": 5, "max": 600, "step": 5,
                     "tooltip": "HTTP request timeout in seconds."},
                ),
            },
            "optional": {
                "api_key": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "Bearer API key if the server requires authentication."},
                ),
            },
        }

    def load(
        self,
        server_url: str,
        model_name_1: str,
        model_name_2: str,
        clip_skip: int,
        timeout: int,
        api_key: str = "",
    ) -> tuple[RemoteDualCLIPConnection]:
        url = server_url.strip()
        key = api_key.strip() or None
        models_map = refresh_models(url, api_key.strip(), timeout)

        def _resolve(name: str) -> str:
            return models_map.get(name, name) or name

        conn1 = RemoteCLIPConnection(
            server_url=url, model_name=_resolve(model_name_1),
            api_key=key, clip_skip=clip_skip, timeout=timeout,
        )
        conn2 = RemoteCLIPConnection(
            server_url=url, model_name=_resolve(model_name_2),
            api_key=key, clip_skip=clip_skip, timeout=timeout,
        )
        logger.info(
            "RemoteDualCLIPLoader: server=%s  model_1=%s  model_2=%s",
            url, conn1.model_name, conn2.model_name,
        )
        return (RemoteDualCLIPConnection(clip1=conn1, clip2=conn2),)


# ── Node: LTXVRemoteCLIPLoader ────────────────────────────────────────────────

class LTXVRemoteCLIPLoader:
    """
    LTX-Video 2.3 hybrid CLIP loader: Gemma-3-12B runs on the remote server,
    the ``text_embedding_projection`` linear layer loads from a local
    ``.safetensors`` file and runs on the ComfyUI machine.

    Why split?  Gemma-3-12B is ~12 GB – offload it to a server with a big GPU.
    The projection weights (``ltx-2.3_text_projection_bf16.safetensors``) are
    ~1.4 GB; keeping them local avoids syncing large files to the server while
    still saving the dominant VRAM cost.

    Required projection .safetensors keys
    --------------------------------------
    • ``text_embedding_projection.weight``  shape [3840, 188160]  (= 3840 × 3840 × 49)

    Connect:
        LTXVRemoteCLIPLoader.clip  →  LTXVTextEncodeRemote.clip
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        models_map = refresh_models(server_config._cache_server_url or "http://localhost:8288")
        model_names = list(models_map.keys())
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8288",
                        "multiline": False,
                        "tooltip": "Base URL of the Remote Text Encoder server.",
                    },
                ),
                "model_name": (
                    model_names,
                    {
                        "tooltip": "Gemma-3-12B model on the server "
                                   "(HF repo-id or path to .safetensors).",
                    },
                ),
                "projection_path": (
                    folder_paths.get_filename_list("text_encoders"),
                    {
                        "tooltip": "Projection file from the text_encoders folder "
                                   "(e.g. ltx-2.3_text_projection_bf16.safetensors).",
                    },
                ),
                "gemma_max_length": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Max Gemma token length. "
                                   "LTX-V 2.3 tokenizer pads prompts to at least 1024 tokens.",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 120,
                        "min": 5,
                        "max": 600,
                        "step": 5,
                        "tooltip": "HTTP request timeout in seconds "
                                   "(all-layer Gemma encode may be slower than single-layer).",
                    },
                ),
            },
            "optional": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Bearer API key if the server requires authentication.",
                    },
                ),
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Overrides the model_name dropdown. "
                                   "Use for models not yet in the discovery list.",
                    },
                ),
            },
        }

    def load(
        self,
        server_url: str,
        model_name: str,
        projection_path: str,
        gemma_max_length: int,
        timeout: int,
        api_key: str = "",
        custom_model: str = "",
    ) -> tuple:
        # ── Resolve model name ────────────────────────────────────────────────
        actual_model = custom_model.strip() if custom_model.strip() else model_name
        if actual_model == MODEL_PLACEHOLDER:
            raise ValueError(
                "No Gemma model selected. Either pick from the dropdown or use 'custom_model'."
            )
        if not custom_model.strip():
            from .server_config import _cached_models_map
            if actual_model in _cached_models_map and _cached_models_map[actual_model]:
                actual_model = _cached_models_map[actual_model]

        refresh_models(server_url.strip(), api_key.strip(), timeout)

        # ── Load local projection weights ─────────────────────────────────────
        proj_path = folder_paths.get_full_path_or_raise("text_encoders", projection_path)

        logger.info("LTXVRemoteCLIPLoader: loading projection from %s", proj_path)
        sd = _safetensors_load_file(proj_path)

        logger.info("LTXVRemoteCLIPLoader: keys in file: %s",
                    {k: tuple(v.shape) for k, v in sd.items()})

        # ── Detect projection type from state-dict keys ───────────────────────
        has_weight      = "text_embedding_projection.weight" in sd
        has_video       = "text_embedding_projection.video_aggregate_embed.weight" in sd
        has_audio       = "text_embedding_projection.audio_aggregate_embed.weight" in sd
        has_legacy      = "text_projection" in sd

        if has_video and has_audio:
            # ── dual_linear ─────────────────────────────────────────────────
            proj_type = "dual_linear"
            vid_w = sd["text_embedding_projection.video_aggregate_embed.weight"].float()
            vid_b = sd.get("text_embedding_projection.video_aggregate_embed.bias")
            aud_w = sd["text_embedding_projection.audio_aggregate_embed.weight"].float()
            aud_b = sd.get("text_embedding_projection.audio_aggregate_embed.bias")

            in_f      = vid_w.shape[1]
            out_video = vid_w.shape[0]
            out_audio = aud_w.shape[0]

            projection = _DualLinearProjection(in_f, out_video, out_audio)
            projection.video_aggregate_embed.weight = nn.Parameter(vid_w)
            if vid_b is not None:
                projection.video_aggregate_embed.bias = nn.Parameter(vid_b.float())
            projection.audio_aggregate_embed.weight = nn.Parameter(aud_w)
            if aud_b is not None:
                projection.audio_aggregate_embed.bias = nn.Parameter(aud_b.float())

            logger.info(
                "LTXVRemoteCLIPLoader: dual_linear projection  in=%d  video_out=%d  audio_out=%d",
                in_f, out_video, out_audio,
            )

        elif has_weight or has_legacy:
            # ── single_linear ────────────────────────────────────────────────
            proj_type  = "single_linear"
            proj_weight = (sd["text_embedding_projection.weight"] if has_weight
                           else sd["text_projection"]).float()

            # Some checkpoints pack it flat; reshape if needed
            if proj_weight.ndim == 1:
                out_f = 3840
                if proj_weight.numel() % out_f != 0:
                    raise ValueError(
                        f"Cannot reshape 1-D projection tensor of size "
                        f"{proj_weight.numel()} into [3840, N]"
                    )
                proj_weight = proj_weight.reshape(out_f, -1)

            out_f, in_f = proj_weight.shape
            projection = nn.Linear(in_f, out_f, bias=False)
            projection.weight = nn.Parameter(proj_weight)

            logger.info(
                "LTXVRemoteCLIPLoader: single_linear projection  in=%d  out=%d",
                in_f, out_f,
            )

        else:
            raise ValueError(
                f"Cannot determine projection type from keys: {list(sd.keys())}. "
                "Expected 'text_embedding_projection.weight' (single_linear) or "
                "'text_embedding_projection.video_aggregate_embed.weight' (dual_linear)."
            )

        projection.eval()

        conn = LTXVRemoteCLIPConnection(
            server_url=server_url.strip(),
            model_name=actual_model,
            api_key=api_key.strip() or None,
            timeout=timeout,
            gemma_max_length=gemma_max_length,
            projection=projection,
            projection_type=proj_type,
        )
        logger.info(
            "LTXVRemoteCLIPLoader: server=%s  model=%s  projection_type=%s",
            conn.server_url, conn.model_name, proj_type,
        )
        return (conn,)


# ── Node: CLIPTextEncodeRemote ────────────────────────────────────────────────

class CLIPTextEncodeRemote:
    """
    Drop-in replacement for CLIPTextEncode that sends the prompt to the
    Remote Text Encoder server and returns standard ComfyUI CONDITIONING.

    Connect:
        RemoteCLIPLoader.clip  →  CLIPTextEncodeRemote.clip
        primitive string       →  CLIPTextEncodeRemote.text
        CONDITIONING output    →  KSampler.positive / .negative
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {}),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text prompt to encode.",
                    },
                ),
            },
            "optional": {
                "max_length": (
                    "INT",
                    {
                        "default": 77,
                        "min": 16,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Maximum token length (must match the model's limit).",
                    },
                ),
            },
        }

    def encode(
        self,
        clip: RemoteCLIPConnection,
        text: str,
        max_length: int = 77,
    ) -> tuple[list]:
        try:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Remote encoder returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Remote Text Encoder. Is the server running?"
            ) from exc
        return ([[cond, {"pooled_output": pooled}]],)


# ── Node: CLIPTextEncodeCoupleRemote (SDXL dual encoder) ─────────────────────

class CLIPTextEncodeCoupleRemote:
    """
    SDXL-style dual-encoder conditioning node.

    In SDXL, conditioning is produced by two CLIP models:
      • clip-l  (ViT-L/14)  – 77 tokens × 768-dim
      • clip-g  (ViT-bigG)  – 77 tokens × 1280-dim, also provides the pooled vector

    The combined conditioning tensor is the concatenation along the last axis,
    and the pooled output comes from clip-g only.

    Connect:
        RemoteDualCLIPLoader.clip  →  .clip   (model 1 = clip-l, model 2 = clip-g)
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "Dual CLIP from RemoteDualCLIPLoader (clip-l as model 1, clip-g as model 2)."},
                ),
                "text_l": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompt for CLIP-L (typically the short, precise description).",
                    },
                ),
                "text_g": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Prompt for CLIP-G (typically the full, rich description).",
                    },
                ),
            },
        }

    def encode(
        self,
        clip: RemoteDualCLIPConnection,
        text_l: str,
        text_g: str,
    ) -> tuple[list]:
        remote_clip_l = clip.clip1
        remote_clip_g = clip.clip2
        # Encode with CLIP-L
        try:
            data_l = remote_clip_l.encode(text_l, return_pooled=False)
        except Exception as exc:
            raise RuntimeError(f"CLIP-L encode failed: {exc}") from exc

        # Encode with CLIP-G (also provides the pooled output for SDXL)
        try:
            data_g = remote_clip_g.encode(text_g, return_pooled=True)
        except Exception as exc:
            raise RuntimeError(f"CLIP-G encode failed: {exc}") from exc

        emb_l = _b64_to_tensor(data_l["cond"][0], data_l["shape"])   # [T, D_l]
        emb_g = _b64_to_tensor(data_g["cond"][0], data_g["shape"])   # [T, D_g]

        # Pad the shorter sequence to match token lengths
        t_l, t_g = emb_l.shape[0], emb_g.shape[0]
        if t_l < t_g:
            pad = torch.zeros(t_g - t_l, emb_l.shape[1])
            emb_l = torch.cat([emb_l, pad], dim=0)
        elif t_g < t_l:
            pad = torch.zeros(t_l - t_g, emb_g.shape[1])
            emb_g = torch.cat([emb_g, pad], dim=0)

        # Concatenate along the hidden-dim axis: [1, T, D_l + D_g]
        combined = torch.cat([emb_l, emb_g], dim=-1).unsqueeze(0)

        # Pooled output from CLIP-G [1, D_g]
        pooled = _b64_to_tensor(
            data_g["cond"][1]["pooled_output"], data_g["pooled_shape"]
        ).unsqueeze(0)

        conditioning = [[combined, {"pooled_output": pooled}]]
        return (conditioning,)


# ── Node: LTXVTextEncodeRemote (LTX-Video 2.3 dual encoder) ────────────────────

class LTXVTextEncodeRemote:
    """
    LTX-Video 2.3 text conditioning node.

    LTX-Video 2.3 (Lightricks) uses two text encoders:
      • T5-XXL (primary cross-attention)   – max 256 tokens, 4096-dim
      • Gemma-3-12B (secondary guidance)   – max 512 tokens, variable-dim

    This node calls the server’s /comfy/encode/ltxv endpoint in a single
    HTTP request and assembles the ComfyUI CONDITIONING expected by LTX-V
    sampler nodes:

        [[t5_cond [1, T5, 4096], {
            "pooled_output":  t5_pooled  [1, 4096],
            "gemma_embeds":   gemma_cond [1, Tg, D],
            "gemma_pooled":   gemma_pool [1, D],
        }]]

    Connect:
        RemoteDualCLIPLoader.clip  →  .clip   (model 1 = T5-XXL, model 2 = Gemma-3-12B)
        text prompt                →  .text
        CONDITIONING output        →  LTXVScheduler / LTXVSampler
    """

    CATEGORY = "conditioning/remote"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "Dual CLIP from RemoteDualCLIPLoader (T5-XXL as model 1, Gemma-3-12B as model 2)."},
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text prompt to encode with both T5 and Gemma.",
                    },
                ),
                "t5_max_length": (
                    "INT",
                    {
                        "default": 256,
                        "min": 16,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "T5 token budget (LTX-Video default: 256).",
                    },
                ),
                "gemma_max_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "step": 16,
                        "tooltip": "Gemma token budget (LTX-Video default: 512).",
                    },
                ),
            },
        }

    def encode(
        self,
        clip,
        text: str,
        t5_max_length: int = 256,
        gemma_max_length: int = 512,
    ) -> tuple[list]:
        """
        Dual path  (RemoteDualCLIPConnection): calls /comfy/encode/ltxv with
            both T5 and Gemma models in one round-trip.
        Single path (RemoteCLIPConnection): treats the connection as a single
            Gemma encoder and returns standard ComfyUI conditioning. Use this
            for LTX-V 2.3 where the text_projection is not a text encoder and
            only Gemma needs to run remotely.
        """
        # ── LTXVRemoteCLIPConnection (remote Gemma + local projection) ────
        if isinstance(clip, LTXVRemoteCLIPConnection):
            cond, pooled, extra = clip.encode_ltxv(text)
            return ([[cond, {"pooled_output": pooled, **extra}]],)

        # ── Single-encoder fast path ──────────────────────────────────────
        if isinstance(clip, RemoteCLIPConnection):
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return ([[cond, {"pooled_output": pooled}]],)

        # ── Dual-encoder path ─────────────────────────────────────────────
        remote_t5 = clip.clip1
        remote_gemma = clip.clip2
        try:
            data = remote_t5.encode_ltxv(
                gemma_model_name=remote_gemma.model_name,
                text=text,
                t5_max_length=t5_max_length,
                gemma_max_length=gemma_max_length,
            )
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"LTXV encode returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Remote Text Encoder at {remote_t5.server_url!r}. "
                "Is the server running?"
            ) from exc

        # ── Decode T5 tensors ───────────────────────────────────────────────
        t5_emb = _b64_to_tensor(
            data["t5"]["embeddings_b64"], data["t5"]["shape"]
        )  # [1, T5, 4096]
        t5_pooled = _b64_to_tensor(
            data["t5"]["pooled_b64"], data["t5"]["pooled_shape"]
        )  # [1, 4096]

        # ── Decode Gemma tensors ──────────────────────────────────────────
        g_emb = _b64_to_tensor(
            data["gemma"]["embeddings_b64"], data["gemma"]["shape"]
        )  # [1, Tg, D]
        g_pooled = _b64_to_tensor(
            data["gemma"]["pooled_b64"], data["gemma"]["pooled_shape"]
        )  # [1, D]

        # ── Build ComfyUI CONDITIONING ─────────────────────────────────────
        # T5 provides the main token sequence; Gemma goes in the extra dict.
        # LTX-Video’s DiT reads:
        #   encoder_hidden_states        ← t5_emb [1, T5, 4096]
        #   encoder_attention_mask       ← full-1 mask (no padding tracked here)
        #   gemma_embeds                 ← g_emb  [1, Tg, D]
        #   gemma_pooled                 ← g_pooled [1, D]
        conditioning = [[
            t5_emb,
            {
                "pooled_output": t5_pooled,
                "gemma_embeds": g_emb,
                "gemma_pooled": g_pooled,
            },
        ]]
        return (conditioning,)
