"""
Model manager: loads, caches, and runs CLIP / T5 / Gemma text encoders.

Supported model families
------------------------
clip-l      CLIP ViT-L/14 (used in SD 1.x, SDXL)
clip-g      CLIP ViT-bigG (used in SDXL)
t5-xxl      T5-XXL (used in SD3, FLUX)
t5-xl       T5-XL (smaller T5 variant)
open-clip   Generic OpenCLIP hub model
gemma       Gemma-3 decoder-only encoder (used in Imagen 3, Lumina, etc.)

Models are identified by their Hugging Face repo-id, a local directory,
or the full path to a single .safetensors weights file.  When a single
.safetensors file is given the parent directory must contain config.json
and the tokenizer files.

FP8 weight dtypes
-----------------
fp8e4m3     torch.float8_e4m3fn  (recommended – wider dynamic range)
fp8e5m2     torch.float8_e5m2

FP8 weights are kept in compressed storage.  The forward pass runs under
torch.autocast(bf16) so the compute dtype is bf16.  On Hopper (sm90) and
MI300X hardware, torch will dispatch native FP8 matmuls automatically.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from config import ServerConfig

logger = logging.getLogger(__name__)

# ── dtype helper ──────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # FP8 variants – weights stored in FP8, compute in bf16 via autocast
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    # Convenience aliases
    "fp8": torch.float8_e4m3fn,
}

# dtypes that require autocast during the forward pass
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def _resolve_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP.get(name.lower(), torch.float16)


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def scan_local_models(models_dir: str) -> list[tuple[str, str]]:
    """
    Walk *models_dir* and return a list of ``(model_id, family)`` pairs where
    ``model_id`` is the absolute path that ``ModelManager`` can load directly.

    Discovery rules
    ---------------
    1. Sub-directory containing ``config.json``  → model directory
    2. Top-level ``.safetensors`` file            → single-file model
    3. Sub-directory containing only ``.safetensors`` files (no ``config.json``)
       is skipped – those are likely incomplete downloads.

    The *family* is inferred from the directory / file name via
    ``_detect_family``.
    """
    results: list[tuple[str, str]] = []
    root = Path(models_dir)
    if not root.is_dir():
        return results

    # 1 & 3 – scan sub-directories
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            model_id = str(child.resolve())
            results.append((model_id, _detect_family(model_id)))

    # 2 – top-level .safetensors files
    for sf in sorted(root.glob("*.safetensors")):
        model_id = str(sf.resolve())
        results.append((model_id, _detect_family(model_id)))

    return results


def _cast_to_fp8(model: Any, dtype: torch.dtype) -> Any:
    """
    Cast all floating-point parameters and buffers in *model* to *dtype*
    (one of the FP8 types) in-place and return the model.

    Embedding layers are kept in bf16 because FP8 embeddings cause index
    operations to fail on most hardware.  LayerNorm / RMSNorm weights are
    also kept in bf16 for numerical stability.
    """
    _SKIP_TYPES = ("embed", "norm", "ln_")

    for name, param in model.named_parameters():
        name_l = name.lower()
        if any(s in name_l for s in _SKIP_TYPES):
            continue
        if param.is_floating_point():
            param.data = param.data.to(dtype)

    for name, buf in model.named_buffers():
        name_l = name.lower()
        if any(s in name_l for s in _SKIP_TYPES):
            continue
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype)

    return model


# ── Model families ────────────────────────────────────────────────────────────

#: Well-known HF repo IDs → (family, default_dtype)
KNOWN_MODELS: Dict[str, Tuple[str, str]] = {
    "openai/clip-vit-large-patch14": ("clip", "fp16"),
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": ("clip", "fp16"),
    "stabilityai/stable-diffusion-xl-base-1.0": ("clip", "fp16"),
    "google/t5-v1_1-xxl": ("t5", "fp16"),
    "google/flan-t5-xxl": ("t5", "fp16"),
    "google/t5-v1_1-xl": ("t5", "fp16"),
    # LTX-Video 2.3 (Lightricks)
    "Lightricks/LTX-Video": ("t5", "fp16"),           # repo ships its own T5 config
    "Lightricks/ltx-video-2b-v0.9.5": ("t5", "fp16"),
    # Gemma-3 variants
    "google/gemma-3-12b": ("gemma", "fp8e4m3"),
    "google/gemma-3-12b-it": ("gemma", "fp8e4m3"),
    "google/gemma-3-4b": ("gemma", "fp16"),
    "google/gemma-3-1b": ("gemma", "fp16"),
}


def _detect_family(model_id: str) -> str:
    """Infer the model family from its repo-id, local directory, or file path."""
    p = Path(model_id)
    # Build a search string from stem, parent dir name, and full id
    search_str = (p.stem + " " + p.parent.name + " " + model_id).lower()

    if "gemma" in search_str:
        return "gemma"
    # T5 variants – covers "t5", "ltx" (LTX-Video uses T5-XXL),
    # "text_encoder" / "text_projection", and FLUX/SD3 t5 names
    if any(tok in search_str for tok in ("t5", "ltx", "text_encoder", "text_projection", "flan")):
        return "t5"
    return "clip"


#: Default HF repo-id used to fetch config + tokenizer when a single
#: .safetensors file is placed in a flat directory that has no config.json.
_FAMILY_FALLBACK_REPO: dict[str, str] = {
    "clip":  "openai/clip-vit-large-patch14",
    "t5":    "google/t5-v1_1-xxl",
    # Use the text-only (CausalLM) variant so AutoModelForCausalLM.from_config
    # produces Gemma3ForCausalLM with model.layers.* keys, matching standalone
    # text-encoder safetensors files.  The "-it" (instruction-tuned) variant is
    # Gemma3ForConditionalGeneration and puts language weights under
    # language_model.model.* which mismatches the checkpoint layout.
    "gemma": "google/gemma-3-12b",
}


def _config_source(model_id: str, family: str, cache_dir: Optional[str]) -> str:
    """
    Return the source to pass to ``*.from_pretrained()`` for loading the model
    architecture and tokenizer.

    * If *model_id* is a directory (or a HF repo-id), return it directly.
    * If it is a single ``.safetensors`` file whose parent directory contains
      a ``config.json``, return the parent directory.
    * Otherwise fall back to the canonical HF repo for the family so that
      ``from_pretrained`` can fetch config/tokenizer without needing local
      config files alongside the weights file.
    """
    if not _is_single_safetensors(model_id):
        return model_id
    parent = Path(model_id).parent
    if (parent / "config.json").exists():
        return str(parent)
    fallback = _FAMILY_FALLBACK_REPO.get(family, "openai/clip-vit-large-patch14")
    logger.info(
        "No config.json found next to %s – using %s for architecture/tokenizer",
        model_id, fallback,
    )
    return fallback


def _safetensors_stored_dtype(path: str) -> Optional[torch.dtype]:
    """
    Return the dominant weight dtype stored in the safetensors file.

    Embedding and norm layers are often kept in bf16 even in mixed-precision
    FP8 checkpoints.  We scan the first few *weight* tensors (which are the
    heavy linear projections) to find the actual compute dtype.
    """
    _FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Prefer tensors whose keys look like projection weights
            weight_keys = [k for k in keys if k.endswith(".weight")
                           and not any(s in k for s in
                                       ("embed_tokens", "layernorm", "norm",
                                        "embed_positions", "bias"))]
            # Fall back to all keys if no obvious weights found
            scan_keys = (weight_keys or keys)[:20]
            first_dtype = None
            for k in scan_keys:
                dt = f.get_tensor(k).dtype
                if first_dtype is None:
                    first_dtype = dt
                if dt in _FP8_DTYPES:
                    return dt          # found an FP8 tensor – report immediately
            # No FP8 tensor found; return dtype of the first scanned key
            return first_dtype
    except Exception:
        return None



def _materialize_meta_params(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> int:
    """
    Any parameter/buffer still on the meta device after load_state_dict was not
    present in the checkpoint.  Materialise them as zero tensors on *device* so
    that forward passes don't crash with "Tensor.item() cannot be called on meta
    tensors".
    Returns the number of tensors that were materialised.
    """
    count = 0
    for module in model.modules():
        for attr in list(module._parameters):
            p = module._parameters[attr]
            if p is not None and p.is_meta:
                module._parameters[attr] = torch.nn.Parameter(
                    torch.zeros(p.shape, dtype=dtype, device=device),
                    requires_grad=p.requires_grad,
                )
                count += 1
        for attr in list(module._buffers):
            b = module._buffers[attr]
            if b is not None and b.is_meta:
                module._buffers[attr] = torch.zeros(b.shape, dtype=dtype, device=device)
                count += 1
    return count


def _is_single_safetensors(model_id: str) -> bool:
    """Return True when model_id points directly to a .safetensors file."""
    return model_id.endswith(".safetensors") and os.path.isfile(model_id)


# ── CachedModel ───────────────────────────────────────────────────────────────

class CachedModel:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        family: str,
        device: torch.device,
        *,
        fp8: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.family = family
        self.device = device
        self.fp8 = fp8          # whether weights are stored in an FP8 dtype
        self.last_used: float = time.monotonic()

    def touch(self) -> None:
        self.last_used = time.monotonic()


# ── ModelManager ──────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, cfg: ServerConfig, device: torch.device):
        self._cfg = cfg
        self._device = device
        self._dtype = _resolve_dtype(cfg.dtype)
        self._cache: Dict[str, CachedModel] = {}
        self._lock = threading.Lock()

        # Background TTL reaper
        if not cfg.keep_models_loaded and cfg.model_ttl_seconds > 0:
            t = threading.Thread(target=self._reaper, daemon=True)
            t.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        model_id: str,
        texts: list[str],
        max_length: Optional[int] = None,
        return_pooled: bool = False,
    ) -> Dict[str, Any]:
        """
        Encode *texts* with *model_id* and return a dict with:
          - "embeddings": float32 numpy array [B, T, D]  (last hidden states)
          - "pooled":     float32 numpy array [B, D]     (if return_pooled=True)
        """
        cm = self._get_or_load(model_id)
        cm.touch()
        max_len = max_length or self._cfg.max_token_length

        with torch.inference_mode():
            if cm.family == "clip":
                return self._encode_clip(cm, texts, max_len, return_pooled)
            elif cm.family == "t5":
                return self._encode_t5(cm, texts, max_len, return_pooled)
            elif cm.family == "gemma":
                return self._encode_gemma(cm, texts, max_len, return_pooled)
            else:
                raise ValueError(f"Unknown model family: {cm.family!r}")

    def encode_dual(
        self,
        t5_model_id: str,
        gemma_model_id: str,
        texts: list[str],
        t5_max_length: int = 256,
        gemma_max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Encode *texts* with a T5 encoder and a Gemma encoder in a single call.

        Used by LTX-Video 2.3 which conditions the DiT on both:
          • T5-XXL hidden states  [B, T5_tokens, 4096]
          • Gemma-3-12B hidden states  [B, gemma_tokens, D_gemma]

        Returns a dict with keys:
          t5_embeddings, t5_pooled, t5_shape, t5_pooled_shape,
          gemma_embeddings, gemma_pooled, gemma_shape, gemma_pooled_shape
        """
        t5_result = self.encode(
            t5_model_id, texts, max_length=t5_max_length, return_pooled=True
        )
        gemma_result = self.encode(
            gemma_model_id, texts, max_length=gemma_max_length, return_pooled=True
        )
        return {
            "t5_embeddings": t5_result["embeddings"],
            "t5_pooled": t5_result.get("pooled", t5_result["embeddings"].mean(axis=1)),
            "gemma_embeddings": gemma_result["embeddings"],
            "gemma_pooled": gemma_result.get("pooled", gemma_result["embeddings"].mean(axis=1)),
        }

    def encode_gemma_raw(
        self,
        model_id: str,
        texts: list[str],
        max_length: int = 1024,
    ) -> Dict[str, Any]:
        """
        Encode *texts* with a Gemma model and return ALL hidden states.

        Intended for LTX-Video 2.3's ``LTXVRemoteCLIPLoader`` which applies the
        text-projection layer locally.

        Returns
        -------
        all_hidden : float16 numpy array [B, num_layers+1, T_nonpadded, D]
        pooled     : float32 numpy array [B, D]
        """
        cm = self._get_or_load(model_id)
        cm.touch()
        if cm.family != "gemma":
            raise ValueError(
                f"encode_gemma_raw requires a 'gemma' model family; "
                f"model {model_id!r} has family {cm.family!r}"
            )
        with torch.inference_mode():
            return self._encode_gemma_all_layers(cm, texts, max_length)

    def loaded_models(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def unload(self, model_id: str) -> bool:
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                self._gc()
                logger.info("Unloaded model: %s", model_id)
                return True
            return False

    def unload_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._gc()
            logger.info("All models unloaded.")

    # ── Private: loading ──────────────────────────────────────────────────────

    def _get_or_load(self, model_id: str) -> CachedModel:
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]

        logger.info("Loading model: %s  (dtype=%s, device=%s)", model_id, self._cfg.dtype, self._device)
        cm = self._load(model_id)

        with self._lock:
            self._cache[model_id] = cm

        return cm

    def _load(self, model_id: str) -> CachedModel:
        family = _detect_family(model_id)
        is_safetensors_file = _is_single_safetensors(model_id)
        source_dir = _config_source(model_id, family, self._cfg.model_cache_dir)
        fp8 = _is_fp8(self._dtype)

        # Detect whether the checkpoint itself is already stored in an FP8 dtype.
        # If so, we load as-is and skip the bf16 upcast + re-quantise cycle which
        # would temporarily double GPU memory usage.
        _FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
        stored_dtype = _safetensors_stored_dtype(model_id) if is_safetensors_file else None
        file_already_fp8 = stored_dtype in _FP8_DTYPES

        if file_already_fp8:
            # Keep the weights in their stored FP8 dtype; no conversion needed.
            load_dtype = stored_dtype
            logger.info("Checkpoint %s is already %s – skipping bf16 upcast", model_id, stored_dtype)
        elif fp8:
            # Weights are in a higher precision; load as bf16 then quantise.
            load_dtype = torch.bfloat16
        else:
            load_dtype = self._dtype

        hf_kwargs: Dict[str, Any] = {
            "cache_dir": self._cfg.model_cache_dir,
            "torch_dtype": load_dtype,
        }

        # When loading from a single safetensors file:
        #   1. Create the model on the "meta" device (no RAM allocated for weights)
        #   2. Load safetensors directly to the target device + dtype in one shot
        #   3. assign=True replaces meta tensors without an extra copy
        # This eliminates the CPU-RAM spike from random-init weights + state-dict
        # both being live simultaneously.
        if is_safetensors_file:
            cfg_cache = self._cfg.model_cache_dir
            arch_config = AutoConfig.from_pretrained(source_dir, cache_dir=cfg_cache)

            if family == "clip":
                tokenizer = CLIPTokenizer.from_pretrained(source_dir, cache_dir=cfg_cache)
                with init_empty_weights():
                    model = CLIPTextModel(arch_config)

            elif family == "t5":
                tokenizer = T5Tokenizer.from_pretrained(source_dir, cache_dir=cfg_cache)
                with init_empty_weights():
                    model = T5EncoderModel(arch_config)

            elif family == "gemma":
                tokenizer = AutoTokenizer.from_pretrained(
                    source_dir, cache_dir=cfg_cache, use_fast=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(arch_config)

            else:
                raise ValueError(f"Unsupported model family: {family!r}")

            logger.info("Loading weights from %s directly to %s (%s)", model_id, self._device, load_dtype)
            state_dict = safetensors_load_file(model_id, device=str(self._device))
            # Cast to target dtype in-place before assigning to avoid a second copy
            state_dict = {k: v.to(dtype=load_dtype) for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

            # ── Auto-remap key prefix if there's a total mismatch ─────────────
            # Symptom: all keys are both "missing" and "unexpected" – the model
            # and checkpoint use different top-level prefix conventions, e.g.:
            #   File:  model.layers.*          (standalone text encoder)
            #   Model: language_model.model.*  (Gemma3ForConditionalGeneration)
            if len(missing) > 0 and len(missing) == len(unexpected):
                model_keys  = set(model.state_dict().keys())
                file_keys   = set(state_dict.keys())
                # Find a prefix that, when prepended to file keys, lands in model keys.
                # Try the most common remapping: bare model.* → language_model.model.*
                candidate_prefixes = [
                    ("model.",    "language_model.model."),
                    ("model.",    "language_model."),
                    ("",          "language_model."),
                ]
                for src_pfx, dst_pfx in candidate_prefixes:
                    remapped = {
                        (dst_pfx + k[len(src_pfx):])
                        if k.startswith(src_pfx) else k: v
                        for k, v in state_dict.items()
                    }
                    # Accept remap if at least 50% of the *checkpoint* keys land in
                    # the model.  Using model key count as denominator is wrong for
                    # multimodal checkpoints (vision tower inflates the total).
                    overlap = len(set(remapped) & model_keys)
                    if overlap >= len(file_keys) * 0.5:
                        logger.info(
                            "Key prefix remapped: '%s' → '%s'  (%d/%d keys matched)",
                            src_pfx, dst_pfx, overlap, len(model_keys),
                        )
                        missing, unexpected = model.load_state_dict(
                            remapped, strict=False, assign=True
                        )
                        break
                else:
                    logger.warning(
                        "100%% key mismatch and no automatic prefix remap worked. "
                        "The checkpoint may use an unsupported key layout."
                    )
            if missing:
                logger.warning("%d missing keys in state dict (may be normal for tied weights)", len(missing))
            if unexpected:
                logger.warning("%d unexpected keys in state dict", len(unexpected))
            del state_dict
            # FP8 arithmetic is not implemented for embedding lookups
            # (Gemma3 multiplies hidden states by embed_scale in-place).  Keep
            # the embedding table in bfloat16 regardless of the load dtype.
            if fp8:
                emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
                if emb is not None and emb.weight.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
                    emb.weight = torch.nn.Parameter(emb.weight.to(torch.bfloat16))
            # Resolve tied weights (e.g. lm_head ↔ shared) so no meta tensors remain
            if hasattr(model, "tie_weights"):
                model.tie_weights()
            # Any keys that were absent from the checkpoint still live on meta
            # device – materialise them as zeros so forward() works.
            n_meta = _materialize_meta_params(model, self._device, load_dtype)
            if n_meta:
                logger.warning(
                    "%d parameters/buffers were missing from the checkpoint and "
                    "have been zero-initialised (model may produce degraded output)",
                    n_meta,
                )

        else:
            # low_cpu_mem_usage=True streams shards one at a time instead of
            # staging the full model in CPU RAM before moving to GPU.
            hf_kwargs["low_cpu_mem_usage"] = True

            if family == "clip":
                tokenizer = CLIPTokenizer.from_pretrained(source_dir, cache_dir=self._cfg.model_cache_dir)
                model = CLIPTextModel.from_pretrained(source_dir, **hf_kwargs)

            elif family == "t5":
                tokenizer = T5Tokenizer.from_pretrained(source_dir, cache_dir=self._cfg.model_cache_dir)
                model = T5EncoderModel.from_pretrained(source_dir, **hf_kwargs)

            elif family == "gemma":
                tokenizer = AutoTokenizer.from_pretrained(
                    source_dir,
                    cache_dir=self._cfg.model_cache_dir,
                    use_fast=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    source_dir,
                    torch_dtype=load_dtype,
                    cache_dir=self._cfg.model_cache_dir,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                )

            else:
                raise ValueError(f"Unsupported model family: {family!r}")

        # ── Cast to FP8 after loading (parameter-by-parameter) ────────────────
        # Skip if the checkpoint was already stored in FP8 – no conversion needed.
        if fp8 and not file_already_fp8:
            logger.info("Casting model weights to %s", self._dtype)
            model = _cast_to_fp8(model, self._dtype)

        model.eval()
        # Safetensors branch loads directly to the target device; only the
        # from_pretrained branch still needs an explicit device transfer.
        if not is_safetensors_file:
            model.to(self._device)
        effective_fp8 = fp8 or file_already_fp8
        logger.info("Model ready: %s  (family=%s, fp8=%s)", model_id, family, effective_fp8)
        return CachedModel(tokenizer, model, family, self._device, fp8=effective_fp8)

    # ── Private: encoding ─────────────────────────────────────────────────────

    def _encode_clip(
        self,
        cm: CachedModel,
        texts: list[str],
        max_length: int,
        return_pooled: bool,
    ) -> Dict[str, Any]:
        inputs = cm.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = cm.model(**inputs, output_hidden_states=True)

        # last hidden state: [B, T, D]
        embeddings = outputs.last_hidden_state.to(torch.float32).cpu().numpy()

        result: Dict[str, Any] = {"embeddings": embeddings}
        if return_pooled:
            # CLIPTextModel exposes pooler_output [B, D]
            pooled = outputs.pooler_output.to(torch.float32).cpu().numpy()
            result["pooled"] = pooled

        return result

    def _encode_t5(
        self,
        cm: CachedModel,
        texts: list[str],
        max_length: int,
        return_pooled: bool,
    ) -> Dict[str, Any]:
        inputs = cm.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = cm.model(**inputs)

        embeddings = outputs.last_hidden_state.to(torch.float32).cpu().numpy()

        result: Dict[str, Any] = {"embeddings": embeddings}
        if return_pooled:
            # T5 encoder has no dedicated pooler; mean-pool over non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (
                (outputs.last_hidden_state.to(torch.float32) * attention_mask.to(self._device)).sum(1)
                / attention_mask.sum(1).clamp(min=1e-9)
            )
            result["pooled"] = pooled.cpu().numpy()

        return result

    def _encode_gemma(
        self,
        cm: CachedModel,
        texts: list[str],
        max_length: int,
        return_pooled: bool,
    ) -> Dict[str, Any]:
        """
        Run Gemma-3 as a pure text encoder.

        Strategy
        --------
        • Tokenize with left-padding so the *last* real token is always at the
          final unmasked position – this makes the last-token hidden state a
          natural sentence embedding (GPT-style pooling).
        • Extract last_hidden_state [B, T, D] as the sequence embedding, and
          the hidden state at the last non-padding position as the pooled vector.
        • FP8 weights: the forward pass runs under torch.autocast(bf16) so
          activations and intermediate matmuls are in bf16.
        """
        # Left-pad so that the EOS / last real token lands at the final position
        cm.tokenizer.padding_side = "left"
        inputs = cm.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        forward_kwargs: Dict[str, Any] = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

        if cm.fp8:
            # Autocast to bf16 for compute – weights are decompressed on-the-fly
            # on hardware with native FP8 support (H100 / MI300X)
            ctx = torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)
        else:
            ctx = torch.autocast(device_type=self._device.type, enabled=False)

        with ctx:
            outputs = cm.model(**forward_kwargs)

        hidden: torch.Tensor = outputs.hidden_states[-1]  # [B, T, D] – last layer

        embeddings = hidden.to(torch.float32).cpu().numpy()
        result: Dict[str, Any] = {"embeddings": embeddings}

        if return_pooled:
            # Last-non-padding-token pooling (GPT-style)
            attn = inputs["attention_mask"]          # [B, T]
            # Index of the last '1' in each row
            last_idx = attn.cumsum(dim=1).argmax(dim=1)  # [B]
            pooled = hidden[torch.arange(hidden.size(0), device=self._device), last_idx]  # [B, D]
            result["pooled"] = pooled.to(torch.float32).cpu().numpy()

        return result

    def _encode_gemma_all_layers(
        self,
        cm: CachedModel,
        texts: list[str],
        max_length: int,
    ) -> Dict[str, Any]:
        """
        Run Gemma-3 and return ALL hidden states stacked.

        Returns
        -------
        all_hidden : float16 numpy array, shape [B, num_layers+1, T_nonpadded, D]
            All hidden states (embedding layer + every transformer layer) trimmed to
            only the non-padding token positions (left-padded tokenization, so real
            tokens sit at the tail of the sequence).
        pooled : float32 numpy array, shape [B, D]
            Last-layer hidden state at the last non-padding token (GPT-style pooling).
        """
        cm.tokenizer.padding_side = "left"
        inputs = cm.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        forward_kwargs: Dict[str, Any] = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )

        if cm.fp8:
            ctx = torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)
        else:
            ctx = torch.autocast(device_type=self._device.type, enabled=False)

        with ctx:
            outputs = cm.model(**forward_kwargs)

        # outputs.hidden_states: tuple of (num_layers+1) tensors, each [B, T, D]
        # Stack along dim 1 → [B, num_layers+1, T, D]
        all_hidden = torch.stack(list(outputs.hidden_states), dim=1)  # [B, L, T, D]

        # Trim to non-padding tokens.  With left-padding, real tokens are at the END.
        attn_mask = inputs["attention_mask"]                    # [B, T]  (0=pad, 1=real)
        n_nonpad = int(attn_mask.sum(dim=1).max().item())       # max real tokens in batch
        trimmed = all_hidden[:, :, -n_nonpad:, :]               # [B, L, n_nonpad, D]

        # Pooled: last-layer hidden state at the last non-padding token
        last_layer = outputs.hidden_states[-1]                  # [B, T, D]
        last_idx = attn_mask.cumsum(dim=1).argmax(dim=1)        # [B]
        pooled = last_layer[
            torch.arange(last_layer.size(0), device=self._device), last_idx
        ]                                                        # [B, D]

        return {
            # Store as float16 to halve transfer bandwidth; client casts back to float32
            "all_hidden": trimmed.to(torch.float16).cpu().numpy(),
            "pooled": pooled.to(torch.float32).cpu().numpy(),
        }

    # ── Private: memory management ────────────────────────────────────────────

    @staticmethod
    def _gc() -> None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reaper(self) -> None:
        """Periodically evict models that haven't been used recently."""
        while True:
            time.sleep(30)
            ttl = self._cfg.model_ttl_seconds
            if ttl <= 0:
                continue
            now = time.monotonic()
            with self._lock:
                stale = [mid for mid, cm in self._cache.items() if now - cm.last_used > ttl]
                for mid in stale:
                    del self._cache[mid]
                    logger.info("TTL-evicted model: %s", mid)
                if stale:
                    self._gc()
