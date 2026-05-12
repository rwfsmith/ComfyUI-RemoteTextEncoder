"""
ComfyUI Remote Text Encoder – Custom Node Package
==================================================

Provides nodes that route text encoding through the Remote Text Encoder
server instead of a local GPU:

  RemoteCLIPLoader          →  REMOTE_CLIP       (single encoder)
  RemoteDualCLIPLoader      →  REMOTE_DUAL_CLIP  (two encoders in one wire)
  CLIPTextEncodeRemote      →  CONDITIONING      (single encoder)
  CLIPTextEncodeCoupleRemote  →  CONDITIONING    (SDXL: clip-l + clip-g)
  LTXVTextEncodeRemote      →  CONDITIONING      (LTX-Video: T5 + Gemma)
"""

from .nodes import (
    CLIPTextEncodeCoupleRemote,
    CLIPTextEncodeRemote,
    LTXVRemoteCLIPLoader,
    LTXVTextEncodeRemote,
    RemoteCLIPLoader,
    RemoteDualCLIPLoader,
)

NODE_CLASS_MAPPINGS = {
    "RemoteCLIPLoader": RemoteCLIPLoader,
    "RemoteDualCLIPLoader": RemoteDualCLIPLoader,
    "CLIPTextEncodeRemote": CLIPTextEncodeRemote,
    "CLIPTextEncodeCoupleRemote": CLIPTextEncodeCoupleRemote,
    "LTXVTextEncodeRemote": LTXVTextEncodeRemote,
    "LTXVRemoteCLIPLoader": LTXVRemoteCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCLIPLoader": "Remote CLIP Loader",
    "RemoteDualCLIPLoader": "Remote Dual CLIP Loader",
    "CLIPTextEncodeRemote": "CLIP Text Encode (Remote)",
    "CLIPTextEncodeCoupleRemote": "CLIP Text Encode Couple (Remote / SDXL)",
    "LTXVTextEncodeRemote": "LTX-Video Text Encode (Remote / T5 + Gemma)",
    "LTXVRemoteCLIPLoader": "LTX-V Remote CLIP Loader (Gemma + local projection)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
