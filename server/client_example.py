"""
Example client that calls the Remote Text Encoder server.

Usage:
    python client_example.py --server http://localhost:8288
"""

import argparse
import base64

import numpy as np
import requests


def decode_b64_array(b64: str, shape: list[int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)


def encode_texts(server: str, model: str, texts: list[str], api_key: str | None = None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "texts": texts,
        "return_pooled": True,
    }

    resp = requests.post(f"{server}/encode", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    embeddings = decode_b64_array(data["embeddings_b64"], data["shape"])
    print(f"Model      : {data['model']}")
    print(f"Shape      : {embeddings.shape}  (batch × tokens × hidden_dim)")
    print(f"dtype      : {embeddings.dtype}")

    if data.get("pooled_b64"):
        pooled = decode_b64_array(data["pooled_b64"], [len(texts), embeddings.shape[-1]])
        print(f"Pooled     : {pooled.shape}")

    return embeddings


def comfy_encode(server: str, model: str, text: str, api_key: str | None = None):
    """Demonstrates the ComfyUI-compatible /comfy/encode endpoint."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model_name": model, "text": text, "clip_skip": 1}
    resp = requests.post(f"{server}/comfy/encode", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8288")
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--texts", nargs="+", default=["a photo of a cat", "a photo of a dog"])
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    print("=== /encode endpoint ===")
    embeddings = encode_texts(args.server, args.model, args.texts, args.api_key)

    print("\n=== /comfy/encode endpoint ===")
    result = comfy_encode(args.server, args.model, args.texts[0], args.api_key)
    shape = result["shape"]
    emb = decode_b64_array(result["cond"][0], shape)
    print(f"Embedding shape : {emb.shape}")
    pooled = decode_b64_array(result["cond"][1]["pooled_output"], result["pooled_shape"])
    print(f"Pooled shape    : {pooled.shape}")
