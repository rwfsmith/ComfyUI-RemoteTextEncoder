# Remote Text Encoder Server for ComfyUI

A FastAPI-based server that offloads CLIP and T5 text encoding to a remote GPU machine, supporting both **NVIDIA CUDA** and **AMD ROCm** backends.

## Architecture

```
ComfyUI (client)
    │
    │  HTTP POST /encode  or  /comfy/encode
    ▼
Remote Text Encoder Server  (this project)
    │
    ├─ ModelManager  – loads & caches CLIP / T5 models
    ├─ device.py     – auto-detects CUDA or ROCm
    └─ GPU (NVIDIA or AMD)
```

## Supported Models

| Model | HF repo-id | Family |
|---|---|---|
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | clip |
| CLIP ViT-bigG | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | clip |
| T5-XXL | `google/t5-v1_1-xxl` | t5 |
| T5-XL | `google/t5-v1_1-xl` | t5 |
| Flan-T5-XXL | `google/flan-t5-xxl` | t5 |

Any HF-compatible CLIP or T5 encoder can be used by passing its repo-id.

---

## Installation

### 1. Clone / copy the project

```bash
git clone <repo>  # or copy files
cd RemoteTextEncoder
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch for your GPU backend

#### NVIDIA CUDA (12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### AMD ROCm 6.0
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

#### CPU only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` already lists `torch` as a dependency for convenience, but if you installed a specific CUDA/ROCm wheel first, pip will respect it.

---

## Running the Server

```bash
python server.py
```

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8288` | Port |
| `--device` | `auto` | `auto` \| `cuda` \| `rocm` \| `cpu` |
| `--dtype` | `fp16` | `fp32` \| `fp16` \| `bf16` |
| `--api-key` | _(none)_ | Require Bearer token |
| `--no-keep-loaded` | – | Evict models after TTL |
| `--model-ttl` | `300` | Idle seconds before eviction |
| `--log-level` | `INFO` | Logging verbosity |

### Examples

```bash
# CUDA, fp16 (default)
python server.py --port 8288

# ROCm AMD GPU
python server.py --device rocm --dtype fp16

# CPU with auth
python server.py --device cpu --api-key mysecret

# Keep models loaded, BF16 (good for A100 / MI300X)
python server.py --dtype bf16
```

### Environment variables

Every flag has an `RTE_*` environment variable equivalent:

```bash
export RTE_DEVICE=rocm
export RTE_DTYPE=bf16
export RTE_API_KEY=mysecret
export RTE_PORT=8288
python server.py
```

---

## API Reference

### `GET /`
Health check.

### `GET /info`
Returns device info and server configuration.

### `GET /models`
Lists currently loaded (in-VRAM) models.

### `POST /encode`
Main encoding endpoint.

**Request body:**
```json
{
  "model": "openai/clip-vit-large-patch14",
  "texts": ["a photo of a cat", "a photo of a dog"],
  "max_length": 77,
  "return_pooled": true
}
```

**Response:**
```json
{
  "model": "openai/clip-vit-large-patch14",
  "shape": [2, 77, 768],
  "dtype": "float32",
  "embeddings_b64": "<base64 raw float32 bytes>",
  "pooled_b64": "<base64 raw float32 bytes>"
}
```

**Decoding in Python:**
```python
import base64, numpy as np

raw = base64.b64decode(response["embeddings_b64"])
embeddings = np.frombuffer(raw, dtype=np.float32).reshape(response["shape"])
# shape: [batch, tokens, hidden_dim]
```

### `POST /comfy/encode`
ComfyUI-compatible endpoint matching the `CLIPTextEncodeRemote` custom node protocol.

**Request:**
```json
{
  "model_name": "openai/clip-vit-large-patch14",
  "text": "a beautiful landscape",
  "clip_skip": 1
}
```

**Response:**
```json
{
  "cond": ["<embeddings_b64>", {"pooled_output": "<pooled_b64>"}],
  "shape": [77, 768],
  "pooled_shape": [768]
}
```

### `DELETE /models/{model_id}`
Unload a specific model from VRAM.

### `DELETE /models`
Unload all models from VRAM.

---

## ComfyUI Integration

1. Start the server on your GPU machine (default port `8288`).
2. In ComfyUI, install a **Remote CLIP** custom node (e.g., `ComfyUI-RemoteCLIP`).
3. Point it at `http://<server-ip>:8288/comfy/encode`.
4. Select your model name matching the HF repo-id loaded on the server.

Alternatively, write a custom node that calls `POST /encode` directly and converts the base64 tensors back into `torch.Tensor` objects for the conditioning output.

---

## Security

- Set `--api-key <token>` (or `RTE_API_KEY`) to require authentication.
- Clients must send `Authorization: Bearer <token>` with every request.
- Use a reverse proxy (nginx, Caddy) with TLS for remote/internet-facing deployments.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA not available` on ROCm | Ensure PyTorch ROCm wheel is installed; `torch.cuda.is_available()` returns `True` under HIP |
| OOM during model load | Use `--dtype fp16` or `--dtype bf16`; or reduce `--model-ttl` to free VRAM sooner |
| Slow first request | Models download from HF Hub on first use; subsequent requests use the cache |
| `sentencepiece` import error | `pip install sentencepiece protobuf` |
