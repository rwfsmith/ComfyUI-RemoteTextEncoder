# Remote Text Encoder

Offload CLIP / T5 / Gemma text encoding to a separate GPU machine and use the
results directly in ComfyUI.  Frees the ComfyUI machine from having to load
large text encoder models locally.

```
RemoteTextEncoder/
├── server/    ← run this on your GPU / encoding machine
└── (root)     ← install this folder as a ComfyUI custom node
```

---

## ComfyUI Node – Installation

1. Clone or copy this folder into your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/RemoteTextEncoder/
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your server URL** in `rte_config.json` (in the node folder) so the
   model dropdown populates automatically on startup:
   ```json
   { "server_url": "http://your-server-hostname:8288" }
   ```
   Without this step the dropdown will be empty until the node is executed
   once — after that the URL is saved automatically for future restarts.

4. Restart ComfyUI.

Press **R** on the **Remote CLIP Loader** node at any time to refresh the
model list (e.g. after loading a new model on the server).

---

## Server – Installation

See [server/README.md](server/README.md) for full details.

```bash
cd server
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA
# or: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0  # ROCm
pip install -r requirements.txt
python server.py --port 8288
```

Place models in `server/models/` — see `server/models/README.txt` for the
supported layouts (HF model directory or single `.safetensors` file).

---

## Node Reference

| Node | Inputs | Output |
|---|---|---|
| **Remote CLIP Loader** | `server_url`, `model_name`, `clip_skip`, `timeout`, `api_key` | `REMOTE_CLIP` |
| **CLIP Text Encode (Remote)** | `remote_clip`, `text`, `max_length` | `CONDITIONING` |
| **CLIP Text Encode Couple (Remote / SDXL)** | `remote_clip_l`, `remote_clip_g`, `text_l`, `text_g` | `CONDITIONING` |
| **LTX-Video Text Encode (Remote)** | `remote_t5`, `remote_gemma`, `text`, `t5_max_length`, `gemma_max_length` | `CONDITIONING` |

### SD 1.x / SD 2.x
```
RemoteCLIPLoader ──► CLIPTextEncodeRemote (positive) ──┐
RemoteCLIPLoader ──► CLIPTextEncodeRemote (negative) ──┤
                                                        ▼
                              KSampler ◄── VAELoader, CheckpointLoader …
```

### SDXL
```
RemoteCLIPLoader (clip-l) ──┐
                             ├──► CLIPTextEncodeCoupleRemote (positive) ──┐
RemoteCLIPLoader (clip-g) ──┘                                             │
                                                                           ▼
RemoteCLIPLoader (clip-l) ──┐                                           KSampler
                             ├──► CLIPTextEncodeCoupleRemote (negative) ──┘
RemoteCLIPLoader (clip-g) ──┘
```

### LTX-Video 2.3
```
RemoteCLIPLoader (T5-XXL)      ──┐
                                  ├──► LTXVTextEncodeRemote ──► CONDITIONING
RemoteCLIPLoader (Gemma-3-12B) ──┘
```
