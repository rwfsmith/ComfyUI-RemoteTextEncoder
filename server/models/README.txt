# Place your local text encoder models here.
#
# Supported layouts
# -----------------
# 1. Model directory   models/my-clip-model/
#                          config.json         ← required
#                          tokenizer.json      ← required
#                          *.safetensors       ← weights
#
# 2. Single file       models/gemma3-12b-fp8.safetensors
#                      (parent dir must contain config.json + tokenizer files)
#
# Both will appear automatically in GET /v1/models and in the ComfyUI
# node dropdown.  No server restart needed; GET /v1/models rescans on
# every call.
