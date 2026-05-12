"""
Server discovery helper for the Remote Text Encoder ComfyUI nodes.

The model dropdown is populated at startup by querying GET /v1/models on the
configured server.  Edit ``rte_config.json`` (next to this file) to set the
server URL so the dropdown is populated without needing to run the node first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger("comfyui.remote_text_encoder.discovery")

MODEL_PLACEHOLDER = "<no models found – check server URL>"

_CONFIG_FILE = Path(__file__).with_name("rte_config.json")
_DEFAULT_SERVER_URL = "http://localhost:8288"


def _load_server_url() -> str:
    try:
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            url = data.get("server_url", "").strip()
            if url.startswith("http"):
                return url
    except Exception as exc:
        logger.warning("Could not read rte_config.json: %s", exc)
    return _DEFAULT_SERVER_URL


def _save_server_url(url: str) -> None:
    try:
        _CONFIG_FILE.write_text(
            json.dumps({"server_url": url.strip()}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Could not save rte_config.json: %s", exc)


def _build_headers(api_key: str) -> dict:
    h: dict = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def fetch_models(
    server_url: str,
    api_key: str = "",
    timeout: int = 10,
) -> dict[str, str]:
    """
    Call GET /v1/models on the Remote Text Encoder server.

    Returns a ``{short_name: full_path}`` dict so the caller can populate a
    dropdown with short names while still sending the correct full path to the
    server for ``/encode`` requests.

    Returns ``{MODEL_PLACEHOLDER: ""}`` when the server is unreachable.
    """
    url = server_url.rstrip("/") + "/v1/models"
    try:
        resp = requests.get(url, headers=_build_headers(api_key), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        result: dict[str, str] = {}
        for obj in data.get("data", []):
            short = obj.get("id", "")
            full = obj.get("path", short)  # server sets path=full abs path
            if short:
                result[short] = full or short
        return result if result else {MODEL_PLACEHOLDER: ""}
    except Exception as exc:
        logger.warning(
            "RTE model discovery failed for %s – %s  "
            "(Set the correct URL in the node widget and press R to retry.)",
            url, exc,
        )
        return {MODEL_PLACEHOLDER: ""}


# ── Cache ───────────────────────────────────────────────────────────────────

_cached_models_map: dict[str, str] = {}
_cache_server_url: str = _load_server_url()  # pre-seed from config so R works immediately


def get_models_cached(
    server_url: str, api_key: str = "", timeout: int = 30
) -> dict[str, str]:
    """Return cached ``{short_name: full_path}`` dict, refreshing when the URL changes."""
    global _cached_models_map, _cache_server_url
    if server_url != _cache_server_url or not _cached_models_map:
        _cached_models_map = fetch_models(server_url, api_key, timeout)
        _cache_server_url = server_url
    return _cached_models_map


def refresh_models(
    server_url: str, api_key: str = "", timeout: int = 30
) -> dict[str, str]:
    """Always hit the server (called when the user presses R on the node)."""
    global _cached_models_map, _cache_server_url
    _cached_models_map = fetch_models(server_url, api_key, timeout)
    _cache_server_url = server_url
    _save_server_url(server_url)
    return _cached_models_map
