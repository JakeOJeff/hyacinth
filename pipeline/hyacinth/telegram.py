"""Post the daily video + heatmap to a public Telegram channel (Bot API)."""

from __future__ import annotations

import os
from pathlib import Path

import requests

API = "https://api.telegram.org/bot{token}/{method}"
CAPTION_LIMIT = 1024  # Telegram media caption limit (characters)


def _call(token: str, method: str, data: dict, files: dict) -> dict:
    r = requests.post(API.format(token=token, method=method), data=data, files=files, timeout=300)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {body.get('description', r.status_code)}")
    return body["result"]


def post(out_dir: Path, caption: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")  # e.g. @hyacinth_forecast
    if not token or not chat:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; skipping post")
        return False
    if len(caption) > CAPTION_LIMIT:
        caption = caption[: CAPTION_LIMIT - 1] + "…"
    video = out_dir / "forecast.mp4"
    heat = out_dir / "heatmap.png"
    if video.exists():
        with video.open("rb") as f:
            _call(token, "sendVideo", {"chat_id": chat, "caption": caption, "supports_streaming": "true"},
                  {"video": f})
    else:
        print("[telegram] no forecast.mp4; posting the heatmap only")
    with heat.open("rb") as f:
        _call(token, "sendPhoto",
              {"chat_id": chat, "caption": caption if not video.exists() else "Jam heatmap, next 72 h"},
              {"photo": f})
    print(f"[telegram] posted to {chat}")
    return True
