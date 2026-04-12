"""
Example sort hook for the Pi — copy to your own file (e.g. my_seesaw.py) and wire servos.

Run:
  python3 pi_camera.py --ir-loop --server http://MAC:8000 --gpio-pin 24 \\
    --sort-hook my_seesaw:apply_sort_decision

`payload` is the same dict as the JSON saved under results/ (server adds image_sha256, etc.).
"""

from __future__ import annotations


def apply_sort_decision(payload: dict) -> None:
    decision = (payload.get("decision") or payload.get("label") or "").strip().lower()
    if decision == "trash":
        _tilt_seesaw_trash()
    elif decision == "recycling":
        _tilt_seesaw_recycling()
    else:
        print(f"[my_seesaw] unknown decision: {decision!r}", flush=True)


def _tilt_seesaw_trash() -> None:
    """Replace with your servo code: one side up, other down."""
    print("[my_seesaw] TRASH tilt (stub)", flush=True)


def _tilt_seesaw_recycling() -> None:
    print("[my_seesaw] RECYCLING tilt (stub)", flush=True)
