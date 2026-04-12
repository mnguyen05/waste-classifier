"""
Seesaw servos for the Pi — use with pi_camera.py:

  python3 pi_camera.py --ir-loop --server http://MAC:8000 --gpio-pin 24 \\
    --sort-hook pi_seesaw_servos:apply_sort_decision

The server returns label/decision "trash" or "recycling"; this maps recycling → your recycle tilt.
"""

from __future__ import annotations

import time

from gpiozero import AngularServo

# Servo 1 — left side of platform (BCM pin numbers)
servo1 = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)
# Servo 2 — right side of platform
servo2 = AngularServo(13, min_pulse_width=0.0006, max_pulse_width=0.0023)


def set_both(angle1, angle2):
    servo1.angle = angle1
    servo2.angle = angle2


def detach_both():
    servo1.angle = None
    servo2.angle = None


def sort_object(label: str) -> None:
    """label is 'recycle' or 'trash' (your original convention)."""
    if label == "recycle":
        print("Recycling — tilting to recycling bin", flush=True)
        set_both(-30, -55)
        time.sleep(1.5)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("Platform reset.\n", flush=True)
    else:
        print("Trash — tilting to trash bin", flush=True)
        set_both(55, 40)
        time.sleep(1.5)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("Platform reset.\n", flush=True)


def apply_sort_decision(payload: dict) -> None:
    """
    Called by pi_camera.py after Mac (or local) inference.
    payload['decision'] / payload['label'] are 'trash' or 'recycling'.
    """
    raw = (payload.get("decision") or payload.get("label") or "").strip().lower()
    if raw == "recycling":
        sort_object("recycle")
    elif raw == "trash":
        sort_object("trash")
    else:
        print(f"[pi_seesaw_servos] unknown decision: {raw!r}", flush=True)


def neutral_at_startup() -> None:
    set_both(0, 0)
    time.sleep(1)
    detach_both()
