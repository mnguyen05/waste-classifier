#!/usr/bin/env python3
"""
Seesaw servos + hook for pi_camera.py --sort-hook seesaw_servos:apply_sort_decision

Server / model uses "recycling" and "trash"; this file maps them to your tilt logic.
BCM pins: servo1=18, servo2=13 (same as your original script).
"""

from __future__ import annotations

import time

from gpiozero import AngularServo

# Seconds to hold tilt before returning to neutral (object slide time)
TILT_HOLD_SEC = 5.0

# ── Servo setup ─────────────────────────────────────────────
servo1 = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)
servo2 = AngularServo(13, min_pulse_width=0.0006, max_pulse_width=0.0023)


def set_both(angle1: float, angle2: float) -> None:
    servo1.angle = angle1
    servo2.angle = angle2


def detach_both() -> None:
    servo1.angle = None
    servo2.angle = None


def sort_object(label: str) -> None:
    """label is 'recycle' or 'trash' (internal names)."""
    if label == "recycle":
        print("Recycling — tilting to recycling bin", flush=True)
        set_both(-30, -55)
        time.sleep(TILT_HOLD_SEC)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("Platform reset.\n", flush=True)
    else:
        print("Trash — tilting to trash bin", flush=True)
        set_both(55, 40)
        time.sleep(TILT_HOLD_SEC)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("Platform reset.\n", flush=True)


def classify_object(payload: dict) -> str:
    """
    Map server JSON payload -> internal labels used by sort_object().
    Server returns "recycling" / "trash"; we map recycling -> recycle.
    """
    decision = (payload.get("decision") or payload.get("label") or "").strip().lower()
    if decision == "recycling":
        return "recycle"
    if decision == "trash":
        return "trash"
    raise ValueError(f"Unknown decision from server: {decision!r}")


def apply_sort_decision(payload: dict) -> None:
    """
    Called by pi_camera.py after the Mac returns JSON (or local inference).
    payload['decision'] / payload['label'] are 'recycling' or 'trash'.
    """
    try:
        label = classify_object(payload)
    except ValueError as e:
        print(f"[seesaw_servos] {e}", flush=True)
        return
    print(f"Classified as: {label}", flush=True)
    sort_object(label)


def _demo_loop() -> None:
    """Original Enter-to-simulate behavior (optional)."""
    print("Setting both servos to neutral...", flush=True)
    set_both(0, 0)
    time.sleep(1)
    detach_both()
    print("System ready.\n", flush=True)

    while True:
        raw = input("Type recycle or trash (q to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw not in {"recycle", "trash"}:
            print("Enter 'recycle' or 'trash'.", flush=True)
            continue
        label = raw
        sort_object(label)


if __name__ == "__main__":
    try:
        _demo_loop()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("Cleanup done.", flush=True)

