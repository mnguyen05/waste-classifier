#!/usr/bin/env python3
"""
Seesaw servos + hook for pi_camera.py --sort-hook seesaw_servos:apply_sort_decision

Server / model uses "recycling" and "trash"; this file maps them to your tilt logic.
BCM pins: servo1=18, servo2=13.

Servos usually need:
  - External 5V supply (not from Pi 3.3V), common GND with Pi, signal wires to GPIO.
  - pigpio daemon for stable pulses (recommended):
      sudo apt install -y pigpio python3-pigpio
      sudo pigpiod
    Then run pi_camera as usual. Without pigpiod, gpiozero falls back to the default
    backend and servos may barely move or not move.

GPIO 18 is used for PWM; on some Pis it conflicts with analog audio — if stuck, try
moving servos to other free BCM pins (e.g. 12, 19) and change BCM numbers below.
"""

from __future__ import annotations

import os
import random
import time

from gpiozero import AngularServo

# BCM pin numbers for servo signal wires
SERVO1_PIN = 18
SERVO2_PIN = 13

_MIN_PW = 0.0006
_MAX_PW = 0.0023

_factory = None
servo1: AngularServo | None = None
servo2: AngularServo | None = None


def _ensure_servos() -> None:
    """Create servos on first use; prefer pigpio for reliable PWM."""
    global _factory, servo1, servo2
    if servo1 is not None and servo2 is not None:
        return

    use_pigpio = os.environ.get("SEESAW_NO_PIGPIO", "").strip() not in ("1", "true", "yes")
    pin_kw: dict = {"min_pulse_width": _MIN_PW, "max_pulse_width": _MAX_PW}

    if use_pigpio:
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory

            _factory = PiGPIOFactory()
            pin_kw["pin_factory"] = _factory
            print(
                "[seesaw_servos] Using PiGPIOFactory (pigpiod). If servos are still dead, "
                "run: sudo pigpiod",
                flush=True,
            )
        except (ImportError, OSError, Exception) as e:
            _factory = None
            print(
                f"[seesaw_servos] pigpio unavailable ({e!r}); using default GPIO backend — "
                "servos often need: sudo apt install pigpio && sudo pigpiod",
                flush=True,
            )
    else:
        print("[seesaw_servos] SEESAW_NO_PIGPIO set — using default pin factory.", flush=True)

    servo1 = AngularServo(SERVO1_PIN, **pin_kw)
    servo2 = AngularServo(SERVO2_PIN, **pin_kw)
    print(
        f"[seesaw_servos] Servos on BCM {SERVO1_PIN} / {SERVO2_PIN} ready.",
        flush=True,
    )


def set_both(angle1: float, angle2: float) -> None:
    _ensure_servos()
    assert servo1 is not None and servo2 is not None
    servo1.angle = angle1
    servo2.angle = angle2


def detach_both() -> None:
    if servo1 is None or servo2 is None:
        return
    servo1.angle = None
    servo2.angle = None


def sort_object(label: str) -> None:
    """label is 'recycle' or 'trash' (internal names)."""
    _ensure_servos()
    if label == "recycle":
        print("[seesaw_servos] Recycling — tilting", flush=True)
        set_both(-30, -55)
        time.sleep(1.5)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("[seesaw_servos] Platform reset (recycling).\n", flush=True)
    else:
        print("[seesaw_servos] Trash — tilting", flush=True)
        set_both(55, 40)
        time.sleep(1.5)
        set_both(0, 0)
        time.sleep(0.5)
        detach_both()
        print("[seesaw_servos] Platform reset (trash).\n", flush=True)


def apply_sort_decision(payload: dict) -> None:
    """
    Called by pi_camera.py after the Mac returns JSON (or local inference).
    payload['decision'] / payload['label'] are 'recycling' or 'trash'.
    """
    print(
        f"[seesaw_servos] apply_sort_decision keys={list(payload.keys())} "
        f"decision={payload.get('decision')!r} label={payload.get('label')!r}",
        flush=True,
    )
    decision = (payload.get("decision") or payload.get("label") or "").strip().lower()
    if decision == "recycling":
        sort_object("recycle")
    elif decision == "trash":
        sort_object("trash")
    else:
        print(f"[seesaw_servos] unknown decision: {decision!r}", flush=True)


def _demo_loop() -> None:
    """Original Enter-to-simulate behavior (optional)."""

    def classify_object() -> str:
        return random.choice(["recycle", "trash"])

    print("[seesaw_servos] Demo: setting neutral...", flush=True)
    set_both(0, 0)
    time.sleep(1)
    detach_both()
    print("[seesaw_servos] System ready.\n", flush=True)

    while True:
        input("Press Enter to simulate dropping an object...")
        label = classify_object()
        print(f"Classified as: {label}", flush=True)
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
