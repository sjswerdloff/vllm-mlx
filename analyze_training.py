#!/usr/bin/env python3
"""Analyze training loss from JSONL log files.

Computes moving averages at multiple window sizes and reports
trend, plateau detection, and per-window min/max.

Usage:
  python analyze_training.py [path/to/training.jsonl]

Defaults to ~/models/qwopus-merger-checkpoints/training.jsonl

Author: Clement (clement-7074f29f)
"""

import json
import os
import sys


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def moving_average(data: list[dict], window: int) -> list[dict]:
    results = []
    for i in range(0, len(data), window):
        chunk = data[i : i + window]
        if len(chunk) < window // 2:
            break
        losses = [d["loss"] for d in chunk]
        results.append(
            {
                "step": chunk[-1]["step"],
                "avg": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "count": len(chunk),
            }
        )
    return results


def analyze(path: str):
    data = load_data(path)
    if not data:
        print("No data found.")
        return

    wall_h = data[-1].get("wall_elapsed", 0) / 3600
    print(f"File: {path}")
    print(f"Steps: {len(data)} | Last step: {data[-1]['step']} | Wall time: {wall_h:.1f}h")
    print(f"Loss: first={data[0]['loss']:.3f} last={data[-1]['loss']:.3f}")
    print()

    for window in [50, 200]:
        avgs = moving_average(data, window)
        if len(avgs) < 2:
            continue

        print(f"Moving Average (window={window})")
        print(f"{'Step':>6} | {'Avg':>8} | {'Min':>6} | {'Max':>6}")
        print("-" * 38)
        for a in avgs:
            print(f"{a['step']:6d} | {a['avg']:8.3f} | {a['min']:6.2f} | {a['max']:6.2f}")

        # Trend: compare first half avg to second half avg
        mid = len(avgs) // 2
        first_half = sum(a["avg"] for a in avgs[:mid]) / mid
        second_half = sum(a["avg"] for a in avgs[mid:]) / (len(avgs) - mid)
        delta = second_half - first_half
        pct = 100 * delta / first_half if first_half != 0 else 0

        if abs(pct) < 2:
            trend = "PLATEAU"
        elif delta < 0:
            trend = "DESCENDING"
        else:
            trend = "RISING"

        print(f"\nTrend: {trend} ({pct:+.1f}%)")
        print(f"  First half avg: {first_half:.3f}")
        print(f"  Second half avg: {second_half:.3f}")
        print()


if __name__ == "__main__":
    default = os.path.expanduser(
        "~/models/qwopus-merger-checkpoints/training.jsonl"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    analyze(path)
