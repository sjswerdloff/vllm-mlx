#!/usr/bin/env python3
"""Tests for mlx_schedulers — no model loading required."""

from mlx_schedulers import CosineDecayWithWarmup, ReduceOnPlateau, WarmupStableDecay


def test_cosine_warmup_phase():
    s = CosineDecayWithWarmup(peak_lr=1e-3, total_steps=1000, warmup_steps=100)
    assert s(0) == 0.0, "Step 0 should be 0"
    assert abs(s(50) - 0.5e-3) < 1e-8, "Step 50 should be half peak"
    assert abs(s(100) - 1e-3) < 1e-8, "Step 100 should be peak"
    print("  warmup phase: OK")


def test_cosine_decay_phase():
    s = CosineDecayWithWarmup(peak_lr=1e-3, total_steps=1000, warmup_steps=0)
    assert abs(s(0) - 1e-3) < 1e-8, "Step 0 should be peak (no warmup)"
    mid_lr = s(500)
    assert abs(mid_lr - 0.5e-3) < 1e-6, f"Step 500 should be ~half peak, got {mid_lr}"
    end_lr = s(1000)
    assert end_lr < 1e-7, f"Step 1000 should be ~0, got {end_lr}"
    print("  decay phase: OK")


def test_cosine_min_lr():
    s = CosineDecayWithWarmup(peak_lr=1e-3, total_steps=100, warmup_steps=0, min_lr=1e-5)
    end_lr = s(100)
    assert abs(end_lr - 1e-5) < 1e-8, f"Should reach min_lr at end, got {end_lr}"
    print("  min_lr floor: OK")


def test_cosine_auto_step():
    s = CosineDecayWithWarmup(peak_lr=1e-3, total_steps=100, warmup_steps=10)
    lrs = [s() for _ in range(101)]
    assert lrs[0] == 0.0
    assert abs(lrs[10] - 1e-3) < 1e-8
    assert lrs[100] < 1e-7
    # Should be monotonically non-increasing after warmup
    for i in range(11, 101):
        assert lrs[i] <= lrs[i - 1] + 1e-10, f"Step {i}: {lrs[i]} > {lrs[i-1]}"
    print("  auto-step monotonic: OK")


def test_plateau_no_reduce_when_improving():
    s = ReduceOnPlateau(initial_lr=1e-3, patience=10, window=5)
    # Steadily decreasing loss — should never reduce
    for i in range(100):
        reduced = s.step(10.0 - i * 0.1)
        assert not reduced, f"Should not reduce at step {i}"
    assert s.lr == 1e-3
    print("  no reduce when improving: OK")


def test_plateau_reduces_on_flat():
    s = ReduceOnPlateau(initial_lr=1e-3, factor=0.5, patience=20, window=5, cooldown=0)
    # Flat loss — should reduce after patience
    for i in range(100):
        s.step(5.0)
    assert s.lr < 1e-3, f"Should have reduced, got {s.lr}"
    assert s.num_reductions > 0
    print(f"  reduces on flat: OK (lr={s.lr:.6f}, reductions={s.num_reductions})")


def test_plateau_respects_min_lr():
    s = ReduceOnPlateau(initial_lr=1e-3, factor=0.1, patience=10, window=5, min_lr=1e-5, cooldown=0)
    for i in range(500):
        s.step(5.0)
    assert s.lr >= 1e-5, f"Should not go below min_lr, got {s.lr}"
    print(f"  min_lr respected: OK (lr={s.lr:.6f})")


def test_plateau_cooldown():
    s = ReduceOnPlateau(initial_lr=1e-3, factor=0.5, patience=10, window=5, cooldown=20)
    # Flat loss
    reductions = []
    for i in range(200):
        if s.step(5.0):
            reductions.append(i)
    # Reductions should be spaced by at least cooldown + patience
    if len(reductions) >= 2:
        gap = reductions[1] - reductions[0]
        assert gap >= 30, f"Gap between reductions should be >= 30 (cooldown+patience), got {gap}"
    print(f"  cooldown spacing: OK (reductions at steps {reductions[:5]})")


def test_wsd_three_phases():
    s = WarmupStableDecay(peak_lr=1e-3, total_steps=1000, warmup_steps=100, decay_fraction=0.1)
    # Warmup
    assert s(0) == 0.0
    assert abs(s(100) - 1e-3) < 1e-8
    # Stable (steps 100-900)
    assert abs(s(500) - 1e-3) < 1e-8, f"Stable phase should be peak, got {s(500)}"
    assert abs(s(899) - 1e-3) < 1e-8
    # Decay (steps 900-1000)
    mid_decay = s(950)
    assert mid_decay < 1e-3 and mid_decay > 0, f"Mid-decay should be between 0 and peak, got {mid_decay}"
    end = s(1000)
    assert end < 1e-6, f"End should be ~0, got {end}"
    print("  three phases: OK")


def test_wsd_min_lr():
    s = WarmupStableDecay(peak_lr=1e-3, total_steps=100, warmup_steps=0, decay_fraction=0.2, min_lr=1e-5)
    assert abs(s(100) - 1e-5) < 1e-8, f"Should reach min_lr at end, got {s(100)}"
    print("  min_lr floor: OK")


def test_wsd_auto_step():
    s = WarmupStableDecay(peak_lr=1e-3, total_steps=100, warmup_steps=10, decay_fraction=0.1)
    lrs = [s() for _ in range(101)]
    # Warmup: increasing
    for i in range(1, 10):
        assert lrs[i] >= lrs[i - 1] - 1e-10
    # Stable: constant
    for i in range(11, 90):
        assert abs(lrs[i] - 1e-3) < 1e-8, f"Step {i}: {lrs[i]} should be peak"
    # Decay: decreasing
    for i in range(91, 101):
        assert lrs[i] <= lrs[i - 1] + 1e-10, f"Step {i}: {lrs[i]} > {lrs[i-1]}"
    print("  auto-step phases: OK")


if __name__ == "__main__":
    print("CosineDecayWithWarmup:")
    test_cosine_warmup_phase()
    test_cosine_decay_phase()
    test_cosine_min_lr()
    test_cosine_auto_step()

    print("\nWarmupStableDecay:")
    test_wsd_three_phases()
    test_wsd_min_lr()
    test_wsd_auto_step()

    print("\nReduceOnPlateau:")
    test_plateau_no_reduce_when_improving()
    test_plateau_reduces_on_flat()
    test_plateau_respects_min_lr()
    test_plateau_cooldown()

    print("\nAll tests passed.")
