"""Learning rate schedulers for MLX training.

Standalone module — no dependencies beyond standard library.
Can be copied to any MLX training project.

Schedulers:
  CosineDecayWithWarmup — Standard choice for fine-tuning and projector
    training (LLaVA, etc). Use when you know the total number of steps
    upfront. Smoothly decays LR from peak to min_lr over the run.

  WarmupStableDecay (WSD) — From MiniCPM (2024). Constant LR for most
    of training, rapid linear decay at the end. Less wasted compute
    than cosine's long tail. Use when you want most steps at peak LR.

  ReduceOnPlateau — Adaptive choice when you don't know how long to
    train. Monitors the loss moving average and drops LR when it stops
    improving. Good for exploratory runs. Legacy for transformers —
    cosine or WSD preferred.

Usage — Cosine (callable, pass directly as learning_rate):

    from mlx_schedulers import CosineDecayWithWarmup
    import mlx.optimizers as optim

    scheduler = CosineDecayWithWarmup(
        peak_lr=1e-4, total_steps=5000, warmup_steps=200, min_lr=1e-6
    )
    optimizer = optim.AdamW(learning_rate=scheduler)

    for step in range(5000):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        # Scheduler advances automatically — nothing else needed

Usage — WSD (callable, same as cosine):

    from mlx_schedulers import WarmupStableDecay
    import mlx.optimizers as optim

    scheduler = WarmupStableDecay(
        peak_lr=1e-4, total_steps=5000, warmup_steps=200,
        decay_fraction=0.1, min_lr=1e-6
    )
    optimizer = optim.AdamW(learning_rate=scheduler)
    # Last 10% of steps decay linearly; the rest is at peak LR

Usage — Plateau (stateful, must step explicitly):

    from mlx_schedulers import ReduceOnPlateau
    import mlx.optimizers as optim

    scheduler = ReduceOnPlateau(
        initial_lr=1e-4, factor=0.5, patience=200, min_lr=1e-6, window=50
    )
    optimizer = optim.AdamW(learning_rate=scheduler.lr)

    for step in range(max_steps):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())

        # Feed loss to scheduler — it updates lr if plateau detected
        if scheduler.step(loss.item()):
            optimizer.learning_rate = scheduler.lr
            print(f"LR reduced to {scheduler.lr:.2e}")

Author: Clement (clement-7074f29f)
"""

import math


class CosineDecayWithWarmup:
    """Linear warmup then cosine decay.

    Standard choice for projector/fine-tuning training (LLaVA, etc).
    Returns LR as a callable — pass directly as optimizer learning_rate.
    """

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self._step = 0

    def __call__(self, step=None):
        """Return LR for current or given step. Advances internal counter."""
        if step is None:
            step = self._step
            self._step += 1

        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / max(self.warmup_steps, 1)

        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        progress = min((step - self.warmup_steps) / max(decay_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


class WarmupStableDecay:
    """Warmup → Stable → Decay (WSD) schedule.

    From MiniCPM (2024). Constant LR for most of training, then rapid
    linear decay at the end. Wastes less compute than cosine's long tail.
    Good when you want most of training at peak LR and a clean finish.

    Three phases:
      1. Warmup: linear 0 → peak_lr over warmup_steps
      2. Stable: constant peak_lr (majority of training)
      3. Decay: linear peak_lr → min_lr over final decay_fraction of steps

    Returns LR as a callable — pass directly as optimizer learning_rate.
    """

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        decay_fraction: float = 0.1,
        min_lr: float = 0.0,
    ):
        """
        Args:
            peak_lr: Maximum learning rate (used during stable phase)
            total_steps: Total training steps
            warmup_steps: Steps for linear warmup (phase 1)
            decay_fraction: Fraction of total_steps for final decay (phase 3).
                0.1 means last 10% of training is decay. Default: 0.1
            min_lr: Minimum LR at end of decay
        """
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.stable_end = total_steps - self.decay_steps
        self.min_lr = min_lr
        self._step = 0

    def __call__(self, step=None):
        if step is None:
            step = self._step
            self._step += 1

        if step < self.warmup_steps:
            # Phase 1: warmup
            return self.peak_lr * step / max(self.warmup_steps, 1)
        elif step < self.stable_end:
            # Phase 2: stable
            return self.peak_lr
        else:
            # Phase 3: linear decay
            progress = (step - self.stable_end) / max(self.decay_steps, 1)
            progress = min(progress, 1.0)
            return self.peak_lr + (self.min_lr - self.peak_lr) * progress


class ReduceOnPlateau:
    """Reduce LR when a metric stops improving.

    Tracks a moving average of the metric. When the average hasn't improved
    by at least `threshold` for `patience` steps, multiplies LR by `factor`.

    Unlike CosineDecayWithWarmup, this is NOT a callable scheduler — it's
    stateful and must be stepped explicitly with each loss value. Update
    the optimizer's learning_rate after each step.
    """

    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.5,
        patience: int = 200,
        threshold: float = 0.01,
        min_lr: float = 1e-7,
        window: int = 50,
        cooldown: int = 50,
    ):
        """
        Args:
            initial_lr: Starting learning rate
            factor: Multiply LR by this when reducing (0 < factor < 1)
            patience: Steps without improvement before reducing
            threshold: Minimum relative improvement to count as progress
            min_lr: Floor — won't reduce below this
            window: Moving average window for smoothing the metric
            cooldown: Steps to wait after a reduction before checking again
        """
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.window = window
        self.cooldown = cooldown

        self._history: list[float] = []
        self._best_avg: float = float("inf")
        self._steps_since_improvement: int = 0
        self._cooldown_remaining: int = 0
        self._reductions: int = 0

    def step(self, metric: float) -> bool:
        """Record a metric value and potentially reduce LR.

        Args:
            metric: Loss value (lower is better)

        Returns:
            True if LR was reduced this step
        """
        self._history.append(metric)

        # Not enough history yet
        if len(self._history) < self.window:
            return False

        # Compute moving average
        recent = self._history[-self.window :]
        avg = sum(recent) / len(recent)

        # Cooldown after reduction
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        # Check for improvement
        if self._best_avg == float("inf"):
            # First valid average is always an improvement
            self._best_avg = avg
            self._steps_since_improvement = 0
            return False

        relative_improvement = (self._best_avg - avg) / abs(self._best_avg) if self._best_avg != 0 else 0

        if relative_improvement > self.threshold:
            self._best_avg = avg
            self._steps_since_improvement = 0
            return False

        self._steps_since_improvement += 1

        if self._steps_since_improvement >= self.patience:
            # Reduce LR
            new_lr = max(self.lr * self.factor, self.min_lr)
            if new_lr < self.lr:
                self.lr = new_lr
                self._reductions += 1
                self._steps_since_improvement = 0
                self._cooldown_remaining = self.cooldown
                self._best_avg = avg  # Reset baseline to current level
                return True

        return False

    @property
    def num_reductions(self) -> int:
        return self._reductions
