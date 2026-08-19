"""The divergence guard: what stops a blown-up run instead of paying for it.

The run this exists for: a large model on 16 GPUs at lr=2e-4. It
diverged between step 15000 and 20000 and then trained for **eight more hours**
producing garbage. Nothing stopped it and nothing said anything. Measured from
its own checkpoints, with a forward pass on real validation data::

    step     loss      output std (target ~0.69)
     5000    5.13      0.700
    10000   24.70      0.769     <- a spike, and it RECOVERED
    15000    1.96      0.729
    20000   1.05e8     652       <- gone

`test_the_measured_large_run_shape_does_not_trip_the_guard` is the test that
matters: the step-10000 spike is a 5x excursion that a run survived, and any
guard that fires on it is worse than no guard at all. The same test then feeds
the step-20000 state to the *same* guard instance and requires the abort, so it
cannot pass by the guard being asleep.

Every test here was run against deliberately broken implementations; the
`MUTANT:` line on each says which one it catches.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oceanarches.lightning_modules.ocean_forecast import (
    DIVERGENCE_FACTOR,
    DIVERGENCE_PATIENCE,
    DIVERGENCE_WARMUP_STEPS,
    DivergenceGuard,
    divergence_refusal,
)

#: The four measured (step, loss) anchors quoted above.
MEASURED = ((5000, 5.13), (10000, 24.70), (15000, 1.96), (20000, 1.05e8))
#: Where the run started, before the warm-up had brought it down.
INITIAL_LOSS = 935.0


def feed(guard: DivergenceGuard, losses, first_step: int = 0):
    """Push a loss series through `guard`; return `(step, reason)` or None."""
    for offset, value in enumerate(losses):
        step = first_step + offset
        reason = guard.update(step, value)
        if reason is not None:
            return step, reason
    return None


def large_run_losses(seed: int = 0, spike_width: int = 400) -> list[float]:
    """A per-step loss series with the measured shape, steps 0..15000.

    Log-linear between the anchors -- 935 at step 0, 5.13 at 5000, 1.96 at
    15000 -- with a sharp multiplicative spike centred on step 10000 that
    reaches exactly the measured 24.70 and then recovers, and lognormal
    batch-to-batch noise on top so that no single step is the clean anchor
    value. `spike_width` is deliberately wider than the guard's patience: a
    spike is not survived here by being shorter than the counter.
    """
    rng = np.random.default_rng(seed)
    baseline_knots = [(0, INITIAL_LOSS), (5000, 5.13), (15000, 1.96)]
    steps = np.arange(0, 15001)
    baseline = np.exp(
        np.interp(
            steps,
            [step for step, _ in baseline_knots],
            [math.log(value) for _, value in baseline_knots],
        )
    )
    peak = 24.70 / float(baseline[10000])
    half = spike_width // 2
    distance = np.abs(steps - 10000)
    spike = np.where(distance <= half, math.log(peak) * (1 - distance / half), 0.0)
    noise = rng.normal(0.0, 0.15, size=steps.shape)
    return list(baseline * np.exp(spike + noise))


# ---------------------------------------------------------------------------
# 1. It fires when the run really has gone
# ---------------------------------------------------------------------------
def test_the_guard_fires_on_a_sustained_explosion():
    """A healthy run, then the step-20000 loss, sustained: abort.

    And abort on the patience-th consecutive bad step, not before -- a guard
    that fires at once is the one that would have killed the step-10000 spike.

    MUTANT: `if self.consecutive < self.patience: return None` -> `return
    reason` fires on the first bad step (assert on the step index below);
    dropping the comparison entirely never fires at all.
    """
    guard = DivergenceGuard()
    assert feed(guard, [2.0] * 1000) is None, "a flat, healthy loss must not trip it"

    hit = feed(guard, [1.05e8] * (2 * DIVERGENCE_PATIENCE), first_step=1000)
    assert hit is not None, "a loss of 1.05e8 against a reference of ~2 must abort"
    step, reason = hit
    assert step == 1000 + DIVERGENCE_PATIENCE - 1, (
        f"aborted on bad step {step - 1000 + 1}, expected {DIVERGENCE_PATIENCE}"
    )
    assert "consecutive" in reason and "1.05e+08" in reason


def test_an_excursion_shorter_than_the_patience_is_survived():
    """Even a 100x excursion is forgiven while it is brief, then forgotten.

    MUTANT: counting total bad steps instead of consecutive ones (drop the
    `self.consecutive = 0` reset on a good step) makes the second burst abort,
    because the two bursts together exceed the patience.

    MUTANT: `>=` -> `>` in the patience comparison shows up in the first test.
    """
    guard = DivergenceGuard()
    assert feed(guard, [2.0] * 1000) is None
    for burst in range(2):
        first = 1000 + burst * 2000
        assert feed(guard, [1e6] * (DIVERGENCE_PATIENCE - 1), first_step=first) is None
        assert feed(guard, [2.0] * 1000, first_step=first + DIVERGENCE_PATIENCE) is None


# ---------------------------------------------------------------------------
# 2. It does NOT fire on the shape a real run had -- the test that matters
# ---------------------------------------------------------------------------
def test_the_measured_large_run_shape_does_not_trip_the_guard():
    """5.13 -> 24.70 -> 1.96, spike and recovery included, must train on.

    The step-10000 spike is a ~5x excursion above the run's best smoothed loss;
    the step-20000 divergence is ~5e7 above it. Seven orders of magnitude
    separate them, and `DIVERGENCE_FACTOR = 100` sits 20x above the survivable
    one and 500000x below the fatal one. This test pins both ends: the same
    guard instance that let the spike through aborts on the step-20000 state.

    MUTANT: `DIVERGENCE_FACTOR = 5` (a plausible-looking "5x the reference"
    rule) aborts inside the spike.
    """
    guard = DivergenceGuard()
    losses = large_run_losses()
    # The trajectory really does have the measured shape.
    assert losses[0] > 500
    assert 4.0 < np.median(losses[4900:5100]) < 6.5
    assert max(losses[9900:10100]) > 20.0
    assert 1.5 < np.median(losses[14900:15001]) < 2.5

    survived = feed(guard, losses)
    assert survived is None, f"the guard killed a run that recovered: {survived}"
    assert guard.observed > DIVERGENCE_WARMUP_STEPS, "the guard was never even armed"
    assert guard.reference is not None and guard.reference < 3.0

    # ...and the same armed guard does stop the step-20000 state.
    hit = feed(guard, [1.05e8] * (2 * DIVERGENCE_PATIENCE), first_step=15001)
    assert hit is not None, "the guard let the divergence through as well"


def test_a_gradual_exponential_blow_up_is_caught_early():
    """The real divergence was not a jump: 1.96 -> 1.05e8 over 5000 steps.

    That is about 1.0036x per step, and it is why the reference is the MINIMUM
    smoothed loss rather than a tracking one: a 100-step EMA following a blow-up
    that slow sits only ~1.4x behind it, so the loss never exceeds any multiple
    of it and the run explodes with the guard watching. A minimum cannot be
    dragged upwards.

    The abort also has to come early enough to be worth having -- the point is
    the GPU hours after it, not the diagnosis.

    MUTANT: `self.reference = self.mean` unconditionally (an EMA reference
    rather than the running minimum of one) never fires here at all.
    """
    guard = DivergenceGuard()
    assert feed(guard, [1.96] * 1000) is None

    ramp = list(np.exp(np.linspace(math.log(1.96), math.log(1.05e8), 5001)))
    hit = feed(guard, ramp, first_step=1000)
    assert hit is not None, "a 7.7-order-of-magnitude blow-up went unnoticed"
    step, _ = hit
    into_the_ramp = step - 1000
    assert into_the_ramp < 2000, (
        f"aborted {into_the_ramp} steps into a 5000-step blow-up; too late to save the run"
    )
    assert ramp[into_the_ramp] < 1e5, "waited until the loss was astronomically large"


def test_the_first_steps_of_a_run_cannot_abort_it():
    """Early training is exempt, whatever the loss does.

    A run legitimately starts high -- this one began near 935 and reached 5.13
    by step 5000 -- and the first few losses are a bad reference for the rest.
    The minimum-of-an-EMA rule already makes a falling loss safe; the warm-up is
    the belt to that pair of braces, and it is what this pins.

    MUTANT: drop the `self.observed <= self.warmup_steps` clause and the burst
    below aborts a run 20 steps in.
    """
    guard = DivergenceGuard(patience=5, warmup_steps=100)
    assert feed(guard, [1.0] * 20) is None
    assert feed(guard, [1e6] * 50, first_step=20) is None, "aborted during the warm-up"
    assert guard.observed <= guard.warmup_steps

    hit = feed(guard, [1e6] * 200, first_step=70)
    assert hit is not None, "the warm-up never ended"


def test_the_spike_never_even_reaches_the_threshold():
    """The margin, stated as a number rather than left implicit.

    MUTANT: any factor at or below the ratio asserted here fires on the spike.
    """
    guard = DivergenceGuard()
    worst = 0.0
    for step, value in enumerate(large_run_losses()):
        guard.update(step, value)
        if guard.reference and guard.observed > DIVERGENCE_WARMUP_STEPS:
            worst = max(worst, value / guard.reference)
    assert worst < 20.0, f"the healthy run reached {worst:.1f}x its reference"
    assert DIVERGENCE_FACTOR > 5 * worst, (
        f"{DIVERGENCE_FACTOR}x leaves too little margin over a {worst:.1f}x spike"
    )


# ---------------------------------------------------------------------------
# 3. Non-finite: no tolerance at all
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_a_non_finite_loss_aborts_on_the_first_occurrence(value):
    """NaN and inf abort immediately -- during the warm-up, on step 0, always.

    MUTANT: dropping the `math.isfinite` branch makes every case here hang on
    for the patience (inf) or never fire at all (NaN, because every comparison
    with NaN is False).
    """
    assert DivergenceGuard().update(0, value) is not None, "not caught on the first step"

    warm = DivergenceGuard()
    assert feed(warm, [2.0] * 1000) is None
    hit = feed(warm, [value], first_step=1000)
    assert hit == (1000, hit[1])
    assert "NaN" in hit[1] or "inf" in hit[1]


def test_a_non_finite_loss_is_caught_even_with_the_patience_turned_up():
    """The tolerance applies to explosions, never to NaN.

    MUTANT: routing the non-finite case through the consecutive-step counter
    makes this wait 10000 steps.
    """
    guard = DivergenceGuard(patience=10000)
    assert guard.update(7, float("nan")) is not None


# ---------------------------------------------------------------------------
# 4. A healthy run is never touched
# ---------------------------------------------------------------------------
def test_a_normal_decreasing_loss_never_trips_it():
    """935 down to 0.7 over 20000 noisy steps, which is what a good run looks like.

    The early steps are the ones a naive guard fires on: the loss legitimately
    starts three orders of magnitude above where it ends.

    MUTANT: comparing against the FIRST loss seen, or against the running
    *maximum* smoothed loss, keeps this green but breaks the explosion tests;
    comparing the loss against a fixed absolute ceiling (say 100) fires here on
    the first few hundred steps.
    """
    rng = np.random.default_rng(1)
    steps = np.arange(0, 20001)
    curve = INITIAL_LOSS * np.exp(-steps / 2500.0) + 0.7
    losses = curve * np.exp(rng.normal(0.0, 0.2, size=steps.shape))
    assert losses[0] > 500 and 0.5 < losses[-1] < 1.5

    guard = DivergenceGuard()
    assert feed(guard, list(losses)) is None


def test_a_noisy_plateau_with_bad_batches_never_trips_it():
    """A converged run whose individual batches vary by an order of magnitude.

    MUTANT: dropping the patience (fire on the first exceedance) with a low
    factor turns every unlucky batch into a dead job.
    """
    rng = np.random.default_rng(2)
    losses = list(0.9 * np.exp(rng.normal(0.0, 0.4, size=8000)))
    for index in range(500, 8000, 700):  # a genuinely bad batch every so often
        losses[index] *= 10.0
    assert feed(DivergenceGuard(), losses) is None


# ---------------------------------------------------------------------------
# 5. What the abort says
# ---------------------------------------------------------------------------
def test_the_abort_message_names_the_step_the_loss_the_checkpoint_and_the_way_back(tmp_path):
    """All four, because the run that motivated this printed none of them.

    MUTANT: dropping any one of the four interpolations fails a line here.
    """
    checkpoint = tmp_path / "checkpoints" / "checkpoint_global_step=15000.ckpt"
    message = divergence_refusal(
        step=20000,
        loss=1.05e8,
        reason="a reason the guard gave",
        checkpoint=checkpoint,
        lr=2e-4,
    )
    assert "20000" in message
    assert "1.05e+08" in message
    assert str(checkpoint) in message
    assert "a reason the guard gave" in message
    assert "++module.module.lr=0.0001" in message, "no concrete lower learning rate"
    assert "++module.module.divergence_guard=False" in message, "no way to turn it off"


def test_the_message_still_helps_when_nothing_has_been_saved_yet():
    """A run that diverges before its first checkpoint must not print `None`.

    MUTANT: `str(checkpoint)` without the None branch prints "None" and sends
    the reader looking for a file called None.
    """
    message = divergence_refusal(step=300, loss=float("inf"), reason="r", checkpoint=None, lr=None)
    assert "saved nothing yet" in message
    assert "\n      None\n" not in message
    assert "++module.module.lr=" in message
