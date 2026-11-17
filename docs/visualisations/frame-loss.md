# Frame loss and gap filling

This simulator shows the two rules at the heart of the demo: how bursts of frames are **lost**
(`RandomFrameDropper`) and how the model **fills the gap** (`gap_fill_plan`). Both are in
`streaming/scripts/frame_dropper.py`.

<iframe src="../sims/frame-loss.html" title="Frame loss and gap filling simulator"
        style="width:100%;height:1130px;border:1px solid var(--md-default-fg-color--lightest);border-radius:8px"
        loading="lazy"></iframe>

[Open it on its own page](sims/frame-loss.html){ target="_blank" }

!!! note "Same rules, different random numbers"
    The simulator follows the same rules as the relay, but uses its own random-number generator, so seed 42 here is not
    the same pattern as seed 42 on the board. What matches is how the system behaves.

## What to try

1. **Start from the defaults** (probability 0.3, longest burst 3). Count the lost frames in the strip, then look at the
   loss rate over 20,000 frames. It is close to 38%, not 30%. Why? The probability is only the chance that a burst
   *starts*. Each burst then removes 1 to 3 frames (2 on average), and the frame after a burst is always kept.
2. **Push the probability to 1.** Not every frame is lost. After each burst one frame is always kept, so the lost
   frames can never touch, and there is always a real frame on each side of a gap to interpolate between. That is why
   the rule exists.
3. **Set the longest burst to 1.** Every gap is one frame. This is the easiest case for the model.
4. **Click a burst** in the strip. The gap panel below shows exactly how that burst is filled.
5. **Compare gaps of 1, 2 and 3 missing frames.** One missing frame is exact: one midpoint of its two neighbours. Two
   missing frames is **not**: the first frame is made from frames 0 and 3, whose midpoint is at 1.5, so it lands half
   a frame period off, and the second is made from that one and frame 3, landing a quarter off. Three missing frames is
   exact again.
6. **Find the pattern.** Which gaps are exact? Those where the distance between the two kept frames (missing + 1) is a
   power of two: 2, 4, 8. That is why the demo's longest burst is 3: a gap of 4 is the longest exact case that is also
   plausible.
7. **Count the model calls.** A gap of *n* missing frames always costs exactly *n* calls, never more. Halving does not
   waste work.

## Why this matters for the hardware

Each lost frame is one model call, and at 37% loss that is about 465 calls for the 1251-frame trailer. On the NPU a call
takes 12.1 ms. On the CPU it takes 133 ms. The [pipeline budget](pipeline-budget.md) shows what that does to the
whole relay.
