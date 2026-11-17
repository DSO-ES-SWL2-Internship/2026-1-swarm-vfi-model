# Pipeline budget

A video relay is a chain of steps: decode, scale, run the model, encode. A 24 fps video gives each frame **41.7 ms**.
This simulator shows when a chain fits in that budget, and what moving a step to faster hardware does (and does not do).

<iframe src="../sims/pipeline-budget.html" title="Pipeline budget simulator"
        style="width:100%;height:1020px;border:1px solid var(--md-default-fg-color--lightest);border-radius:8px"
        loading="lazy"></iframe>

[Open it on its own page](sims/pipeline-budget.html){ target="_blank" }

## What to try

1. **Start with "All CPU (Demo 2)".** The model step is 51 ms per frame, over the budget. It falls behind, and the video
   lags. This is what you saw in Demo 2.
2. **Click "With hardware (Demo 3)".** Everything is under 10 ms, and it keeps up with a large margin.
3. **Look at the table at the bottom.** The measured time per frame (52.0 ms all-CPU, 8.9 ms with hardware) is almost
   exactly the **slowest step** (51.3 and 8.5 ms), and far from the **sum** of the steps (64.2 and 16.4 ms). The relay
   runs its steps in parallel, so only the slowest one counts.
4. **Turn the queues off.** Now the steps add up. A queue gives each step its own thread, which is what lets them
   overlap. With the hardware preset, the total goes from 8.5 to 16.4 ms.
5. **Make a fast step faster.** With the hardware preset, set *Encode* to 0. Nothing changes: it was never the slowest
   step. That is the trap of optimising the wrong step.
6. **Find the bottleneck.** With the hardware preset, the slowest step is *Scale + convert* at 8.5 ms, not the model.
   Make the model slower and see how far it can grow before it becomes the bottleneck.
7. **Click "Model on the GPU instead".** The GPU took 9.5 s for one model call. Even though only 37% of the frames need
   it, the model step is thousands of milliseconds per frame, and the whole relay is hopeless.

## Where the numbers come from

The two measured presets are the relay's own statistics from 16 November 2026: the average time per input frame in each
step, over the 1251-frame trailer, run flat out on the board (`relay_stats` in the relay's output). The decode step is not
timed separately by the relay, so it is left at 0 and is not part of those numbers. The model step is the average over
**all** input frames, including the 63% that are not lost and need no model call.

The idea is the one in [The GStreamer pipeline](../gstreamer-pipeline.md#threads-and-queues).
