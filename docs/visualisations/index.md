# Visualisations

Small simulators that run in your browser. Each one takes an idea from this project, lets you change the settings, and
shows what happens. Each has a short guide with suggestions for what to try.

| Simulator | What you can explore |
| --- | --- |
| [Frame loss and gap filling](frame-loss.md) | Set how often bursts of frames are lost, and how long they can be. See which frames are lost, why the loss rate is not the probability, and how the model fills a gap by repeated halving, including why a gap of 3 is not exact |
| [Pipeline budget](pipeline-budget.md) | Set the time of each step of the relay. See why the slowest step sets the pace when the steps overlap, why a faster step elsewhere changes nothing, and how the measured CPU-only and hardware runs compare with the predictions |

They have no dependencies and nothing to install. Each is a single HTML file in `docs/visualisations/sims/`, so you can
open it straight from the repository, or copy it anywhere.
