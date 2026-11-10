import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relay import RandomFrameDropper

def test_non_consecutive_drops():
    """Checks the frame dropper such that (1) only second frame onwards and 
    non-consecutive frames can be/are dropped, (2) repeatable given the same 
    random seed, and (3) the frame counting is correct."""
    dropper = RandomFrameDropper(probability=0.2, seed=42)

    decisions = [dropper.should_drop() for _ in range(1000)]

    # The first frame must always be retained
    assert decisions[0] is False

    # Verbose printing if consecutive frames dropped
    for i in range(len(decisions) - 1):
        if decisions[i] and decisions[i+1]:
            print(f"Frame {i} and next frame dropped: surrounding frames: {decisions[i-1]}, {decisions[i]}, {decisions[i+1]}")

    # No two consecutive frames may be dropped.
    assert not any(
            decisions[i] and decisions[i+1]
            for i in range(len(decisions) - 1)
            )

    # Confirm the accounting is correct
    assert dropper.dropped_frames == sum(decisions)

    # Repeating the experiment with the same seed must yield the same seq
    repeat = RandomFrameDropper(probability=0.2, seed=42)
    assert decisions == [
            repeat.should_drop() for _ in range(1000)
            ]

    dropper.report()
    print("All tests passed!")


if __name__ == "__main__":
    test_non_consecutive_drops()
