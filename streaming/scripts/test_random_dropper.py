import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_dropper import RandomFrameDropper, gap_fill_plan


def runs(decisions):
    """Lengths of the bursts of consecutive drops."""
    lengths, current = [], 0
    for dropped in decisions:
        if dropped:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def test_isolated_drops():
    """max_run=1: (1) the first frame is kept, (2) no two consecutive frames are dropped, (3) the same seed gives
    the same pattern, (4) the accounting is right."""
    dropper = RandomFrameDropper(probability=0.2, seed=42)
    decisions = [dropper.should_drop() for _ in range(1000)]

    assert decisions[0] is False
    assert not any(a and b for a, b in zip(decisions, decisions[1:]))
    assert dropper.dropped_frames == sum(decisions)
    repeat = RandomFrameDropper(probability=0.2, seed=42)
    assert decisions == [repeat.should_drop() for _ in range(1000)]


def test_bursts():
    """max_run=3: bursts of 1 to 3, all lengths occur, a kept frame after each burst, first frame kept."""
    dropper = RandomFrameDropper(probability=0.3, seed=7, max_run=3)
    decisions = [dropper.should_drop() for _ in range(5000)]

    assert decisions[0] is False
    lengths = runs(decisions)
    assert max(lengths) == 3 and min(lengths) == 1
    assert set(lengths) == {1, 2, 3}
    assert lengths[:-1] == dropper.run_lengths[:len(lengths) - 1]  # the last burst can run past the end
    assert dropper.dropped_frames == sum(decisions)
    # about p*L / (1 + p*L) of the frames, with L = 2 the mean burst length: 37.5%
    assert abs(dropper.dropped_frames / 5000 - 0.375) < 0.03
    dropper.report()


def test_gap_fill_plan():
    """Every plan makes interval - 1 frames, each from two frames that already exist; powers of two split exactly."""
    for interval in range(2, 10):
        plan = gap_fill_plan(interval)
        assert sorted(position for position, _, _ in plan) == list(range(1, interval))
        have = {0, interval}
        for position, left, right in plan:
            assert left in have and right in have
            have.add(position)
    assert gap_fill_plan(2) == [(1, 0, 2)]
    assert gap_fill_plan(4) == [(2, 0, 4), (1, 0, 2), (3, 2, 4)]
    for interval in (2, 4, 8):  # exact midpoints
        assert all(position * 2 == left + right for position, left, right in gap_fill_plan(interval))


if __name__ == "__main__":
    test_isolated_drops()
    test_bursts()
    test_gap_fill_plan()
    print("All tests passed!")
