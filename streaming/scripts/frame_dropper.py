import random

class RandomFrameDropper:
    """Simulate isolated frame losses in a decoded video stream. Drops non-consecutive frames."""

    def __init__(self, probability=0.2, seed=42):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Drop probability must be between 0 and 1.")

        self.probability = probability
        self.rng = random.Random(seed)

        self.previous_dropped = False
        self.total_frames = 0
        self.dropped_frames = 0
        self.dropped_indices = []

    def should_drop(self):
        """Return True when the current frame should be dropped."""
        frame_index = self.total_frames
        self.total_frames += 1

        # Never drop the first frame
        if frame_index == 0:
            self.previous_dropped = False
            return False

        # Never drop two consecutive frames
        if self.previous_dropped:
            self.previous_dropped = False
            return False

        drop = self.rng.random() < self.probability
       
        self.previous_dropped = drop

        if drop:
            self.dropped_frames += 1
            self.dropped_indices.append(frame_index)

        return drop

    def report(self):
        """Prints out information about total frames and frames dropped."""
        actual_rate = (self.dropped_frames / self.total_frames) if self.total_frames else 0.0

        # Print out the statistics
        print("\n--- Frame Loss Statistics ---")
        print(f"Total frames: {self.total_frames}")
        print(f"Dropped frames: {self.dropped_frames}")
        print(f"Actual loss rate: {actual_rate:.2%}")
        print(f"Dropped indices: {self.dropped_indices}")
