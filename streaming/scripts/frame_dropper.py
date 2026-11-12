import random


class RandomFrameDropper:
	"""Simulate frame losses in a decoded video stream: bursts of 1 to max_run consecutive frames.

	At each kept frame, a burst starts with the given probability; its length is drawn uniformly from 1..max_run.
	The frame after a burst is always kept, so two bursts never touch and the relay always has a frame on each side of
	a gap to interpolate between. With max_run=1 (the default) the drops are isolated frames, and the same seed gives
	the same pattern as the first version of this class.

	Note: the drop rate is not the probability. It is the probability times the mean burst length, divided by
	(1 + probability x mean burst length) -- e.g. probability 0.3 and max_run 3 drops about 38% of the frames.
	"""

	def __init__(self, probability=0.2, seed=42, max_run=1):
		if not 0.0 <= probability <= 1.0:
			raise ValueError("Drop probability must be between 0 and 1.")
		if max_run < 1:
			raise ValueError("max_run must be at least 1.")

		self.probability = probability
		self.max_run = max_run
		self.rng = random.Random(seed)

		self.run_left = 0  # frames still to drop in the current burst
		self.previous_dropped = False
		self.total_frames = 0
		self.dropped_frames = 0
		self.dropped_indices = []
		self.run_lengths = []  # one entry per burst

	def _drop(self, frame_index):
		self.previous_dropped = True
		self.dropped_frames += 1
		self.dropped_indices.append(frame_index)
		return True

	def should_drop(self):
		"""Return True when the current frame should be dropped."""
		frame_index = self.total_frames
		self.total_frames += 1

		# Never drop the first frame
		if frame_index == 0:
			return False

		# Carry on with the current burst
		if self.run_left > 0:
			self.run_left -= 1
			return self._drop(frame_index)

		# Always keep the frame after a burst
		if self.previous_dropped:
			self.previous_dropped = False
			return False

		if self.rng.random() < self.probability:
			length = self.rng.randint(1, self.max_run) if self.max_run > 1 else 1
			self.run_lengths.append(length)
			self.run_left = length - 1
			return self._drop(frame_index)
		return False

	def report(self):
		"""Prints out information about total frames and frames dropped."""
		actual_rate = (self.dropped_frames / self.total_frames) if self.total_frames else 0.0

		# Print out the statistics
		print("\n--- Frame Loss Statistics ---")
		print(f"Total frames: {self.total_frames}")
		print(f"Dropped frames: {self.dropped_frames} in {len(self.run_lengths)} bursts")
		print(f"Actual loss rate: {actual_rate:.2%}")
		print("Burst lengths: " + ", ".join(
			f"{n}x{sum(1 for r in self.run_lengths if r == n)}" for n in range(1, self.max_run + 1)))


def gap_fill_plan(interval):
	"""How to fill the gap between two kept frames that are `interval` frames apart (interval - 1 frames missing).

	Returns [(position, left, right), ...] with positions 0 (the earlier frame) to interval (the later one): each entry
	is a frame to make as the midpoint of the frames at `left` and `right`, listed so that both already exist when it is
	made. Repeated halving is exact when the interval is a power of two (2: 1 frame; 4: 3 frames, as in the 6 -> 24 fps
	mode; 8: 7 frames). Otherwise the odd halves put a frame up to half a frame period off its true place.
	"""
	plan = []

	def split(low, high):
		if high - low < 2:
			return
		mid = (low + high) // 2
		plan.append((mid, low, high))
		split(low, mid)
		split(mid, high)

	split(0, interval)
	return plan
