"""Project-agnostic uniform-binning helper shared across pipeline stages.

This is a pure NumPy utility with no filesystem behavior. It lives beside the
serialization helpers rather than inside them so ``src.core.io`` stays limited
to writing artifacts.
"""

from __future__ import annotations

import numpy as np


def create_uniform_bin_vector_of_given_length(
    bin_n: int, vector_length: int
) -> np.ndarray:
    """Return per-position bin indices with ``bin_n`` near-uniform bins.

    The returned vector has length ``vector_length`` and contains the bin
    index (``0`` to ``bin_n - 1``) for each position. The cut points are
    rounded positions along the vector, so every bin holds the same number of
    elements up to a difference of one when the split is uneven.

    Args:
        bin_n: Number of bins; must be positive.
        vector_length: Number of positions to assign; must be positive.

    Returns:
        One-dimensional integer array of length ``vector_length``.

    Raises:
        AssertionError: If either argument is not positive.
    """

    assert bin_n > 0, "bin_n must be greater than 0"
    assert vector_length > 0, "vector_length must be greater than 0"

    cuts = np.linspace(0, vector_length, bin_n + 1).round().astype(int)
    segments = [np.full(cuts[i + 1] - cuts[i], i) for i in range(len(cuts) - 1)]
    return np.concatenate(segments)
