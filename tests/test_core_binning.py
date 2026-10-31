"""Unit tests for the uniform-bin helper in ``src.core.binning``.

Pins the invariant that ``create_uniform_bin_vector_of_given_length`` returns
an ``np.ndarray`` of the requested length split as evenly as possible into the
requested number of bins, and raises for a non-positive argument. A failure
here means a consumer would receive a bin vector of the wrong length or shape
and mis-bucket every observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.binning import create_uniform_bin_vector_of_given_length


def test_bin_vector_two_bins_splits_evenly() -> None:
    vector = create_uniform_bin_vector_of_given_length(2, 10)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (10,)
    assert vector.dtype.kind == "i"
    assert list(vector) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_bin_vector_uneven_length_keeps_bins_approximately_uniform() -> None:
    vector = create_uniform_bin_vector_of_given_length(3, 10)
    assert vector.shape == (10,)
    bincount = np.bincount(vector)
    assert bincount.sum() == 10
    assert bincount.max() - bincount.min() <= 1
    assert set(vector.tolist()) == {0, 1, 2}


def test_bin_vector_single_bin() -> None:
    vector = create_uniform_bin_vector_of_given_length(1, 5)
    assert (vector == 0).all()


def test_bin_vector_rejects_nonpositive_arguments() -> None:
    with pytest.raises(AssertionError, match="bin_n"):
        create_uniform_bin_vector_of_given_length(0, 10)
    with pytest.raises(AssertionError, match="vector_length"):
        create_uniform_bin_vector_of_given_length(2, 0)
