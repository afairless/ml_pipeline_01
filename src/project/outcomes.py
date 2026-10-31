"""Project outcome registry, comparison vocabulary, and transform resolver.

The registry is the single source of truth for which outcomes the pipeline
runs, in which order, and under which transform ids. ``OUTCOMES`` maps each
outcome name to one frozen :class:`Outcome`; because the mapping is read-only
and its insertion order is preserved, iterating it reproduces the pipeline's
run order and a mutation attempt fails instead of silently changing the
vocabulary.

An :class:`Outcome` name **is** the outcome column name in the outcome table
and the ``{outcome}`` placeholder value used by per-outcome stage paths. The
four demo outcomes are ``outcome_1`` through ``outcome_4``; ``outcome_1`` and
``outcome_3`` are modeled, while ``outcome_2`` and ``outcome_4`` exist only as
analysis counterparts. Every demo outcome uses the ``identity`` transform on
both the model and reporting scales, so negative demo values remain valid.

:class:`OutcomeComparison` names the two analysis pairs the outcome-analysis
stage differences, always as ``left - right``. :data:`OUTCOME_COMPARISONS`
keeps them in report and plot order.

:func:`resolve_transform` is the only transform resolver. Exactly two
forward/inverse pairs are registered:

``identity`` / ``identity``
    The no-op pair used by every demo outcome and accepted for any numeric
    value.
``log1p`` / ``expm1``
    The non-trivial pair that exercises transform resolution. Its forward
    direction requires a value greater than ``-1`` and raises a boundary
    ``ValueError`` outside that domain. The demo does not use this pair
    because ``outcome_3`` may be negative; a future project that enables a
    non-identity transform must register a matching inverse before use.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

Transform = Callable[[float], float]
"""A single numeric transform from one scale to another."""


@dataclass(frozen=True, slots=True)
class TransformPair:
    """A registered forward transform and the inverse that reports on the
    original scale.

    Attributes:
        forward: The model-space transform.
        inverse: The reporting transform that maps a model-space value back
            to the original scale.
    """

    forward: Transform
    inverse: Transform


def _identity(value: float) -> float:
    """Return ``value`` unchanged."""
    return value


def _log1p(value: float) -> float:
    """Return ``log(1 + value)``, rejecting values at or below ``-1``.

    The domain check is written as ``not value > -1.0`` so a NaN input is
    rejected by the same boundary rule rather than producing a silent NaN.
    """
    if not value > -1.0:
        raise ValueError(f"log1p transform domain requires value > -1; got {value!r}")
    return math.log1p(value)


def _expm1(value: float) -> float:
    """Return ``exp(value) - 1``, the inverse of the ``log1p`` transform."""
    return math.expm1(value)


_REGISTERED_TRANSFORMS: Mapping[str, TransformPair] = MappingProxyType(
    {
        "identity": TransformPair(forward=_identity, inverse=_identity),
        "log1p": TransformPair(forward=_log1p, inverse=_expm1),
    }
)


def resolve_transform(transform_id: str) -> TransformPair:
    """Resolve a registered transform id to its forward/inverse pair.

    Args:
        transform_id: A registered id: ``identity`` or ``log1p``.

    Returns:
        The frozen :class:`TransformPair` registered under ``transform_id``.

    Raises:
        ValueError: ``transform_id`` is not registered.
    """
    try:
        return _REGISTERED_TRANSFORMS[transform_id]
    except KeyError:
        raise ValueError(
            f"unknown transform id {transform_id!r}; registered ids are "
            f"{sorted(_REGISTERED_TRANSFORMS)}"
        ) from None


@dataclass(frozen=True, slots=True)
class Outcome:
    """One pipeline outcome and the transform ids it carries.

    Attributes:
        name: The outcome name. It is also the outcome column name in the
            outcome table and the ``{outcome}`` placeholder value.
        outcome_transform: Registered id of the model-space transform.
        reporting_backtransform: Registered id of the reporting transform
            that maps predictions back to the original scale.
        modeled: Whether the train/predict/evaluate stages run for this
            outcome.

    Raises:
        ValueError: ``name`` is empty, or either transform id is not
            registered. Validating on construction keeps a typo from
            registering an outcome that no stage can resolve.
    """

    name: str
    outcome_transform: str = "identity"
    reporting_backtransform: str = "identity"
    modeled: bool = False

    def __post_init__(self) -> None:
        """Validate the frozen fields on construction."""
        if not self.name:
            raise ValueError(
                "outcome name must be non-empty; an Outcome.name is the "
                "outcome-table column name and the {outcome} placeholder"
            )
        resolve_transform(self.outcome_transform)
        resolve_transform(self.reporting_backtransform)


# The registry: mapping key == Outcome.name, insertion order == pipeline run
# order. MappingProxyType makes the registry read-only, so a consumer that
# tries to mutate it gets a TypeError instead of silently changing the
# vocabulary every stage and layout resolves against.
OUTCOMES: Mapping[str, Outcome] = MappingProxyType(
    {
        "outcome_1": Outcome(name="outcome_1", modeled=True),
        "outcome_2": Outcome(name="outcome_2", modeled=False),
        "outcome_3": Outcome(name="outcome_3", modeled=True),
        "outcome_4": Outcome(name="outcome_4", modeled=False),
    }
)


def modeled_outcomes() -> tuple[Outcome, ...]:
    """Return the modeled outcomes in registry (run) order.

    The returned values are the frozen registry entries themselves, so
    callers can rely on identity with ``OUTCOMES``.
    """
    return tuple(outcome for outcome in OUTCOMES.values() if outcome.modeled)


def outcome_names() -> tuple[str, ...]:
    """Return every outcome name in registry (run) order.

    The tuple is directly iterable by the executor and matches the keys of
    ``OUTCOMES``. Names are unique by construction because the registry is a
    mapping.
    """
    return tuple(OUTCOMES)


@dataclass(frozen=True, slots=True)
class OutcomeComparison:
    """One left-minus-right outcome comparison of the analysis stage.

    Attributes:
        left: Registered outcome name on the left-hand side of the
            difference.
        right: Registered outcome name on the right-hand side of the
            difference. The comparison direction is always ``left - right``.

    Raises:
        ValueError: Either name is not registered, or both names are the
            same. A self-difference is always zero and would produce a
            meaningless analysis artifact.
    """

    left: str
    right: str

    def __post_init__(self) -> None:
        """Validate the frozen fields on construction."""
        for name in (self.left, self.right):
            if name not in OUTCOMES:
                raise ValueError(
                    f"comparison references unknown outcome {name!r}; "
                    f"registered outcomes are {sorted(OUTCOMES)}"
                )
        if self.left == self.right:
            raise ValueError(
                f"comparison compares {self.left!r} with itself; a "
                "difference requires two distinct outcomes"
            )


# The analysis stage's comparisons, in report and plot order. The direction is
# always left - right; no sign rule beyond that is encoded.
OUTCOME_COMPARISONS: tuple[OutcomeComparison, ...] = (
    OutcomeComparison(left="outcome_1", right="outcome_2"),
    OutcomeComparison(left="outcome_3", right="outcome_4"),
)


def _validate_comparisons(comparisons: tuple[OutcomeComparison, ...]) -> None:
    """Reject a repeated ``(left, right)`` pair.

    Called once at import with ``OUTCOME_COMPARISONS``. Per-instance
    validation cannot catch a duplicate pair, and a duplicate would write the
    same comparison twice under two report entries. The parameter keeps the
    check callable on a hand-built tuple in tests.

    Args:
        comparisons: The comparison vocabulary to validate, in report order.

    Raises:
        ValueError: Two comparisons name the same ordered pair.
    """
    seen: set[tuple[str, str]] = set()
    for comparison in comparisons:
        pair = (comparison.left, comparison.right)
        if pair in seen:
            raise ValueError(
                f"comparison pair {pair!r} is compared more than once; each "
                "left/right pair is one comparison"
            )
        seen.add(pair)


_validate_comparisons(OUTCOME_COMPARISONS)
