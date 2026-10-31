"""Canonical column vocabulary shared by the project layer and its stages.

Every frame that crosses a stage boundary names its identity, feature, and
reporting columns through these constants instead of repeating string
literals. Keeping the names in one module means a rename is a single edit and
the boundary schemas, joins, and manifest writers cannot drift apart.

``KEY_COLUMNS`` is the identity contract: its order is canonical and must
never be reversed, because stages join, sort, and assert alignment on the
``(case_id, case_date)`` pair in exactly that order. The remaining constants
name columns the project layer derives from a key set — a generic feature
column, the estimator identifier attached to extracted importances, and the
importance value itself.
"""

CASE_ID = "case_id"
"""Case identity column: one row per predicted unit."""

CASE_DATE = "case_date"
"""Case date column: the split key every partition predicate uses."""

KEY_COLUMNS = (CASE_ID, CASE_DATE)
"""Canonical identity columns, in join/sort order. Never reverse this tuple."""

FEATURE_COLNAME = "feature"
"""Generic feature column name used where a single feature is reported."""

FEATURE_ESTIMATOR_ID_COLNAME = "feature_estimator_id"
"""Estimator identifier carried alongside extracted feature importances."""

FEATURE_IMPORTANCE_COLNAME = "importance"
"""Feature importance value column produced by the training stage."""
