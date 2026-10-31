"""Registered estimator adapter families.

``src.core.estimator`` holds the generic ``EstimatorAdapter`` protocol and
the ``EstimatorRegistry``; each submodule here wraps one estimator family.
Exactly one family — ``xgboost`` — is registered, which is all the pipeline
produces; adding a family is one new submodule plus one registry entry.
"""

from src.core.estimators.xgboost import ESTIMATOR_ID, XGBoostAdapter, load

__all__ = ["ESTIMATOR_ID", "XGBoostAdapter", "load"]
