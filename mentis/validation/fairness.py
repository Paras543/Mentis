"""
Bias / Fairness detection for Mentis, built on top of Fairlearn.

Computes group fairness metrics (demographic parity, equal
opportunity, equalized odds, selection rate) across one or more
sensitive features, returning plain dicts so callers don't need to
depend on Fairlearn's internal objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mentis.exceptions import ValidationError
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FairnessResult:
    """
    Result of a bias/fairness audit for a single sensitive feature.

    Attributes:
        sensitive_feature: Name of the sensitive feature audited.
        demographic_parity_difference: Max difference in selection
            rate across groups.
        equal_opportunity_difference: Max difference in true positive
            rate across groups.
        equalized_odds_difference: Max difference across both true
            positive and false positive rates across groups.
        selection_rate_by_group: Dict mapping group value -> selection
            rate.
    """

    sensitive_feature: str
    demographic_parity_difference: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    selection_rate_by_group: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitive_feature": self.sensitive_feature,
            "demographic_parity_difference": self.demographic_parity_difference,
            "equal_opportunity_difference": self.equal_opportunity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "selection_rate_by_group": self.selection_rate_by_group,
        }

    def __repr__(self) -> str:
        return (
            f"<FairnessResult feature={self.sensitive_feature!r} "
            f"dp_diff={self.demographic_parity_difference:.3f} "
            f"eo_diff={self.equal_opportunity_difference:.3f}>"
        )


class BiasDetector:
    """
    Computes group fairness metrics for a classifier's predictions
    across sensitive features, using Fairlearn.

    Examples:
        >>> detector = BiasDetector()
        >>> results = detector.detect(y_true, y_pred, sensitive_features={"gender": gender_col})  # doctest: +SKIP
    """

    def detect(
        self,
        y_true: Any,
        y_pred: Any,
        sensitive_features: dict[str, Any],
    ) -> list[FairnessResult]:
        """
        Compute fairness metrics for each provided sensitive feature.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Dict mapping feature name -> array-like
                of group values (e.g. {"gender": df["gender"]}). Each
                entry is audited independently.

        Returns:
            List of `FairnessResult`, one per sensitive feature.

        Raises:
            ValidationError: If `sensitive_features` is empty, lengths
                mismatch, or fairlearn is not installed.

        Examples:
            >>> detector = BiasDetector()
            >>> results = detector.detect(y_true, y_pred, {"gender": gender_col})  # doctest: +SKIP
            >>> results[0].demographic_parity_difference  # doctest: +SKIP
        """
        if not sensitive_features:
            raise ValidationError("At least one sensitive feature must be provided.")

        try:
            from fairlearn.metrics import (
                MetricFrame,
                demographic_parity_difference,
                equalized_odds_difference,
                selection_rate,
                true_positive_rate,
            )
        except ImportError as exc:
            raise ValidationError(
                "fairlearn is not installed. Install it with `pip install fairlearn` "
                "to use bias detection."
            ) from exc

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        results: list[FairnessResult] = []

        for feature_name, group_values in sensitive_features.items():
            group_arr = np.asarray(group_values)

            if len(group_arr) != len(y_true_arr):
                raise ValidationError(
                    f"Sensitive feature '{feature_name}' length ({len(group_arr)}) "
                    f"does not match y_true length ({len(y_true_arr)})."
                )

            try:
                dp_diff = float(
                    demographic_parity_difference(y_true_arr, y_pred_arr, sensitive_features=group_arr)
                )
                eo_diff = float(
                    equalized_odds_difference(y_true_arr, y_pred_arr, sensitive_features=group_arr)
                )

                tpr_frame = MetricFrame(
                    metrics=true_positive_rate,
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    sensitive_features=group_arr,
                )
                tpr_by_group = tpr_frame.by_group
                equal_opp_diff = float(tpr_by_group.max() - tpr_by_group.min())

                selection_frame = MetricFrame(
                    metrics=selection_rate,
                    y_true=y_true_arr,
                    y_pred=y_pred_arr,
                    sensitive_features=group_arr,
                )
                selection_by_group = {
                    str(k): float(v) for k, v in selection_frame.by_group.items()
                }

                results.append(
                    FairnessResult(
                        sensitive_feature=feature_name,
                        demographic_parity_difference=dp_diff,
                        equal_opportunity_difference=equal_opp_diff,
                        equalized_odds_difference=eo_diff,
                        selection_rate_by_group=selection_by_group,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate failures per-feature
                logger.warning(f"Fairness computation failed for '{feature_name}': {exc}")

        if not results:
            raise ValidationError("Fairness computation failed for all sensitive features.")

        return results
    

    