"""
Comparison subpackage: automated multi-model training, evaluation,
and leaderboard ranking for Mentis.
"""

from mentis.comparison.leaderboard import Leaderboard, ModelResult, build_leaderboard
from mentis.comparison.metrics import compute_metrics, primary_metric_for_task
from mentis.comparison.model_zoo import get_model_zoo
from mentis.comparison.trainer import ModelTrainer

__all__ = [
    "ModelTrainer",
    "Leaderboard",
    "ModelResult",
    "build_leaderboard",
    "get_model_zoo",
    "compute_metrics",
    "primary_metric_for_task",
]


