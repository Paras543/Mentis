"""
Shared pytest fixtures for the entire Mentis test suite.

All common DataFrames, trained models, and Guardian instances are
defined here so individual test modules stay focused on assertions
rather than setup boilerplate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from mentis import Guardian
from mentis.config import MentisConfig

# ── Random seed ──────────────────────────────────────────────────────────────
SEED = 42


# ── Raw DataFrames ────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, fully clean DataFrame with no issues."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, 100).astype(float),
            "income": rng.uniform(20_000, 120_000, 100),
            "score": rng.uniform(0, 1, 100),
            "target": rng.integers(0, 2, 100),
        }
    )


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A DataFrame with missing values, duplicates, and a constant column."""
    rng = np.random.default_rng(SEED)
    n = 100
    data = {
        "age": rng.integers(18, 80, n).astype(float),
        "income": rng.uniform(20_000, 120_000, n),
        "constant_col": [1] * n,
        "target": rng.integers(0, 2, n),
    }
    df = pd.DataFrame(data)
    # Inject 20% missing in 'age'
    df.loc[df.sample(20, random_state=SEED).index, "age"] = np.nan
    # Inject 5% missing in 'income'
    df.loc[df.sample(5, random_state=SEED).index, "income"] = np.nan
    # Add 10 duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """A DataFrame with a highly imbalanced target (90/10 split)."""
    rng = np.random.default_rng(SEED)
    n = 200
    labels = [0] * 180 + [1] * 20
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.uniform(0, 10, n),
            "target": labels,
        }
    )


@pytest.fixture
def leakage_df() -> pd.DataFrame:
    """A DataFrame where a feature is perfectly correlated with the target."""
    rng = np.random.default_rng(SEED)
    n = 100
    target = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame(
        {
            "safe_feature": rng.normal(0, 1, n),
            "leaked_feature": target * 0.99 + rng.normal(0, 0.001, n),
            "target": target,
        }
    )


# ── Classification split ──────────────────────────────────────────────────────

@pytest.fixture
def classification_split():
    """Train/test split from sklearn's make_classification."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=SEED,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return train_test_split(X_df, y, test_size=0.25, random_state=SEED)


@pytest.fixture
def X_train_clf(classification_split):
    return classification_split[0]


@pytest.fixture
def X_test_clf(classification_split):
    return classification_split[1]


@pytest.fixture
def y_train_clf(classification_split):
    return classification_split[2]


@pytest.fixture
def y_test_clf(classification_split):
    return classification_split[3]


# ── Regression split ──────────────────────────────────────────────────────────

@pytest.fixture
def regression_split():
    """Train/test split from sklearn's make_regression."""
    res = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=SEED)
    X, y = res[0], res[1]
    X_df = pd.DataFrame(X, columns=[f"r{i}" for i in range(X.shape[1])])
    return train_test_split(X_df, y, test_size=0.25, random_state=SEED)


@pytest.fixture
def X_train_reg(regression_split):
    return regression_split[0]


@pytest.fixture
def X_test_reg(regression_split):
    return regression_split[1]


@pytest.fixture
def y_train_reg(regression_split):
    return regression_split[2]


@pytest.fixture
def y_test_reg(regression_split):
    return regression_split[3]


# ── Fitted models ─────────────────────────────────────────────────────────────

@pytest.fixture
def fitted_clf(X_train_clf, y_train_clf) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model.fit(X_train_clf, y_train_clf)
    return model


@pytest.fixture
def fitted_lr(X_train_clf, y_train_clf) -> LogisticRegression:
    model = LogisticRegression(random_state=SEED, max_iter=500)
    model.fit(X_train_clf, y_train_clf)
    return model


@pytest.fixture
def fitted_reg(X_train_reg, y_train_reg) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train_reg, y_train_reg)
    return model


@pytest.fixture
def fitted_rf_reg(X_train_reg, y_train_reg) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=10, random_state=SEED)
    model.fit(X_train_reg, y_train_reg)
    return model


# ── Guardian instances ────────────────────────────────────────────────────────

@pytest.fixture
def guardian() -> Guardian:
    """Default Guardian with classification task."""
    return Guardian(config=MentisConfig())


@pytest.fixture
def clf_guardian() -> Guardian:
    """Guardian configured for classification."""
    from mentis.config import ProjectConfig
    return Guardian(config=MentisConfig(project=ProjectConfig(task="classification", target="target")))


@pytest.fixture
def reg_guardian() -> Guardian:
    """Guardian configured for regression."""
    from mentis.config import ProjectConfig
    return Guardian(config=MentisConfig(project=ProjectConfig(task="regression")))
