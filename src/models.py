"""Model training, prediction, and calibration for March Madness.

Supports XGBoost (point differential regression with Cauchy loss),
CatBoost, LightGBM, and Random Forest. Includes spline-based probability
calibration and ensemble averaging.
"""

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import log_loss, mean_squared_error


# ---------------------------------------------------------------------------
# Custom XGBoost loss (Cauchy / heavy-tailed)
# ---------------------------------------------------------------------------

def cauchyobj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Cauchy loss function for XGBoost — robust to outliers in point differential.

    Args:
        preds: Current predictions.
        dtrain: Training DMatrix with labels.

    Returns:
        Tuple of (gradient, hessian).
    """
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    grad = x / (x**2 / c**2 + 1)
    hess = -c**2 * (x**2 - c**2) / (x**2 + c**2) ** 2
    return grad, hess


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = {
    "eval_metric": "mae",
    "booster": "gbtree",
    "eta": 0.02,
    "subsample": 0.35,
    "colsample_bytree": 0.7,
    "num_parallel_tree": 10,
    "min_child_weight": 40,
    "gamma": 10,
    "max_depth": 3,
    "verbosity": 0,
}


def train_xgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    n_splits: int = 5,
    repeat_cv: int = 3,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 25,
    use_cauchy: bool = True,
    verbose: bool = True,
) -> tuple[list, list, list[int]]:
    """Train XGBoost with repeated K-Fold CV and collect OOF predictions.

    Args:
        X: Feature matrix.
        y: Target (point differential).
        params: XGBoost parameters. Defaults to DEFAULT_XGB_PARAMS.
        n_splits: Number of CV folds.
        repeat_cv: Number of CV repetitions.
        num_boost_round: Max boosting rounds.
        early_stopping_rounds: Early stopping patience.
        use_cauchy: Whether to use Cauchy custom loss.
        verbose: Print progress.

    Returns:
        Tuple of (oof_predictions_list, cv_results_list, iteration_counts).
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    dtrain = xgb.DMatrix(X, label=y)
    obj = cauchyobj if use_cauchy else None

    # Phase 1: CV to find optimal iterations
    cv_results = []
    for i in range(repeat_cv):
        if verbose:
            print(f"  CV repeat {i + 1}/{repeat_cv}")
        cv_result = xgb.cv(
            params=params,
            dtrain=dtrain,
            obj=obj,
            num_boost_round=num_boost_round,
            folds=KFold(n_splits=n_splits, shuffle=True, random_state=i),
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100 if verbose else 0,
        )
        cv_results.append(cv_result)

    iteration_counts = [np.argmin(r["test-mae-mean"].values) for r in cv_results]

    # Phase 2: OOF predictions
    oof_preds = []
    for i in range(repeat_cv):
        if verbose:
            print(f"  OOF predictions repeat {i + 1}/{repeat_cv}")
        preds = y.copy().astype(float)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        for train_idx, val_idx in kfold.split(X, y):
            dtrain_i = xgb.DMatrix(X[train_idx], label=y[train_idx])
            dval_i = xgb.DMatrix(X[val_idx], label=y[val_idx])
            model = xgb.train(
                params=params,
                dtrain=dtrain_i,
                obj=obj,
                num_boost_round=iteration_counts[i],
                verbose_eval=0,
            )
            preds[val_idx] = model.predict(dval_i)
        oof_preds.append(np.clip(preds, -30, 30))

    return oof_preds, cv_results, iteration_counts


def train_xgb_final(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[dict] = None,
    iteration_counts: Optional[list[int]] = None,
    repeat_cv: int = 3,
    use_cauchy: bool = True,
    verbose: bool = True,
) -> list:
    """Train final XGBoost models on all training data.

    Args:
        X: Full training feature matrix.
        y: Full training target (point differential).
        params: XGBoost parameters.
        iteration_counts: Number of boosting rounds per model (from CV).
        repeat_cv: Number of models to train.
        use_cauchy: Whether to use Cauchy custom loss.
        verbose: Print progress.

    Returns:
        List of trained XGBoost Booster objects.
    """
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()
    if iteration_counts is None:
        iteration_counts = [500] * repeat_cv

    dtrain = xgb.DMatrix(X, label=y)
    obj = cauchyobj if use_cauchy else None

    models = []
    for i in range(repeat_cv):
        if verbose:
            print(f"  Training final model {i + 1}/{repeat_cv}")
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            obj=obj,
            num_boost_round=int(iteration_counts[i] * 1.05),
            verbose_eval=100 if verbose else 0,
        )
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def fit_spline_calibrators(
    oof_preds: list[np.ndarray],
    y: np.ndarray,
) -> list[UnivariateSpline]:
    """Fit UnivariateSpline calibrators mapping point diff → win probability.

    Args:
        oof_preds: List of OOF point differential predictions.
        y: True point differential.

    Returns:
        List of fitted UnivariateSpline objects.
    """
    labels = (y > 0).astype(int)
    spline_models = []

    for preds in oof_preds:
        dat = sorted(zip(preds, labels), key=lambda x: x[0])
        dat_dict = dict(dat)
        spline = UnivariateSpline(list(dat_dict.keys()), list(dat_dict.values()))
        spline_models.append(spline)

    return spline_models


def predict_probabilities(
    X: np.ndarray,
    models: list,
    spline_models: list[UnivariateSpline],
    clip_range: tuple[float, float] = (0.025, 0.975),
) -> np.ndarray:
    """Generate calibrated win probabilities from XGBoost ensemble.

    Args:
        X: Feature matrix for prediction.
        models: List of trained XGBoost Booster objects.
        spline_models: List of fitted spline calibrators.
        clip_range: Min/max probability bounds.

    Returns:
        Array of calibrated win probabilities.
    """
    dtest = xgb.DMatrix(X)
    preds = []
    for model, spline in zip(models, spline_models):
        raw_pred = np.clip(model.predict(dtest), -30, 30)
        prob = np.clip(spline(raw_pred), clip_range[0], clip_range[1])
        preds.append(prob)

    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

def train_rf_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 8,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier as a baseline.

    Args:
        X: Feature matrix (difference features).
        y: Binary target (1 = Team1 wins).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compute MSE and log loss for predictions.

    Args:
        y_true: True binary outcomes (1 = Team1 wins).
        y_pred: Predicted probabilities.
        label: Optional label for printing.

    Returns:
        Dict with 'mse' and 'log_loss' keys.
    """
    mse = mean_squared_error(y_true, y_pred)
    ll = log_loss(y_true, y_pred)

    if label:
        print(f"  {label} — MSE: {mse:.6f}, LogLoss: {ll:.6f}")

    return {"mse": mse, "log_loss": ll}
