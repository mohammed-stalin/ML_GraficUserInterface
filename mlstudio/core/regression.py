"""Regression model comparison with training-only preprocessing.

Improvements over the previous version:
* Skewed positive targets (prices, counts) are modelled on a log scale and
  converted back, which lowers MAE substantially on the bundled Car Price data.
* Ridge tunes its regularisation strength with built-in cross-validation and
  sees every category (sparse one-hot) instead of a fixed alpha.
* Histogram gradient boosting joins the candidate models.
* The fitted best model and its test predictions are returned for plotting.

Run as a script to benchmark a CSV file::

    python -m mlstudio.core.regression "Car Price.csv" --target Selling_Price
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .features import FeatureSpec, build_preprocessor, infer_features, prepare_frame

RANDOM_STATE = 42

# name -> (factory, preprocessor options)
REGRESSORS = {
    "Median baseline": (lambda: DummyRegressor(strategy="median"), {"max_categories": 40}),
    "Ridge (tuned)": (lambda: RidgeCV(alphas=np.logspace(-2, 3, 12)), {"max_categories": None, "sparse": True}),
    "Decision tree": (lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
                      {"max_categories": 40}),
    "Random forest": (lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.8,
                                                    n_jobs=-1, random_state=RANDOM_STATE), {"max_categories": 40}),
    "Extra trees": (lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                random_state=RANDOM_STATE), {"max_categories": 40}),
    "Gradient boosting": (lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE), {"max_categories": 40}),
}


@dataclass
class RegressionReport:
    target: str
    results: pd.DataFrame
    best_model: str
    best_pipeline: Pipeline
    y_test: pd.Series
    y_pred: np.ndarray
    log_target: bool
    spec: FeatureSpec
    input_rows: int
    duplicates_removed: int
    train_rows: int
    test_rows: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "input_rows": self.input_rows,
            "duplicates_removed": self.duplicates_removed,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "log_target": self.log_target,
            "selection": "Lowest training 3-fold cross-validation MAE",
            "results": self.results.to_dict(orient="records"),
        }


def should_log_target(y: pd.Series) -> bool:
    return bool((y > 0).all() and y.skew() > 1.0)


def build_regressor(name: str, spec: FeatureSpec, log_target: bool) -> Pipeline:
    factory, prep_options = REGRESSORS[name]
    model = factory()
    if log_target and name != "Median baseline":
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("prep", build_preprocessor(spec, **prep_options)), ("model", model)])


def compare_regressors(frame: pd.DataFrame, target: str, log_target: bool | None = None,
                       model_names: list[str] | None = None, test_size: float = 0.2) -> RegressionReport:
    """Rank models by 3-fold CV MAE on the training split, then score them on the test split."""
    if target not in frame:
        raise ValueError("Select an existing target column.")
    warnings: list[str] = []
    data = frame.drop_duplicates().dropna(subset=[target]).copy()
    try:
        y = pd.to_numeric(data[target], errors="raise").astype(float)
    except (ValueError, TypeError):
        raise ValueError(f"'{target}' is not numeric. Use the Classification page for categories.") from None
    if len(data) < 20:
        raise ValueError("Use at least 20 distinct rows with a numeric target.")
    if not np.isfinite(y).all():
        raise ValueError("Target values must be finite numbers.")
    if y.nunique() <= 10:
        warnings.append(f"'{target}' has only {y.nunique()} distinct values; classification may suit it better.")
    if log_target is None:
        log_target = should_log_target(y)
    elif log_target and (y <= -1).any():
        raise ValueError("A log-scale target needs values greater than -1.")

    X = data.drop(columns=[target])
    spec = infer_features(X)
    if spec.dropped:
        warnings.append("Ignored columns: " + ", ".join(f"{c} ({why})" for c, why in spec.dropped.items()))
    X = prepare_frame(X, spec)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name in model_names or list(REGRESSORS):
        start = time.perf_counter()
        scores = -cross_val_score(build_regressor(name, spec, log_target), X_train, y_train,
                                  cv=KFold(3, shuffle=True, random_state=RANDOM_STATE),
                                  scoring="neg_mean_absolute_error")
        pipeline = build_regressor(name, spec, log_target).fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        fitted[name] = (pipeline, prediction)
        rows.append({
            "Model": name,
            "CV MAE": float(scores.mean()),
            "CV MAE std": float(scores.std()),
            "Test MAE": float(mean_absolute_error(y_test, prediction)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "Test R2": float(r2_score(y_test, prediction)),
            "Time (s)": time.perf_counter() - start,
        })
    results = pd.DataFrame(rows).sort_values("CV MAE", ignore_index=True)
    best = results.loc[0, "Model"]
    return RegressionReport(
        target=target, results=results, best_model=best, best_pipeline=fitted[best][0],
        y_test=y_test, y_pred=fitted[best][1], log_target=log_target, spec=spec,
        input_rows=len(frame), duplicates_removed=int(frame.duplicated().sum()),
        train_rows=len(X_train), test_rows=len(X_test), warnings=warnings,
    )


if __name__ == "__main__":
    import argparse

    from .dataset import read_table

    parser = argparse.ArgumentParser(description="Benchmark regression models on a tabular file.")
    parser.add_argument("path")
    parser.add_argument("--target", default="Selling_Price")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()
    report = compare_regressors(read_table(args.path), args.target)
    text = json.dumps(report.to_dict(), indent=2)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
