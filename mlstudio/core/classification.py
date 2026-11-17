"""Supervised classification: training, evaluation and model comparison.

Improvements over the original implementation:
* 80/20 stratified train/test split (the old code used ``test_size=0.8`` and
  trained on only 20% of the rows).
* Every model runs in a pipeline with imputation, scaling and one-hot encoding
  fitted on the training split only, so SVM / logistic regression / KNN get
  scaled inputs and text columns no longer crash training.
* Stratified k-fold cross-validation reports a mean +/- std instead of a single
  noisy split score, plus balanced accuracy and macro F1 for imbalanced data.
* Model-agnostic permutation importance and a PCA projection of predictions.
"""
from __future__ import annotations

import time
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .features import FeatureSpec, build_preprocessor, infer_features, prepare_frame

RANDOM_STATE = 42

CLASSIFIERS: dict[str, Callable[[], ClassifierMixin]] = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=3000),
    "Naive Bayes": GaussianNB,
    "SVM (RBF kernel)": lambda: SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE),
    "SVM (linear)": lambda: LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
    "Decision Tree (ID3)": lambda: DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2,
                                                          random_state=RANDOM_STATE),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

TREE_MODELS = ("Decision Tree (ID3)",)


@dataclass
class ClassificationResult:
    model_name: str
    target: str
    pipeline: Pipeline
    spec: FeatureSpec
    labels: list
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    cv_accuracy: float
    cv_accuracy_std: float
    confusion: np.ndarray
    report: pd.DataFrame
    report_text: str
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    train_rows: int
    test_rows: int
    warnings: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["prep"].get_feature_names_out())


def prepare_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Validate the target column and drop rows that make stratified splitting impossible."""
    if target not in df.columns:
        raise ValueError("Select an existing target column.")
    warnings: list[str] = []
    data = df.dropna(subset=[target])
    if len(data) < len(df):
        warnings.append(f"{len(df) - len(data)} rows with a missing target were ignored.")
    y = data[target]
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError(f"'{target}' has a single class. Classification needs at least two.")
    is_number = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if is_number and n_classes > 20:
        raise ValueError(f"'{target}' looks continuous ({n_classes} distinct values). "
                         "Use the Regression page to predict numbers.")
    if n_classes > max(50, 0.5 * len(y)):
        raise ValueError(f"'{target}' has {n_classes} classes, which is too many to classify reliably.")
    counts = y.value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        data = data[~y.isin(rare)]
        warnings.append(f"{len(rare)} class(es) with a single example were ignored.")
        if data[target].nunique() < 2:
            raise ValueError("Not enough examples per class to train and test.")
    if len(data) < 10:
        raise ValueError("At least 10 labelled rows are needed.")
    y = data[target]
    if y.dtype == object:
        y = y.astype(str)
    return data.drop(columns=[target]), y, warnings


def _cv_splitter(y: pd.Series) -> StratifiedKFold:
    folds = int(max(2, min(5, y.value_counts().min())))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def _split(X, y, test_size):
    stratify = y if y.value_counts().min() >= 2 else None
    n_classes = y.nunique()
    test_count = max(int(round(len(y) * test_size)), n_classes)
    return train_test_split(X, y, test_size=test_count, random_state=RANDOM_STATE, stratify=stratify)


def _balanced_accuracy(y_true, y_pred) -> float:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)  # a predicted class may be absent from y_true
        return float(balanced_accuracy_score(y_true, y_pred))


def build_pipeline(model_name: str, spec: FeatureSpec) -> Pipeline:
    if model_name not in CLASSIFIERS:
        raise ValueError(f"Unknown model '{model_name}'.")
    return Pipeline([("prep", build_preprocessor(spec)), ("model", CLASSIFIERS[model_name]())])


def train_classifier(df: pd.DataFrame, target: str, model_name: str, test_size: float = 0.2,
                     include_categorical: bool = True) -> ClassificationResult:
    X, y, warnings = prepare_target(df, target)
    spec = infer_features(X, include_categorical=include_categorical)
    if spec.dropped:
        warnings.append("Ignored columns: " + ", ".join(f"{c} ({why})" for c, why in spec.dropped.items()))
    X = prepare_frame(X, spec)
    X_train, X_test, y_train, y_test = _split(X, y, test_size)

    cv = cross_validate(build_pipeline(model_name, spec), X_train, y_train, cv=_cv_splitter(y_train),
                        scoring="accuracy", n_jobs=None)
    pipeline = build_pipeline(model_name, spec).fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    labels = sorted(pd.unique(pd.concat([y_test, pd.Series(y_pred, index=y_test.index)])), key=str)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = (pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"}))
    return ClassificationResult(
        model_name=model_name, target=target, pipeline=pipeline, spec=spec, labels=labels,
        accuracy=accuracy_score(y_test, y_pred),
        balanced_accuracy=_balanced_accuracy(y_test, y_pred),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        cv_accuracy=float(cv["test_score"].mean()), cv_accuracy_std=float(cv["test_score"].std()),
        confusion=confusion_matrix(y_test, y_pred, labels=labels), report=report_df,
        report_text=classification_report(y_test, y_pred, labels=labels, zero_division=0),
        X_test=X_test, y_test=y_test, y_pred=y_pred,
        train_rows=len(X_train), test_rows=len(X_test), warnings=warnings,
    )


def compare_classifiers(df: pd.DataFrame, target: str, model_names: list[str] | None = None,
                        test_size: float = 0.2, include_categorical: bool = True) -> pd.DataFrame:
    """Cross-validate every model on the same training split; rank by mean CV macro F1."""
    X, y, _ = prepare_target(df, target)
    spec = infer_features(X, include_categorical=include_categorical)
    X = prepare_frame(X, spec)
    X_train, X_test, y_train, y_test = _split(X, y, test_size)
    splitter = _cv_splitter(y_train)
    rows = []
    for name in model_names or list(CLASSIFIERS):
        start = time.perf_counter()
        cv = cross_validate(build_pipeline(name, spec), X_train, y_train, cv=splitter,
                            scoring=("accuracy", "f1_macro"))
        fitted = build_pipeline(name, spec).fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        rows.append({
            "Model": name,
            "CV accuracy": cv["test_accuracy"].mean(),
            "CV std": cv["test_accuracy"].std(),
            "CV macro F1": cv["test_f1_macro"].mean(),
            "Test accuracy": accuracy_score(y_test, y_pred),
            "Test macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Time (s)": time.perf_counter() - start,
        })
    return pd.DataFrame(rows).sort_values("CV macro F1", ascending=False, ignore_index=True)


def feature_importance(result: ClassificationResult, max_rows: int = 2000, top: int = 15) -> pd.Series:
    """Permutation importance on the held-out set, aggregated to original columns."""
    X, y = result.X_test, result.y_test
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=RANDOM_STATE)
        y = y.loc[X.index]
    imp = permutation_importance(result.pipeline, X, y, n_repeats=5, random_state=RANDOM_STATE,
                                 scoring="accuracy", n_jobs=None)
    series = pd.Series(imp.importances_mean, index=X.columns)
    return series.sort_values(ascending=False).head(top)


def projection(result: ClassificationResult) -> tuple[np.ndarray, np.ndarray]:
    """2-D PCA coordinates of the transformed test set plus explained variance ratios."""
    transformed = result.pipeline.named_steps["prep"].transform(result.X_test)
    n_comp = min(2, transformed.shape[1], transformed.shape[0])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    coords = pca.fit_transform(transformed)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords, pca.explained_variance_ratio_
