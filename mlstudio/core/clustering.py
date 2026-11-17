"""K-Means clustering with scaling, automatic k selection and cluster profiles.

Improvements over the original implementation:
* Features are standardised before K-Means/PCA (previously a column such as
  "KM_Driven" dominated every distance because of its scale).
* Missing values are imputed instead of crashing.
* An elbow + silhouette sweep suggests the number of clusters.
* Cluster profiles summarise what each group looks like.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .features import FeatureSpec, build_preprocessor, infer_features, prepare_frame

RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 3000


@dataclass
class ClusterResult:
    k: int
    labels: np.ndarray
    coords: np.ndarray
    explained_variance: np.ndarray
    silhouette: float | None
    inertia: float
    sizes: pd.Series
    profile: pd.DataFrame
    spec: FeatureSpec


def _matrix(df: pd.DataFrame, exclude=(), include_categorical: bool = False):
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    spec = infer_features(X, include_categorical=include_categorical)
    if len(X) < 3:
        raise ValueError("At least 3 rows are needed for clustering.")
    frame = prepare_frame(X, spec)
    return build_preprocessor(spec).fit_transform(frame), spec


def _silhouette(matrix, labels) -> float | None:
    if len(set(labels)) < 2:
        return None
    size = min(SILHOUETTE_SAMPLE, len(labels))
    return float(silhouette_score(matrix, labels, sample_size=size, random_state=RANDOM_STATE))


def sweep_k(df: pd.DataFrame, k_min: int = 2, k_max: int = 10, exclude=(),
            include_categorical: bool = False) -> pd.DataFrame:
    matrix, _ = _matrix(df, exclude, include_categorical)
    k_max = min(k_max, len(matrix) - 1)
    if k_max < k_min:
        raise ValueError("Not enough rows to compare cluster counts.")
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(matrix)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": _silhouette(matrix, model.labels_)})
    return pd.DataFrame(rows)


def best_k(sweep: pd.DataFrame) -> int:
    return int(sweep.loc[sweep["silhouette"].idxmax(), "k"])


def run_kmeans(df: pd.DataFrame, k: int, exclude=(), include_categorical: bool = False) -> ClusterResult:
    matrix, spec = _matrix(df, exclude, include_categorical)
    if not 1 <= k < len(matrix):
        raise ValueError(f"Choose between 1 and {len(matrix) - 1} clusters.")
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(matrix)
    labels = model.labels_
    n_comp = min(2, matrix.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    coords = pca.fit_transform(matrix)
    if n_comp == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    source = df.drop(columns=[c for c in exclude if c in df.columns])
    numeric = source[spec.numeric].astype(float)
    profile = numeric.groupby(labels).mean()
    profile.insert(0, "Size", pd.Series(labels).value_counts().sort_index())
    for col in spec.categorical:
        profile[col + " (most common)"] = source[col].groupby(labels).agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else "")
    profile.index.name = "Cluster"
    return ClusterResult(
        k=k, labels=labels, coords=coords, explained_variance=pca.explained_variance_ratio_,
        silhouette=_silhouette(matrix, labels), inertia=float(model.inertia_),
        sizes=pd.Series(labels).value_counts().sort_index(), profile=profile.reset_index(), spec=spec,
    )
