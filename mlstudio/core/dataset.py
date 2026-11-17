"""Dataset loading, profiling and reversible cleaning operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
MAX_HISTORY = 30


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV (separator auto-detected) or Excel files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use CSV, TSV or Excel.")
    if df.empty or not len(df.columns):
        raise ValueError("The file contains no data.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@dataclass
class Dataset:
    df: pd.DataFrame
    source: Path | None = None
    log: list[str] = field(default_factory=list)
    _history: list[tuple[pd.DataFrame, list[str]]] = field(default_factory=list, repr=False)
    _original: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        if self._original is None:
            self._original = self.df.copy()

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        return cls(read_table(path), Path(path))

    @property
    def name(self) -> str:
        return self.source.name if self.source else "dataset"

    # ---------------------------------------------------------------- history
    def apply(self, description: str, operation: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Run an operation on a copy; commit it only if it succeeds."""
        result = operation(self.df.copy())
        if result is None or not isinstance(result, pd.DataFrame):
            raise TypeError("Cleaning operations must return a DataFrame.")
        if result.empty:
            raise ValueError("This operation would remove every row. It was not applied.")
        self._history.append((self.df, list(self.log)))
        del self._history[:-MAX_HISTORY]
        self.df = result
        self.log.append(description)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> None:
        if self._history:
            self.df, self.log = self._history.pop()

    def reset(self) -> None:
        self._history.clear()
        self.log.clear()
        self.df = self._original.copy()

    # -------------------------------------------------------------- profiling
    def summary(self) -> dict:
        df = self.df
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": len(numeric_columns(df)),
            "categorical_columns": len(categorical_columns(df)),
        }

    def profile(self) -> pd.DataFrame:
        df = self.df
        rows = []
        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            sample = s.dropna().head(3).astype(str).tolist()
            rows.append({
                "Column": col,
                "Type": str(s.dtype),
                "Missing": missing,
                "Missing %": f"{missing / max(len(s), 1):.1%}",
                "Unique": int(s.nunique(dropna=True)),
                "Examples": ", ".join(sample),
            })
        return pd.DataFrame(rows)

    def describe(self) -> pd.DataFrame:
        num = self.df.select_dtypes(include="number")
        if num.empty:
            return pd.DataFrame()
        desc = num.describe().T
        desc.insert(0, "Column", desc.index)
        return desc.reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df)]


# ------------------------------------------------------------------ operations
def _require(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    columns = list(columns) or list(df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Unknown column(s): {', '.join(missing)}")
    return columns


def fill_missing(df: pd.DataFrame, strategy: str, columns: Iterable[str] = ()) -> pd.DataFrame:
    """Fill gaps. Numeric columns use mean/median; text columns always fall back to the mode."""
    columns = _require(df, columns)
    for col in columns:
        s = df[col]
        if not s.isna().any():
            continue
        if strategy in ("mean", "median") and col in numeric_columns(df):
            value = s.mean() if strategy == "mean" else s.median()
        elif strategy == "zero" and col in numeric_columns(df):
            value = 0
        else:
            mode = s.mode(dropna=True)
            if mode.empty:
                continue
            value = mode.iloc[0]
        df[col] = s.fillna(value)
    return df


def drop_missing_rows(df: pd.DataFrame, columns: Iterable[str] = ()) -> pd.DataFrame:
    return df.dropna(subset=_require(df, columns))


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = list(columns)
    if not columns:
        raise ValueError("Select at least one column to drop.")
    if len(columns) >= len(df.columns):
        raise ValueError("At least one column must remain.")
    return df.drop(columns=_require(df, columns))


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def one_hot_encode(df: pd.DataFrame, columns: Iterable[str], max_categories: int = 50) -> pd.DataFrame:
    columns = list(columns)
    if not columns:
        raise ValueError("Select at least one column to encode.")
    _require(df, columns)
    too_many = [c for c in columns if df[c].nunique() > max_categories]
    if too_many:
        raise ValueError(f"Too many categories to one-hot encode (>{max_categories}): {', '.join(too_many)}. "
                         "Drop these columns or use ordinal encoding instead.")
    return pd.get_dummies(df, columns=columns, prefix=columns, dtype=int)


def ordinal_encode(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Replace categories with integer codes (missing values stay missing)."""
    columns = list(columns)
    if not columns:
        raise ValueError("Select at least one column to encode.")
    for col in _require(df, columns):
        codes = pd.Categorical(df[col]).codes.astype(float)
        codes[codes < 0] = np.nan
        df[col] = codes
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: Iterable[str] = (), factor: float = 1.5) -> pd.DataFrame:
    columns = [c for c in _require(df, columns) if c in numeric_columns(df)]
    if not columns:
        raise ValueError("Select at least one numeric column.")
    keep = pd.Series(True, index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        spread = q3 - q1
        inside = df[col].between(q1 - factor * spread, q3 + factor * spread) | df[col].isna()
        keep &= inside
    return df[keep]


def standardize(df: pd.DataFrame, columns: Iterable[str] = ()) -> pd.DataFrame:
    columns = [c for c in _require(df, columns) if c in numeric_columns(df)]
    if not columns:
        raise ValueError("Select at least one numeric column.")
    for col in columns:
        std = df[col].std()
        df[col] = (df[col] - df[col].mean()) / (std if std else 1)
    return df
