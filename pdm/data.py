"""Data loading and feature engineering.

Mirrors the notebook exactly so that what the app shows matches what the
model was trained on:
  1. read the semicolon-separated CSV,
  2. parse the ``date`` column to datetime,
  3. derive temporal features (day_of_week, day_of_month, month).
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the machine-event CSV with the correct separator and dtypes.

    The ``date`` column is parsed to datetime. The raw file stores it as a
    US-style ``mm/dd/yyyy`` string (which pandas reads as ``object``); this
    matches the notebook's month-first interpretation.
    """
    path = path or config.DATA_PATH
    df = pd.read_csv(path, sep=config.CSV_SEP)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format=config.DATE_FORMAT)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append day_of_week / day_of_month / month columns derived from ``date``.

    Returns a copy; the input frame is left untouched.
    """
    out = df.copy()
    dt = out[config.DATE_COL]
    out["day_of_week"] = dt.dt.dayofweek
    out["day_of_month"] = dt.dt.day
    out["month"] = dt.dt.month
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame containing the base model features (pre anomaly score).

    Columns are ordered as ``config.BASE_FEATURES``.
    """
    engineered = add_date_features(df)
    return engineered[config.BASE_FEATURES].copy()


def load_dataset() -> pd.DataFrame:
    """Load the raw data and attach temporal features in one call.

    The returned frame keeps ``date``, ``machine`` and ``event`` alongside the
    engineered features so the app can slice / group by them.
    """
    return add_date_features(load_raw())
