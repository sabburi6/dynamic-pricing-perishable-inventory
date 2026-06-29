"""
Feature engineering for the demand forecaster.

Builds lag, rolling, calendar, and price features. All lags use shift(1)
before any rolling so the window for any row only sees data that precedes
that row chronologically — this prevents temporal leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def make_ts_features(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lagged feature matrix from the daily category pivot.

    Returns a long-format DataFrame with columns:
        date, category, y, lag_1, lag_7, lag_14,
        roll_mean_7, roll_mean_14, roll_std_7,
        day_of_week, month, day_of_month, is_weekend
    """
    records = []
    for cat in pivot_df.columns:
        s = pivot_df[cat]
        df = pd.DataFrame({
            "y":            s,
            "lag_1":        s.shift(1),
            "lag_7":        s.shift(7),
            "lag_14":       s.shift(14),
            "roll_mean_7":  s.shift(1).rolling(7).mean(),
            "roll_mean_14": s.shift(1).rolling(14).mean(),
            "roll_std_7":   s.shift(1).rolling(7).std(),
        })
        df["day_of_week"]  = s.index.dayofweek
        df["month"]        = s.index.month
        df["day_of_month"] = s.index.day
        df["is_weekend"]   = (s.index.dayofweek >= 5).astype(int)
        df["category"]     = cat
        records.append(df.dropna())

    # Force the index to be named "date" before reset_index so we get a
    # "date" column regardless of pandas version (older pandas does not
    # support reset_index(names=...) and the original index name varies
    # depending on whether ts_pivot's index name was set).
    out = pd.concat(records)
    out.index.name = "date"
    return out.reset_index()


def build_demand_feature_set(ts_pivot: pd.DataFrame, price_daily: pd.DataFrame,
                             train_dates, val_dates, holdout_dates):
    """
    Build the full feature DataFrame and slice into train/val/holdout.
    Returns a dict with keys:
        feat_train, feat_val, feat_holdout,
        X_train, y_train, X_val, y_val, X_hold, y_hold
    """
    feat_df = make_ts_features(ts_pivot)
    feat_df["cat_code"] = pd.Categorical(feat_df["category"]).codes

    # price_daily uses "date" as its date column (set in data_loading); merge in.
    feat_df = feat_df.merge(price_daily, on=["date", "category"], how="left")
    feat_df[["avg_price", "avg_discount", "pct_promoted"]] = (
        feat_df[["avg_price", "avg_discount", "pct_promoted"]].fillna(0)
    )

    feat_sorted = feat_df.sort_values("date").reset_index(drop=True)
    is_holdout = feat_sorted["date"].isin(holdout_dates)
    is_val = feat_sorted["date"].isin(val_dates)
    is_train = ~is_holdout & ~is_val

    feat_train = feat_sorted[is_train].copy()
    feat_val = feat_sorted[is_val].copy()
    feat_holdout = feat_sorted[is_holdout].copy()

    cols = config.DEMAND_FEAT_COLS
    return {
        "feat_train":   feat_train,
        "feat_val":     feat_val,
        "feat_holdout": feat_holdout,
        "X_train":      feat_train[cols],
        "y_train":      feat_train["y"],
        "X_val":        feat_val[cols],
        "y_val":        feat_val["y"],
        "X_hold":       feat_holdout[cols],
        "y_hold":       feat_holdout["y"],
    }
