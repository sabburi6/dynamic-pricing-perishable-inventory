"""
Loads and splits the perishable dataset.

Public functions:
- load_perishable()      → cleaned DataFrame (excludes Pharmaceuticals/Frozen_Meals)
- build_daily_aggregates(df) → (ts_pivot, price_daily) for forecasting
- make_three_way_split(ts_pivot) → train/val/holdout date indices
"""
from __future__ import annotations

import pandas as pd

import config


def load_perishable() -> pd.DataFrame:
    """Load Kaggle Perishable Goods dataset and apply category exclusions."""
    if not config.PERISHABLE_CSV.exists():
        raise FileNotFoundError(
            f"Perishable dataset not found at {config.PERISHABLE_CSV}.\n"
            f"Place perishable_goods_management.csv in {config.DATA_DIR}/."
        )
    df = pd.read_csv(config.PERISHABLE_CSV, parse_dates=["transaction_date"])
    clean = df[~df["category"].isin(config.EXCLUDE_CATS)].copy()
    return clean


def build_daily_aggregates(perishable_clean: pd.DataFrame):
    """
    Build the daily category-level pivot used by the demand forecaster
    plus a price_daily table used as merge-in features.
    """
    daily_ts = (
        perishable_clean
        .groupby(["transaction_date", "category"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "daily_units"})
    )

    price_daily = (
        perishable_clean
        .groupby(["transaction_date", "category"])
        .agg(
            avg_price=("selling_price", "mean"),
            avg_discount=("discount_pct", "mean"),
            pct_promoted=("is_promoted", "mean"),
        )
        .reset_index()
        .rename(columns={"transaction_date": "date"})
    )

    ts_pivot = (
        daily_ts
        .pivot(index="transaction_date", columns="category", values="daily_units")
        .fillna(0)
        .sort_index()
    )

    return ts_pivot, price_daily


def make_three_way_split(ts_pivot: pd.DataFrame):
    """Split the date index into train / val / holdout windows."""
    all_dates = ts_pivot.index.sort_values()
    holdout_dates = all_dates[-config.HOLDOUT_DAYS:]
    val_dates = all_dates[-(config.HOLDOUT_DAYS + config.VAL_DAYS):-config.HOLDOUT_DAYS]
    train_dates = all_dates[:-(config.HOLDOUT_DAYS + config.VAL_DAYS)]
    return train_dates, val_dates, holdout_dates


if __name__ == "__main__":
    df = load_perishable()
    print(f"Loaded {len(df):,} rows | {df['category'].nunique()} categories")
    ts_pivot, price_daily = build_daily_aggregates(df)
    train_d, val_d, hold_d = make_three_way_split(ts_pivot)
    print(f"Train: {train_d[0].date()} → {train_d[-1].date()} ({len(train_d)} days)")
    print(f"Val  : {val_d[0].date()} → {val_d[-1].date()} ({len(val_d)} days)")
    print(f"Hold : {hold_d[0].date()} → {hold_d[-1].date()} ({len(hold_d)} days)")
