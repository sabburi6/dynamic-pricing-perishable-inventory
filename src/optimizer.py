"""
Stage 4 — Optimization engine + 60-day batch simulation.

The optimizer evaluates candidate discounts {0%, 10%, 20%, 30%, 40%},
enforces the margin floor (price >= cost * 1.05), and picks the discount
that maximises expected profit per SKU per day.

Key formulas:
- demand_hat = base_demand * (price / ref_price) ** elasticity
- For near-expiry items (days_until_expiry <= 5):
    elasticity = NEAR_EXPIRY_ELAS (-2.0)
    exp_sold   = min(demand_hat, init_qty)
    exp_waste  = max(init_qty - exp_sold, 0)
- Otherwise:
    exp_waste  = spoilage_prob * empirical_waste_fraction * init_qty
    exp_sold   = min(demand_hat, max(init_qty - exp_waste, 0))
- profit       = (price - cost) * exp_sold - cost * exp_waste
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config


# ─────────────────────────────────────────────────────────────────────────────
# Forecast scaling: category-level → per-SKU-per-day
# ─────────────────────────────────────────────────────────────────────────────
def scale_forecasts_to_sku(category_forecasts: Dict[Tuple, float],
                           df_hold: pd.DataFrame) -> Dict[Tuple, float]:
    """
    The demand model produces total daily units PER CATEGORY.
    The simulator iterates per-SKU rows. Divide each forecast by the
    number of active SKUs in that category on that date.
    """
    sku_counts = (
        df_hold.groupby(["transaction_date", "category"])["product_id"]
        .nunique().reset_index()
        .rename(columns={"product_id": "n_skus"})
    )
    scaled = {}
    for _, row in sku_counts.iterrows():
        key = (row["transaction_date"], row["category"])
        n = row["n_skus"]
        if key in category_forecasts and n > 0:
            scaled[key] = category_forecasts[key] / n
        elif key in category_forecasts:
            scaled[key] = category_forecasts[key]
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# Single-row optimizer (used for stability checks; simulation uses batch_simulate)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_row(row: pd.Series, *,
                 spoilage_model, train_cols: list,
                 elasticity_map: dict, waste_frac: float) -> dict:
    """
    Choose the discount that maximises expected profit for one SKU-day row.
    Uses observed units_sold as the demand baseline — appropriate for the
    1k-row pre-holdout stability sample only. The 60-day OOS simulation
    uses batch_simulate with forecasts from the demand model.
    """
    dept = config.CAT_TO_DEPT.get(row["category"], "GROCERY")
    elasticity = elasticity_map.get(dept, -0.15)
    base_price = row["base_price"]
    sell_price = row["selling_price"]
    cost = row["cost_price"]
    init_qty = row["initial_quantity"]
    base_units = row["units_sold"]

    best = {"discount": 0.0, "profit": -np.inf, "opt_price": base_price,
            "exp_sold": base_units, "exp_waste": 0.0, "spoil_risk": 0.0}

    ref_price = sell_price if sell_price > 0 else base_price
    near_expiry = row["days_until_expiry"] <= config.NEAR_EXPIRY_DAYS
    elas_used = config.NEAR_EXPIRY_ELAS if near_expiry else elasticity

    for d in config.DISCOUNT_GRID:
        new_price = base_price * (1 - d)
        if new_price < cost * (1 + config.MIN_MARGIN):
            continue
        demand_hat = base_units * (new_price / ref_price) ** elas_used
        sim = row.copy()
        sim["selling_price"] = new_price
        sim["discount_pct"] = d
        X_row = (pd.get_dummies(pd.DataFrame([sim[config.SPOILAGE_FEATURES]]),
                                columns=["quality_grade"], drop_first=True)
                   .astype(float)
                   .reindex(columns=train_cols, fill_value=0))
        spoil_risk = float(spoilage_model.predict_proba(X_row)[:, 1][0])
        if near_expiry:
            exp_sold = min(demand_hat, init_qty)
            exp_waste = max(init_qty - exp_sold, 0)
        else:
            exp_waste = spoil_risk * waste_frac * init_qty
            exp_sold = min(demand_hat, max(init_qty - exp_waste, 0))
        profit = (new_price - cost) * exp_sold - cost * exp_waste
        if profit > best["profit"]:
            best = {"discount": d, "profit": profit, "opt_price": new_price,
                    "exp_sold": exp_sold, "exp_waste": exp_waste,
                    "spoil_risk": spoil_risk}
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch simulator (used by 60-day holdout simulation)
# ─────────────────────────────────────────────────────────────────────────────
def batch_simulate(df: pd.DataFrame, strategy: str, *,
                   spoilage_model, train_cols: list,
                   elasticity_map: dict, waste_frac: float,
                   demand_forecasts: Dict[Tuple, float]) -> pd.DataFrame:
    """
    Run one pricing strategy on the full holdout dataframe (vectorised).

    strategy ∈ {"no_discount", "fixed_20", "dynamic"}

    Demand forecast is pulled from `demand_forecasts` keyed by
    (transaction_date, category). If a key is missing, falls back to
    observed units_sold for that row.
    """
    # Build base feature frame once (used for all candidates)
    X_base = (pd.get_dummies(df[config.SPOILAGE_FEATURES].copy(),
                             columns=["quality_grade"], drop_first=True)
                .astype(float)
                .reindex(columns=train_cols, fill_value=0))

    base_price_arr = df["base_price"].values
    sell_price_arr = df["selling_price"].values
    cost_arr = df["cost_price"].values
    init_qty_arr = df["initial_quantity"].values

    units_arr = np.array([
        demand_forecasts.get((r, c), u)
        for r, c, u in zip(df["transaction_date"], df["category"], df["units_sold"])
    ])

    dept_series = df["category"].map(config.CAT_TO_DEPT).fillna("GROCERY")
    elas_arr = dept_series.map(elasticity_map).fillna(-0.15).values
    ref_arr = np.where(sell_price_arr > 0, sell_price_arr, base_price_arr)
    near_expiry_arr = df["days_until_expiry"].values <= config.NEAR_EXPIRY_DAYS
    elas_arr_final = np.where(near_expiry_arr, config.NEAR_EXPIRY_ELAS, elas_arr)

    if strategy == "no_discount":
        new_prices = base_price_arr.copy()
        discount_arr = np.zeros(len(df))

    elif strategy == "fixed_20":
        new_prices = base_price_arr * 0.80
        discount_arr = np.full(len(df), 0.20)

    elif strategy == "dynamic":
        best_profit = np.full(len(df), -np.inf)
        best_d = np.zeros(len(df))
        for d in config.DISCOUNT_GRID:
            cand_price = base_price_arr * (1 - d)
            margin_ok = cand_price >= cost_arr * (1 + config.MIN_MARGIN)
            X_d = X_base.copy()
            X_d["selling_price"] = cand_price
            X_d["discount_pct"] = d
            spoil = spoilage_model.predict_proba(X_d.values)[:, 1]
            demand_hat = units_arr * (cand_price / ref_arr) ** elas_arr_final
            exp_sold_ne = np.minimum(demand_hat, init_qty_arr)
            exp_waste_ne = np.maximum(init_qty_arr - exp_sold_ne, 0)
            exp_waste_sm = spoil * waste_frac * init_qty_arr
            exp_waste = np.where(near_expiry_arr, exp_waste_ne, exp_waste_sm)
            exp_sold = np.where(
                near_expiry_arr,
                exp_sold_ne,
                np.minimum(demand_hat, np.maximum(init_qty_arr - exp_waste_sm, 0)),
            )
            profit = (cand_price - cost_arr) * exp_sold - cost_arr * exp_waste
            profit = np.where(margin_ok, profit, -np.inf)
            improved = profit > best_profit
            best_profit = np.where(improved, profit, best_profit)
            best_d = np.where(improved, d, best_d)
        discount_arr = best_d
        new_prices = base_price_arr * (1 - discount_arr)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Final pass to compute KPIs at the chosen prices
    X_fin = X_base.copy()
    X_fin["selling_price"] = new_prices
    X_fin["discount_pct"] = discount_arr
    spoil_fin = spoilage_model.predict_proba(X_fin.values)[:, 1]
    demand_fin = units_arr * (new_prices / ref_arr) ** elas_arr_final
    exp_sold_ne_fin = np.minimum(demand_fin, init_qty_arr)
    exp_waste_ne_fin = np.maximum(init_qty_arr - exp_sold_ne_fin, 0)
    exp_waste_sm_fin = spoil_fin * waste_frac * init_qty_arr
    exp_waste_fin = np.where(near_expiry_arr, exp_waste_ne_fin, exp_waste_sm_fin)
    exp_sold_fin = np.where(
        near_expiry_arr,
        exp_sold_ne_fin,
        np.minimum(demand_fin, np.maximum(init_qty_arr - exp_waste_sm_fin, 0)),
    )
    revenue_fin = new_prices * exp_sold_fin
    profit_fin = (new_prices - cost_arr) * exp_sold_fin - cost_arr * exp_waste_fin
    sell_through = np.where(init_qty_arr > 0, exp_sold_fin / init_qty_arr, 0)

    return pd.DataFrame({
        "strategy":          strategy,
        "category":          df["category"].values,
        "date":              df["transaction_date"].values,
        "days_until_expiry": df["days_until_expiry"].values,
        "revenue":           revenue_fin,
        "profit":            profit_fin,
        "exp_waste":         exp_waste_fin,
        "init_qty":          init_qty_arr,
        "exp_sold":          exp_sold_fin,
        "sell_through":      sell_through,
        "discount":          discount_arr,
    })
