"""
Stage 2 — Price elasticity from Dunnhumby with out-of-sample validation.

Estimates department-level log-log elasticity with store + week fixed effects,
then validates by re-estimating on weeks 1-80 and projecting onto weeks 81-102.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


TRAIN_WEEKS_DEFAULT = 80
DEPARTMENTS = ["PRODUCE", "MEAT", "PASTRY", "GROCERY", "MEAT-PCKGD"]


def load_dunnhumby_elasticity_table() -> pd.DataFrame:
    """Aggregate Dunnhumby transaction_data → product/store/week shelf-price table."""
    if not config.DUNN_DIR.exists():
        raise FileNotFoundError(
            f"Dunnhumby CSV folder not found at {config.DUNN_DIR}.\n"
            f"Place transaction_data.csv and product.csv inside it."
        )
    tx = pd.read_csv(
        config.DUNN_DIR / "transaction_data.csv",
        usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO",
                 "QUANTITY", "SALES_VALUE", "RETAIL_DISC", "COUPON_MATCH_DISC"],
    )
    tx = tx[(tx["QUANTITY"] > 0) & (tx["SALES_VALUE"] > 0)].copy()
    tx["regular_price"] = (
        (tx["SALES_VALUE"] - tx["RETAIL_DISC"] - tx["COUPON_MATCH_DISC"]) / tx["QUANTITY"]
    )

    agg_tx = (
        tx.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
        .agg(total_units=("QUANTITY", "sum"),
             avg_price=("regular_price", "mean"))
        .reset_index()
    )

    prod_df = pd.read_csv(config.DUNN_DIR / "product.csv",
                          usecols=["PRODUCT_ID", "DEPARTMENT"])
    elas = agg_tx.merge(prod_df, on="PRODUCT_ID", how="left")
    elas = elas[(elas["avg_price"] > 0) & (elas["total_units"] > 0)].copy()
    elas["log_units"] = np.log(elas["total_units"])
    elas["log_price"] = np.log(elas["avg_price"])
    return elas


def oos_validate_elasticity(df: pd.DataFrame, dept: str,
                             train_wks: int = TRAIN_WEEKS_DEFAULT):
    """Estimate elasticity on train weeks; compute OOS R² on held-out weeks."""
    sub = df[df["DEPARTMENT"] == dept].copy()
    train = sub[sub["WEEK_NO"] <= train_wks]
    test = sub[sub["WEEK_NO"] > train_wks]
    if len(train) < 200 or len(test) < 50:
        return None

    tr = train.copy()
    tr["STORE_ID"] = tr["STORE_ID"].astype("category")
    tr["WEEK_NO"] = tr["WEEK_NO"].astype("category")
    X_tr = sm.add_constant(
        pd.get_dummies(tr[["log_price", "STORE_ID", "WEEK_NO"]],
                       drop_first=True).astype(float)
    )
    model = sm.OLS(tr["log_units"], X_tr).fit()
    elast = float(model.params["log_price"])

    # OOS prediction uses intercept + price coefficient only (FE unknown future)
    pred_oos = model.params["const"] + elast * test["log_price"]
    ss_res = float(((test["log_units"] - pred_oos) ** 2).sum())
    ss_tot = float(((test["log_units"] - test["log_units"].mean()) ** 2).sum())
    r2_oos = 1 - ss_res / ss_tot

    return {
        "Department":  dept,
        "Elasticity":  round(elast, 4),
        "N_Train":     len(train),
        "N_Test":      len(test),
        "InSample_R2": round(model.rsquared, 4),
        "OOS_R2":      round(r2_oos, 4),
    }


def run_elasticity_validation(elas_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run OOS validation across all configured departments. Returns a DataFrame."""
    if elas_df is None:
        elas_df = load_dunnhumby_elasticity_table()
    rows = []
    for dept in DEPARTMENTS:
        res = oos_validate_elasticity(elas_df, dept)
        if res:
            rows.append(res)
    return pd.DataFrame(rows)


def elasticity_map_from_validation(oos_df: pd.DataFrame) -> dict:
    """Convert validation table → {department: elasticity} for the optimizer."""
    return {row["Department"]: row["Elasticity"] for _, row in oos_df.iterrows()}
