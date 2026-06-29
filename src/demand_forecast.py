"""
Stage 1 — Short-term demand forecasting.

Trains and compares LightGBM / Prophet / Holt-Winters on a 30-day validation
window. Selects winner by directional accuracy (RMSE tiebreak), then retrains
the winner on train+val and produces 60-day holdout forecasts.

Public functions:
- evaluate_lightgbm(...) → val metrics + cv table
- evaluate_prophet(...)  → val metrics
- evaluate_holtwinters(...) → val metrics
- select_winner(...)     → ("LightGBM" | "Prophet" | "Holt-Winters", comparison_df)
- forecast_holdout(...)  → dict[(date, category)] -> predicted units
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

import config

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────
def _lgb_model() -> "lgb.LGBMRegressor":
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_STATE, verbose=-1,
    )


def evaluate_lightgbm(X_train, y_train, X_val, y_val, feat_val) -> dict:
    """Run 5-fold TS-CV + validation evaluation. Returns metrics dict."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]
        m = _lgb_model()
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        y_hat = np.maximum(m.predict(X_te), 0)
        cv_results.append({
            "fold": fold + 1,
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_hat))),
            "MAE":  float(mean_absolute_error(y_te, y_hat)),
        })

    # Train on full training window → evaluate on validation window
    m_val = _lgb_model()
    m_val.fit(X_train, y_train)
    val_preds = np.maximum(m_val.predict(X_val), 0)

    rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    mae = float(mean_absolute_error(y_val, val_preds))
    dir_acc = float(np.mean(
        np.sign(np.diff(y_val.values)) == np.sign(np.diff(val_preds))
    ) * 100)

    feat_val_lgb = feat_val.copy()
    feat_val_lgb["lgb_pred"] = val_preds
    cat_rmse = (
        feat_val_lgb
        .assign(sq_err=(feat_val_lgb["y"] - feat_val_lgb["lgb_pred"]) ** 2)
        .groupby("category")["sq_err"].mean().apply(np.sqrt)
        .rename("LightGBM_RMSE").round(2)
    )

    return {
        "name": "LightGBM",
        "val_rmse": round(rmse, 2),
        "val_mae": round(mae, 2),
        "dir_acc": round(dir_acc, 2),
        "cv_results": pd.DataFrame(cv_results).round(2),
        "cat_rmse": cat_rmse,
        "val_preds": val_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_prophet(ts_train, val_dates, holdout_dates, feat_val) -> dict:
    from prophet import Prophet
    val_list = []
    for cat in ts_train.columns:
        train_ser = ts_train[cat].reset_index()
        train_ser.columns = ["ds", "y"]
        m = Prophet(
            weekly_seasonality=True, yearly_seasonality=True,
            daily_seasonality=False, seasonality_mode="multiplicative",
        )
        m.fit(train_ser)
        periods_needed = len(val_dates) + len(holdout_dates)
        future = m.make_future_dataframe(periods=periods_needed)
        fc = m.predict(future)
        fc_val = fc[fc["ds"].isin(val_dates)][["ds", "yhat"]].copy()
        fc_val["yhat"] = fc_val["yhat"].clip(lower=0)
        fc_val["category"] = cat
        val_list.append(fc_val)

    val_df = (pd.concat(val_list, ignore_index=True)
                .rename(columns={"ds": "date", "yhat": "prophet_pred"}))
    val_eval = feat_val[["date", "category", "y"]].merge(val_df, on=["date", "category"])

    rmse = float(np.sqrt(mean_squared_error(val_eval["y"], val_eval["prophet_pred"])))
    mae = float(mean_absolute_error(val_eval["y"], val_eval["prophet_pred"]))

    aligned = feat_val.merge(val_df, on=["date", "category"], how="left")["prophet_pred"].values
    dir_acc = float(np.mean(
        np.sign(np.diff(feat_val["y"].values)) == np.sign(np.diff(aligned))
    ) * 100)

    cat_rmse = (
        val_eval
        .assign(sq_err=(val_eval["y"] - val_eval["prophet_pred"]) ** 2)
        .groupby("category")["sq_err"].mean().apply(np.sqrt)
        .rename("Prophet_RMSE").round(2)
    )

    return {
        "name": "Prophet",
        "val_rmse": round(rmse, 2),
        "val_mae": round(mae, 2),
        "dir_acc": round(dir_acc, 2),
        "cat_rmse": cat_rmse,
        "val_df": val_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Holt-Winters
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_holtwinters(ts_train, val_dates, feat_val) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    hw_list = []
    for cat in ts_train.columns:
        train_ser = ts_train[cat]
        try:
            model = ExponentialSmoothing(
                train_ser, seasonal_periods=7,
                trend="add", seasonal="add",
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            preds = fitted.forecast(len(val_dates)).clip(lower=0).values
        except Exception:
            preds = np.full(len(val_dates), train_ser.iloc[-7:].mean())
        hw_list.append(pd.DataFrame({
            "date": val_dates.values,
            "category": cat,
            "hw_pred": preds,
        }))

    hw_df = pd.concat(hw_list, ignore_index=True)
    hw_eval = feat_val[["date", "category", "y"]].merge(hw_df, on=["date", "category"])

    rmse = float(np.sqrt(mean_squared_error(hw_eval["y"], hw_eval["hw_pred"])))
    mae = float(mean_absolute_error(hw_eval["y"], hw_eval["hw_pred"]))

    aligned = feat_val.merge(hw_df, on=["date", "category"], how="left")["hw_pred"].values
    dir_acc = float(np.mean(
        np.sign(np.diff(feat_val["y"].values)) == np.sign(np.diff(aligned))
    ) * 100)

    cat_rmse = (
        hw_eval
        .assign(sq_err=(hw_eval["y"] - hw_eval["hw_pred"]) ** 2)
        .groupby("category")["sq_err"].mean().apply(np.sqrt)
        .rename("HoltWinters_RMSE").round(2)
    )

    return {
        "name": "Holt-Winters",
        "val_rmse": round(rmse, 2),
        "val_mae": round(mae, 2),
        "dir_acc": round(dir_acc, 2),
        "cat_rmse": cat_rmse,
        "val_df": hw_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────
def select_winner(lgb_res: dict, pro_res: dict, hw_res: dict) -> Tuple[str, pd.DataFrame]:
    """
    Pick the model with the highest directional accuracy.
    If multiple models are within 1pp of the best, break tie on RMSE.
    """
    comparison = pd.DataFrame({
        "Model":              [lgb_res["name"], pro_res["name"], hw_res["name"]],
        "Val_RMSE":           [lgb_res["val_rmse"], pro_res["val_rmse"], hw_res["val_rmse"]],
        "Val_MAE":            [lgb_res["val_mae"],  pro_res["val_mae"],  hw_res["val_mae"]],
        "Dir_Accuracy_Pct":   [lgb_res["dir_acc"],  pro_res["dir_acc"],  hw_res["dir_acc"]],
    })
    best_dir = comparison["Dir_Accuracy_Pct"].max()
    close = comparison[abs(comparison["Dir_Accuracy_Pct"] - best_dir) <= 1.0]
    if len(close) > 1:
        winner = close.loc[close["Val_RMSE"].idxmin(), "Model"]
    else:
        winner = comparison.loc[comparison["Dir_Accuracy_Pct"].idxmax(), "Model"]
    return winner, comparison


# ─────────────────────────────────────────────────────────────────────────────
# Holdout forecasting (after winner is chosen, retrain on train+val)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_holdout(winner: str, ts_pivot: pd.DataFrame,
                     feat_train: pd.DataFrame, feat_val: pd.DataFrame,
                     feat_holdout: pd.DataFrame, holdout_dates) -> Dict[Tuple, float]:
    """
    Retrain the winner on train+val and produce holdout forecasts.

    Returns a dict keyed by (date, category) → predicted units.
    NB: forecasts are at category level. Caller must scale to SKU level
    (see optimizer.scale_forecasts_to_sku).
    """
    ts_trainval = ts_pivot[~ts_pivot.index.isin(holdout_dates)]
    feat_trainval = pd.concat([feat_train, feat_val]).sort_values("date").reset_index(drop=True)
    X_trainval = feat_trainval[config.DEMAND_FEAT_COLS]
    y_trainval = feat_trainval["y"]
    X_hold = feat_holdout[config.DEMAND_FEAT_COLS]

    if winner == "LightGBM":
        m = _lgb_model()
        m.fit(X_trainval, y_trainval)
        preds = np.maximum(m.predict(X_hold), 0)
        hold_df = feat_holdout.copy()
        hold_df["winner_pred"] = preds

    elif winner == "Prophet":
        from prophet import Prophet
        rows = []
        for cat in ts_pivot.columns:
            tv_ser = ts_trainval[cat].reset_index()
            tv_ser.columns = ["ds", "y"]
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        daily_seasonality=False, seasonality_mode="multiplicative")
            m.fit(tv_ser)
            future = m.make_future_dataframe(periods=len(holdout_dates))
            fc = m.predict(future)
            fc_h = fc[fc["ds"].isin(holdout_dates)][["ds", "yhat"]].copy()
            fc_h["yhat"] = fc_h["yhat"].clip(lower=0)
            fc_h["category"] = cat
            rows.append(fc_h)
        hold_df = feat_holdout.merge(
            pd.concat(rows, ignore_index=True)
              .rename(columns={"ds": "date", "yhat": "winner_pred"}),
            on=["date", "category"], how="left",
        )

    else:  # Holt-Winters
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        rows = []
        for cat in ts_pivot.columns:
            tv_ser = ts_trainval[cat]
            try:
                model = ExponentialSmoothing(
                    tv_ser, seasonal_periods=7,
                    trend="add", seasonal="add",
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True)
                preds = fitted.forecast(len(holdout_dates)).clip(lower=0).values
            except Exception:
                preds = np.full(len(holdout_dates), tv_ser.iloc[-7:].mean())
            rows.append(pd.DataFrame({
                "date": holdout_dates.values,
                "category": cat,
                "winner_pred": preds,
            }))
        hold_df = feat_holdout.merge(pd.concat(rows, ignore_index=True),
                                     on=["date", "category"], how="left")

    return {
        (row["date"], row["category"]): float(row["winner_pred"])
        for _, row in hold_df.iterrows()
    }
