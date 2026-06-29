"""
Central configuration: paths, constants, and modeling hyperparameters.

All other modules import from here. To change a path or threshold,
change it once here.
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
PERISHABLE_CSV = DATA_DIR / "perishable_goods_management.csv"
DUNN_DIR = DATA_DIR / "dunnhumby"

# Where every CSV/figure produced by the pipeline lands
OUTPUTS_DIR = ROOT / "outputs"

# Where models are pickled (so the dashboard can re-score without retraining)
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Data scope
# ─────────────────────────────────────────────────────────────────────────────
# Categories excluded per project design (different shelf-life dynamics)
EXCLUDE_CATS = ["Pharmaceuticals", "Frozen_Meals"]

# ─────────────────────────────────────────────────────────────────────────────
# 3-way temporal split
# ─────────────────────────────────────────────────────────────────────────────
HOLDOUT_DAYS = 60   # last N days — sacred, never touched until Stage 4
VAL_DAYS = 30       # 30 days before holdout — model selection only

# ─────────────────────────────────────────────────────────────────────────────
# Demand forecasting features
# ─────────────────────────────────────────────────────────────────────────────
DEMAND_FEAT_COLS = [
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_std_7",
    "day_of_week", "month", "day_of_month", "is_weekend",
    "cat_code",
    "avg_price", "avg_discount", "pct_promoted",
]

# ─────────────────────────────────────────────────────────────────────────────
# Spoilage classifier features
# ─────────────────────────────────────────────────────────────────────────────
SPOILAGE_FEATURES = [
    "days_until_expiry", "selling_price", "discount_pct", "initial_quantity",
    "is_promoted", "spoilage_sensitivity", "shelf_life_days", "daily_demand",
    "demand_variability", "distribution_hours", "storage_temp", "temp_deviation",
    "quality_grade",
]

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
MIN_MARGIN = 0.05            # price must be >= cost * (1 + MIN_MARGIN)
NEAR_EXPIRY_DAYS = 5         # days_until_expiry threshold for markdown mode
NEAR_EXPIRY_ELAS = -2.0      # bargain-hunter elasticity for near-expiry items

# ─────────────────────────────────────────────────────────────────────────────
# Default elasticities (pre-computed from Dunnhumby with store + week FE)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ELASTICITY_MAP = {
    "PRODUCE":    -0.2279,
    "MEAT":       -0.0584,
    "PASTRY":     -0.2392,
    "GROCERY":    -0.1722,
    "MEAT-PCKGD": -0.1291,
}

# Map perishable-dataset category → Dunnhumby department
CAT_TO_DEPT = {
    "Produce":      "PRODUCE",
    "Meat":         "MEAT",
    "Bakery":       "PASTRY",
    "Seafood":      "MEAT-PCKGD",
    "Deli":         "GROCERY",
    "Dairy":        "GROCERY",
    "Ready_to_Eat": "GROCERY",
    "Beverages":    "GROCERY",
}

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create outputs/ and models/ if they don't exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
