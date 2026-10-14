from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb


FEATURE_COLS = [
    "sw_speed_kms",
    "b_scalar_nt",
    "bz_gsm_nt",
    "sigma_b_nt",
    "proton_density",
    "flow_pressure_npa",
    "ey_convective_mvm",
    "vb_product",
    "sunspot_r",
    "f10_7_index",
    "dst_index_nt",
]


def engineer_heliospheric_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct causal (past-only) rolling heliospheric memory features corresponding
    to transit across the heliosheath (7d, 27d Bartels rotation, 81d, and 365d).
    """
    feat = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col in df.columns:
            s = df[col].ffill().bfill()
            feat[col] = s
            feat[f"{col}_roll7d"] = s.rolling(7, min_periods=1).mean()
            feat[f"{col}_roll27d"] = s.rolling(27, min_periods=1).mean()
            feat[f"{col}_roll81d"] = s.rolling(81, min_periods=1).mean()
            feat[f"{col}_roll365d"] = s.rolling(365, min_periods=1).mean()

    feat["vb_std_7d"] = df["vb_product"].ffill().bfill().rolling(7, min_periods=1).std().fillna(0.0)
    return feat


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    fd_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute overall and rare-event (Forbush Decrease) metrics."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    picp = np.nan
    if y_lower is not None and y_upper is not None:
        picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100.0)

    fd_recovery = np.nan
    if fd_mask is not None and np.sum(fd_mask) > 0:
        true_anom = np.abs(y_true[fd_mask] - np.median(y_true))
        pred_anom = np.abs(y_pred[fd_mask] - np.median(y_true))
        fd_recovery = float(np.clip(np.mean(pred_anom / (true_anom + 1e-6)) * 100.0, 0.0, 100.0))

    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "FD_Recovery_Pct": round(fd_recovery, 2),
        "PICP_95_Pct": round(picp, 2),
    }


def run_blocked_benchmark(df: pd.DataFrame, target_col: str = "nm_oulu_pct_anom") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute strict Solar-Cycle Blocked Cross-Validation comparing:
    1. Naive Climatology / 27-day Persistence
    2. Huber Robust Linear Baseline
    3. Random Forest Regressor
    4. Hybrid LightGBM + Heteroscedastic Residual Uncertainty Model
    """
    from src.models.blocked_cv import SolarCycleBlockedSplitter, detect_forbush_decrease_events

    X_all = engineer_heliospheric_lag_features(df)
    y_all = df[target_col].interpolate(method="time").bfill().ffill()

    fd_events = detect_forbush_decrease_events(y_all)
    fd_dates = set()
    for _, row in fd_events.iterrows():
        for d in pd.date_range(row["onset_date"], row["trough_date"] + pd.Timedelta(days=3)):
            fd_dates.add(d)
    fd_mask_all = np.array([dt in fd_dates for dt in df.index])

    splitter = SolarCycleBlockedSplitter(purge_days=54)
    # Use Solar Cycle 24 (2008-2019) as primary out-of-sample test block
    train_idx, test_idx, block_name = None, None, None
    for tr, te, bname in splitter.split(df):
        if bname == "SC24_Full":
            train_idx, test_idx, block_name = tr, te, bname
            break

    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx].to_numpy(), y_all.iloc[test_idx].to_numpy()
    fd_test_mask = fd_mask_all[test_idx]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    results = []
    preds_df = pd.DataFrame(index=df.index[test_idx])
    preds_df["y_true"] = y_test
    preds_df["is_forbush_event"] = fd_test_mask

    # 1. Climatology / 1-Year Lagged Solar-Cycle Persistence Baseline
    y_pers = y_all.shift(365).bfill().iloc[test_idx].to_numpy()
    m_pers = evaluate_predictions(y_test, y_pers, fd_mask=fd_test_mask)
    m_pers["Model"] = "Climatology / Cycle Persistence"
    results.append(m_pers)
    preds_df["pred_persistence"] = y_pers

    # 2. Huber Robust Linear Regression
    huber = HuberRegressor(max_iter=500, alpha=1.35)
    huber.fit(X_tr_s, y_train)
    y_huber = huber.predict(X_te_s)
    res_std_huber = np.std(y_train - huber.predict(X_tr_s))
    m_huber = evaluate_predictions(
        y_test, y_huber, y_huber - 1.96 * res_std_huber, y_huber + 1.96 * res_std_huber, fd_mask=fd_test_mask
    )
    m_huber["Model"] = "Huber Robust Linear Baseline"
    results.append(m_huber)
    preds_df["pred_huber"] = y_huber

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=250, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_train)
    y_rf = rf.predict(X_te_s)
    tree_preds = np.stack([t.predict(X_te_s) for t in rf.estimators_], axis=0)
    rf_std = np.std(tree_preds, axis=0) + 0.5 * np.std(y_train - rf.predict(X_tr_s))
    m_rf = evaluate_predictions(
        y_test, y_rf, y_rf - 1.96 * rf_std, y_rf + 1.96 * rf_std, fd_mask=fd_test_mask
    )
    m_rf["Model"] = "Random Forest (Blocked CV)"
    results.append(m_rf)
    preds_df["pred_rf"] = y_rf

    # 4. Hybrid LightGBM + Heteroscedastic Variance Estimator
    lgbm_mean = lgb.LGBMRegressor(
        n_estimators=350, learning_rate=0.03, num_leaves=31, subsample_for_bin=5000, random_state=42, verbose=-1
    )
    lgbm_mean.fit(X_tr_s, y_train)
    y_lgb = lgbm_mean.predict(X_te_s)

    # Fit secondary heteroscedastic log-variance model on squared training residuals
    train_sq_err = np.log((y_train - lgbm_mean.predict(X_tr_s)) ** 2 + 1e-4)
    lgbm_var = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, num_leaves=15, random_state=42, verbose=-1)
    lgbm_var.fit(X_tr_s, train_sq_err)
    het_std = np.sqrt(np.exp(lgbm_var.predict(X_te_s))) * 1.35

    m_lgb = evaluate_predictions(
        y_test, y_lgb, y_lgb - 1.96 * het_std, y_lgb + 1.96 * het_std, fd_mask=fd_test_mask
    )
    m_lgb["Model"] = "Hybrid LightGBM + Heteroscedastic GP/Var"
    results.append(m_lgb)

    preds_df["pred_hybrid"] = y_lgb
    preds_df["hybrid_lower95"] = y_lgb - 1.96 * het_std
    preds_df["hybrid_upper95"] = y_lgb + 1.96 * het_std

    res_table = pd.DataFrame(results)[["Model", "R2", "RMSE", "MAE", "FD_Recovery_Pct", "PICP_95_Pct"]]
    rmse_base = res_table.loc[0, "RMSE"]
    res_table["RMSE_Reduction_vs_Base_Pct"] = ((rmse_base - res_table["RMSE"]) / rmse_base * 100.0).round(2)
    return res_table, preds_df