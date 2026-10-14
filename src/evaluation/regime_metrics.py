from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb


def run_confounder_attribution_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare out-of-sample predictive skill on tropical deep-convective OLR anomalies
    across three nested feature sets under blocked temporal validation:
    1. Seasonal Harmonics Only
    2. Climate Confounders Only (ENSO Nino 3.4 + SST/Temp + MSLP + Seasonal)
    3. Confounders + Heliospheric GCR Modulation (Oulu/Moscow NM + phi(t) + 27d Solar Wind)
    """
    work = df.copy().dropna(subset=["wpwp_olr_wm2", "nm_oulu_pct_anom", "nino34_anom"])
    doy = work.index.dayofyear.to_numpy()
    work["sin_ann"] = np.sin(2 * np.pi * doy / 365.25)
    work["cos_ann"] = np.cos(2 * np.pi * doy / 365.25)
    work["sin_semi"] = np.sin(4 * np.pi * doy / 365.25)
    work["cos_semi"] = np.cos(4 * np.pi * doy / 365.25)

    # Target: 27-day smoothed tropical convective OLR anomaly (W/m^2) in late boreal summer / convective season
    target = work["wpwp_olr_wm2"].rolling(14, min_periods=5).mean().bfill()

    seasonal_cols = ["sin_ann", "cos_ann", "sin_semi", "cos_semi"]
    confounder_cols = seasonal_cols + [
        "nino34_anom",
        "wpwp_temp_c",
        "wpwp_mslp_hpa",
        "bob_mslp_hpa",
    ]
    work["gcr_27d"] = work["nm_oulu_pct_anom"].rolling(27, min_periods=5).mean().bfill()
    work["gcr_lag7d"] = work["nm_oulu_pct_anom"].shift(7).bfill()
    work["sw_vb_27d"] = work["vb_product"].rolling(27, min_periods=5).mean().bfill()
    full_cols = confounder_cols + ["nm_oulu_pct_anom", "gcr_27d", "gcr_lag7d", "sw_vb_27d", "f10_7_index"]

    # Blocked split: Train on SC23 + SC25, Evaluate on held-out SC24 (2009-2019)
    test_mask = (work.index >= "2009-01-01") & (work.index <= "2019-11-30")
    train_mask = ~test_mask

    rows = []
    for name, cols in [
        ("1. Seasonal Climatology Baseline", seasonal_cols),
        ("2. Internal Climate Confounders (ENSO + MSLP + Temp)", confounder_cols),
        ("3. Confounders + GCR & Heliospheric Modulation", full_cols),
    ]:
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=30, random_state=42, verbose=-1
        )
        X_tr = work.loc[train_mask, cols].ffill().bfill()
        y_tr = target.loc[train_mask]
        X_te = work.loc[test_mask, cols].ffill().bfill()
        y_te = target.loc[test_mask]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rows.append({
            "Attribution_Model": name,
            "Out_of_Block_R2": round(float(r2_score(y_te, preds)), 4),
            "Out_of_Block_MAE_Wm2": round(float(mean_absolute_error(y_te, preds)), 4),
        })

    out_df = pd.DataFrame(rows)
    conf_mae = out_df.loc[1, "Out_of_Block_MAE_Wm2"]
    out_df["MAE_Skill_Gain_vs_Confounders_Pct"] = (
        (conf_mae - out_df["Out_of_Block_MAE_Wm2"]) / conf_mae * 100.0
    ).round(2)
    return out_df