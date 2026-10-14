from __future__ import annotations
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ingestion.era5_olr_join import build_aligned_dataset
from src.physics.force_field import estimate_modulation_potential_from_nm, kalman_rts_smooth_1d
from src.models.gcr_reconstructor import run_blocked_benchmark
from src.models.red_noise_wavelet import lomb_scargle_red_noise_test, cross_spectral_coherence_red_noise
from src.evaluation.regime_metrics import run_confounder_attribution_experiment


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    Path("notebooks").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: Building Real Aligned Dataset (OMNI2 + NMDB + ERA5 + NOAA ENSO) ===")
    df = build_aligned_dataset("1996-01-01", "2024-12-31")

    print("=== STEP 2: Running 1D Force-Field & Blocked ML Benchmark ===")
    df["phi_oulu_mv"] = estimate_modulation_potential_from_nm(df["nm_oulu_norm"], cutoff_rigidity_gv=0.8)
    bench_table, preds_df = run_blocked_benchmark(df, target_col="nm_oulu_pct_anom")
    bench_table.to_csv("data/processed/model_benchmark_sc24.csv", index=False)
    preds_df.to_parquet("data/processed/sc24_predictions.parquet")
    print("\n--- Blocked CV Benchmark (Held-Out Solar Cycle 24) ---")
    print(bench_table.to_string(index=False))

    print("\n=== STEP 3: Running Lomb-Scargle & Cross-Spectral Red-Noise Tests ===")
    sub_max = df.loc["2000-01-01":"2003-12-31"].copy()
    t_days = np.arange(len(sub_max), dtype=float)
    periods = np.linspace(10.0, 65.0, 300)
    ls_gcr = lomb_scargle_red_noise_test(
        t_days, sub_max["nm_oulu_pct_anom"].to_numpy(), periods, n_surrogates=500
    )
    coh_df = cross_spectral_coherence_red_noise(
        sub_max["nm_oulu_pct_anom"], sub_max["wpwp_olr_wm2"], nperseg=128, n_surrogates=500
    )
    coh_df.to_csv("data/processed/red_noise_coherence_results.csv", index=False)
    rot_band = coh_df[(coh_df["period_days"] >= 25.0) & (coh_df["period_days"] <= 29.5)]
    print("\n--- Solar Rotational Band (25-29.5 Days) Coherence vs AR(1) Red Noise ---")
    print(rot_band.to_string(index=False))

    print("\n=== STEP 4: Running Confounder-Controlled ERA5 & OLR Attribution ===")
    attr_table = run_confounder_attribution_experiment(df)
    attr_table.to_csv("data/processed/confounder_attribution_results.csv", index=False)
    print("\n--- Confounder-Controlled Attribution Results ---")
    print(attr_table.to_string(index=False))

    print("\n=== STEP 5: Generating & Executing All 4 Jupyter Notebooks ===")
    build_and_run_notebooks()
    print("\n[SUCCESS] All data artifacts, figures, and executed notebooks are ready!")


def build_and_run_notebooks() -> None:
    notebooks_spec = {
        "01_data_provenance_and_kalman_qc.ipynb": [
            new_markdown_cell(
                "# 01 | Data Provenance, Missingness Hygiene & State-Space Kalman Smoothing\n"
                "Audits real NASA GSFC OMNI2, NMDB (Oulu & Moscow), NOAA PSL Niño 3.4, and ECMWF ERA5 records (1996–2024)."
            ),
            new_code_cell(
                "import sys; sys.path.append('..')\n"
                "import pandas as pd\nimport matplotlib.pyplot as plt\n"
                "from src.physics.force_field import kalman_rts_smooth_1d\n\n"
                "df = pd.read_parquet('../data/processed/aligned_helio_atmos_daily.parquet')\n"
                "print(f'Aligned Dataset Shape: {df.shape} | Date Range: {df.index.min().date()} to {df.index.max().date()}')\n"
                "df[['sw_speed_kms', 'b_scalar_nt', 'nm_oulu_pct_anom', 'wpwp_olr_wm2', 'nino34_anom']].describe().round(2)"
            ),
            new_code_cell(
                "smoothed, std = kalman_rts_smooth_1d(df['nm_oulu_pct_anom'])\n"
                "fig, ax = plt.subplots(figsize=(12, 4))\n"
                "ax.plot(df.index, df['nm_oulu_pct_anom'], alpha=0.4, label='Raw Oulu NM Anomaly (%)')\n"
                "ax.plot(smoothed.index, smoothed, color='navy', lw=1.2, label='Kalman RTS Smoothed')\n"
                "ax.set_title('Oulu Neutron Monitor (Rc=0.8 GV) Cosmic-Ray Flux Anomaly (1996-2024)')\n"
                "ax.set_ylabel('Relative Count Anomaly (%)')\nax.legend()\nplt.tight_layout()\n"
                "plt.savefig('../figures/fig01_kalman_nmdb.png', dpi=180)\nplt.show()"
            ),
        ],
        "02_force_field_and_forbush_ml.ipynb": [
            new_markdown_cell(
                "# 02 | 1D Force-Field Heliospheric Modulation & Blocked ML Reconstruction\n"
                "Evaluates out-of-sample Solar Cycle 24 reconstruction and Forbush Decrease recovery under strict 54-day purged blocked validation."
            ),
            new_code_cell(
                "import pandas as pd\nimport matplotlib.pyplot as plt\n"
                "bench = pd.read_csv('../data/processed/model_benchmark_sc24.csv')\n"
                "preds = pd.read_parquet('../data/processed/sc24_predictions.parquet')\n"
                "bench"
            ),
            new_code_cell(
                "fig, ax = plt.subplots(figsize=(12, 4.5))\n"
                "ax.plot(preds.index, preds['y_true'], color='black', lw=1.0, label='Observed Oulu NM (%)')\n"
                "ax.plot(preds.index, preds['pred_hybrid'], color='crimson', lw=1.1, label='Hybrid LightGBM + Het-Var (Blocked Test)')\n"
                "ax.fill_between(preds.index, preds['hybrid_lower95'], preds['hybrid_upper95'], color='crimson', alpha=0.2, label='95% Heteroscedastic CI')\n"
                "ax.set_title('Out-of-Block Reconstruction Across Held-Out Solar Cycle 24 (2008-2019)')\n"
                "ax.set_ylabel('GCR Anomaly (%)')\nax.legend()\nplt.tight_layout()\n"
                "plt.savefig('../figures/fig02_sc24_blocked_reconstruction.png', dpi=180)\nplt.show()"
            ),
        ],
        "03_wavelet_coherence_red_noise.ipynb": [
            new_markdown_cell(
                "# 03 | Spectral Periodicity & Cross-Spectral Coherence Under AR(1) Red-Noise Null Testing\n"
                "Tests the ~27-day solar rotational periodicity in GCR flux and tropical convective OLR against Monte Carlo AR(1) red-noise surrogates."
            ),
            new_code_cell(
                "import pandas as pd\nimport matplotlib.pyplot as plt\n"
                "coh = pd.read_csv('../data/processed/red_noise_coherence_results.csv')\n"
                "coh[(coh['period_days'] >= 22) & (coh['period_days'] <= 32)]"
            ),
            new_code_cell(
                "fig, ax = plt.subplots(figsize=(10, 4))\n"
                "sub = coh[coh['period_days'] <= 65]\n"
                "ax.plot(sub['period_days'], sub['coherence'], color='teal', lw=1.8, label='Observed GCR-OLR Coherence')\n"
                "ax.plot(sub['period_days'], sub['red_noise_95'], '--', color='orange', label='95% AR(1) Red-Noise Null')\n"
                "ax.plot(sub['period_days'], sub['red_noise_99'], ':', color='red', label='99% AR(1) Red-Noise Null')\n"
                "ax.axvspan(26.0, 28.5, color='gold', alpha=0.25, label='27-Day Solar Rotation Band')\n"
                "ax.set_xlabel('Period (Days)')\nax.set_ylabel('Magnitude-Squared Coherence')\n"
                "ax.set_title('Cross-Spectral Coherence: Oulu GCR Flux vs. WPWP Convective OLR')\n"
                "ax.legend()\nplt.tight_layout()\n"
                "plt.savefig('../figures/fig03_red_noise_coherence.png', dpi=180)\nplt.show()"
            ),
        ],
        "04_era5_olr_confounder_attribution.ipynb": [
            new_markdown_cell(
                "# 04 | Confounder-Controlled ERA5 & Satellite OLR Attribution\n"
                "Quantifies incremental predictive skill of heliospheric GCR modulation beyond ENSO (Niño 3.4), SST, and seasonal harmonics."
            ),
            new_code_cell(
                "import pandas as pd\n"
                "attr = pd.read_csv('../data/processed/confounder_attribution_results.csv')\n"
                "attr"
            ),
        ],
    }

    for nb_name, cells in notebooks_spec.items():
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb_path = Path("notebooks") / nb_name
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"  -> Executing {nb_path} ...")
        client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
        client.execute()
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)


if __name__ == "__main__":
    main()