# Eddy Covariance Data Processing with hesseflux (Python)

## Problem: `hesseflux` NumPy Deprecation Errors

Initially, when attempting to process Eddy Covariance data using the `hesseflux` Python package, the script encountered `AttributeError` related to deprecated NumPy aliases (`np.bool`, `np.float`, `np.int`). This indicated an incompatibility between the installed `hesseflux` version (5.0 from PyPI) and the current NumPy version in the Python environment.

**Initial Error Messages Observed:**
*   `module 'numpy' has no attribute 'bool'.`
*   `module 'numpy' has no attribute 'float'.`
*   `module 'numpy' has no attribute 'int'.`

## Resolution: Installing `hesseflux` from GitHub and Patching

The `hesseflux` GitHub repository was found to contain a more recent version (5.1.dev10) that had already addressed the `np.bool`, `np.float`, and `np.int` deprecation issues. The resolution involved:

1.  **Uninstalling the old `hesseflux` package:** The previously installed version (5.0) was uninstalled.
2.  **Cloning the `hesseflux` GitHub repository:** The repository was cloned to `/Users/habibw/Documents/Gurteen/hesseflux_repo`.
3.  **Installing `hesseflux` from the cloned repository:** The package was installed directly from the local clone, ensuring the patched version was used.

**Subsequent Issue: `KeyError` during Gap-Filling**

After resolving the NumPy deprecation errors, a `KeyError` was encountered during the `gap-filling` step, specifically for the partitioned flux columns (`GPP_reichstein`, `RECO_reichstein`, `GPP_lasslop`, `RECO_lasslop`). This occurred because the `dff` (flag) DataFrame, which mirrors the data DataFrame, was not being updated with these new columns after flux partitioning.

**Resolution for `KeyError`:**

The `process_eddy_data.py` script was modified to explicitly update the `dff` DataFrame with the newly created partitioned flux columns immediately after they are concatenated to the main `df` DataFrame. These new columns in `dff` were initialized with `0` (indicating good quality) or `2` (if the corresponding data in `df` was `undef_val`).

## Processing Script: `process_eddy_data.py`

This script handles reading the raw Eddy Covariance data, performing necessary pre-processing (like VPD calculation and NaN handling), and then applying `hesseflux`'s core processing steps (spike detection, u* filtering, flux partitioning, and gap-filling).

**Location:** `/Users/habibw/Documents/Gurteen/process_eddy_data.py`

**Key Features:**
*   Processes data for multiple tower locations (`Gurteen`, `Athenry`, `JC1`, `JC2`, `Timoleague`).
*   Reads `YYYY.csv` files from each tower's directory.
*   Converts `DateTime` column to datetime objects and sets it as the DataFrame index.
*   Replaces `-9999` and `-10272.15` with `NaN`.
*   Calculates `VPD` if not present, using `RH` and `air_temp_c`.
*   Renames columns to match `hesseflux`'s expected conventions (`SWIN_1_1_1` to `SW_IN`, `air_temp_c` to `TA`, `u*` to `USTAR`, and quality control flags). **Note:** `PPFD` column is retained as `PPFD` for Light Response Curve calculations.
*   Initializes and updates a flag DataFrame (`dff`) to track data quality throughout the processing.
*   Applies `hesseflux.madspikes` for spike detection.
*   Applies `hesseflux.ustarfilter` for u* filtering.
*   Applies `hesseflux.nee2gpp` for flux partitioning (Reichstein and Lasslop methods).
*   Applies `hesseflux.gapfill` for gap-filling.
*   Saves the processed data to `processed_hesseflux_YYYY.csv` files within each tower's directory.
*   Generates and saves plots (time series, diurnal cycles, and diurnal-seasonal fingerprint plots) for key variables in the `plot` subdirectory within each tower's directory.
*   Generates small multiples fingerprint plots for NEE (raw and filled) across both years, with Hour of Day on the Y-axis and Day of Year on the X-axis, with improved layout and colorbar positioning.
*   Generates seasonal Light Response Curve (LRC) plots for raw and filled NEE data, using the provided Mitscherlich function and `PPFD` as the light variable. These plots are saved in the `lrc_plots` subdirectory within each tower's `plot` folder, adhering to publication quality standards. Each subplot displays fitted parameters (a, b, c) and R-squared values.

**Usage:**
```bash
python3 /Users/habibw/Documents/Gurteen/process_eddy_data.py
```

## Lullymore Data Processing: `process_lullymore_data.py`

This script processes Eddy Covariance data for the Lullymore tower.

**Location:** `/Users/habibw/Documents/Lullymore/process_lullymore_data.py`

**Key Features:**
*   Reads `LM_Flux_data_2022_2023.csv` which contains data for multiple years (2022, 2023).
*   Handles the `﻿DateTime` column (with BOM character) and renames it to `DateTime`.
*   Renames `co2_flux_filled` to `NEE`, `ppfd` to `PPFD`, `u*` to `USTAR`, `LE_filled` to `LE`, and `H_filled` to `H`.
*   Initializes missing columns (`RH`, `air_temp_c`, `SW_IN`, `TA`, `NEE_QC`, `LE_QC`, `H_QC`) with `NaN` or `0` as appropriate.
*   Applies full `hesseflux` processing steps: spike detection, u* filtering, flux partitioning (Reichstein and Lasslop methods), and gap-filling.
*   Saves processed data to `processed_hesseflux_YYYY.csv` files within the Lullymore directory.
*   Generates and saves all associated plots (time series, diurnal cycles, diurnal-seasonal fingerprint, and seasonal LRC plots) in the `plot` subdirectory.

**Usage:**
```bash
python3 /Users/habibw/Documents/Lullymore/process_lullymore_data.py
```

## Clarabog Data Processing: `process_clarabog_data.py`

This script processes Eddy Covariance data for the Clarabog tower.

**Location:** `/Users/habibw/Documents/Clarabog/process_clarabog_data.py`

**Key Features:**
*   Reads yearly `clara_YYYY.csv` files (e.g., `clara_2018.csv`, `clara_2019.csv`, etc.).
*   Renames columns to match `hesseflux`'s expected conventions:
    *   `Net Ecosystem Exchange` to `NEE`
    *   `Latent heat Flux` to `LE`
    *   `Sensible Heat Flux` to `H`
    *   `Photosynthetic Photon Flux Density` to `PPFD`
    *   `Net Radiation` to `SW_IN`
    *   `Air Temperature` to `TA`
    *   `Relative Humidity` to `RH`
    *   `U star` to `USTAR`
*   Initializes missing quality control columns (`NEE_QC`, `LE_QC`, `H_QC`) with `0`.
*   Calculates `VPD` if `RH` and `TA` are present.
*   Applies full `hesseflux` processing steps: spike detection, u* filtering, flux partitioning (Reichstein and Lasslop methods), and gap-filling.
*   Saves processed data to `processed_hesseflux_YYYY.csv` files within the Clarabog directory.
*   Generates and saves all associated plots (time series, diurnal cycles, diurnal-seasonal fingerprint, and seasonal LRC plots) in the `plot` subdirectory.

**Usage:**
```bash
python3 /Users/habibw/Documents/Clarabog/process_clarabog_data.py
```

## Satellite VI Data Merging: `merge_satellite_vis.py`

This script merges daily Vegetation Index (VI) data for Clarabog and Lullymore into existing combined VI files.

**Location:** `/Users/habibw/Documents/Gurteen/merge_satellite_vis.py`

**Key Features:**
*   Identifies existing MODIS daily VI files (e.g., `modis_evi_daily.csv`).
*   Identifies new daily VI files for Clarabog and Lullymore (e.g., `modis_evi_daily_clarabog_lullymore.csv`).
*   Renames `system:time_start` to `DateTime` in both existing and new VI dataframes for consistency.
*   Converts the `DateTime` column to datetime objects.
*   Melts the dataframes to a long format, ensuring a 'Tower' column for proper merging.
*   Concatenates existing and new VI data, drops duplicates, and sorts by 'Tower' and 'DateTime'.
*   Overwrites the existing combined VI files with the merged data.
*   Deletes the individual Clarabog and Lullymore VI files after successful merging.

**Usage:**
```bash
python3 /Users/habibw/Documents/Gurteen/merge_satellite_vis.py
```

## Consolidated Climatological Dataset

This section describes the creation of a consolidated, half-hourly climatological dataset from all available tower data.

**Location:** `/Users/habibw/Documents/consolidated climatological dataset/consolidate_data.py` and `/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv`

**Key Features:**
*   Loads processed Eddy Covariance data from all tower locations (Gurteen, Athenry, JC1, JC2, Timoleague, Lullymore, Clarabog).
*   Aggregates data to a half-hourly climatology based on Day of Year, Hour, and Minute.
*   The resulting `consolidated_half_hourly_climatology_data.csv` provides a complete, single-year representation of the average annual cycle for all measured variables at each tower, with missing values preserved where no data was available across all years for a given half-hourly slot.

**Usage:**
```bash
python3 /Users/habibw/Documents/consolidated climatological dataset/consolidate_data.py
```

## Weekly Light Response Curve (LRC) Generation

This section describes the generation of weekly LRCs and their parameters from the consolidated climatological dataset.

**Location:** `/Users/habibw/Documents/consolidated climatological dataset/generate_weekly_lrcs.py` and `/Users/habibw/Documents/consolidated climatological dataset/all_towers_weekly_lrc_parameters.csv`

**Key Features:**
*   Loads the `consolidated_half_hourly_climatology_data.csv`.
*   For each tower, generates weekly LRC plots (NEE vs. PPFD) with fitted Mitscherlich curves.
*   Saves individual plots in tower-specific subfolders within the consolidated dataset directory.
*   Generates `all_towers_weekly_lrc_parameters.csv` summarizing LRC parameters (a, b, c) and R² values for each tower and week.

**Usage:**
```bash
python3 /Users/habibw/Documents/consolidated climatological dataset/generate_weekly_lrcs.py
```

## LRC Parameter Correlation Analysis

This section details the analysis of correlations between LRC parameters (a, b, c) and satellite-derived Vegetation Indices (VIs).

**Location:** `/Users/habibw/Documents/consolidated climatological dataset/analyze_lrc_parameter_correlations.py` and `/Users/habibw/Documents/consolidated climatological dataset/all_towers_best_lrc_parameter_correlations.csv`

**Key Features:**
*   Loads the `consolidated_half_hourly_climatology_data.csv` and `all_towers_weekly_lrc_parameters.csv`.
*   Analyzes correlations between LRC parameters (a, b, c) and VIs (EVI, NDMI, NDVI, SAVI) using linear, exponential, logarithmic, and power functions.
*   Identifies the best-fitting VI and correlation type (based on R²) for each LRC parameter for each tower.
*   Generates small multiple plots visualizing these best correlations for each tower.
*   Saves `all_towers_best_lrc_parameter_correlations.csv` summarizing the best fits.

**Usage:**
```bash
python3 /Users/habibw/Documents/consolidated climatological dataset/analyze_lrc_parameter_correlations.py
```

## Machine Learning NEE Upscaling Model

This section describes the development and evaluation of a machine learning model for upscaling NEE using satellite-derived VIs and PPFD.

**Location:** `/Users/habibw/Documents/consolidated climatological dataset/train_upscaling_model.py`

**Key Features:**
*   Loads the `consolidated_half_hourly_climatology_data.csv`.
*   Uses satellite-derived VIs (EVI, NDMI, NDVI, SAVI) and PPFD, along with temporal features (Week of Year), as input features.
*   Splits data into training (5 randomly selected towers) and testing (2 randomly selected towers) sets.
*   Trains and evaluates both **XGBoost** and **Random Forest** Regressor models.
*   Performs hyperparameter tuning for both models using `GridSearchCV`.
*   Generates performance plots (Measured vs. Modeled NEE scatter plots), bar charts comparing average measured vs. modeled NEE for training and testing towers, and feature importance plots.

**Model Performance (after Hyperparameter Tuning):**

*   **XGBoost Model:**
    *   Best Parameters: `learning_rate=0.01`, `max_depth=3`, `n_estimators=200`
    *   Training R²: 0.261, RMSE: 8.673
    *   Testing R²: 0.351, RMSE: 9.009

*   **Random Forest Model:**
    *   Best Parameters: `max_depth=5`, `min_samples_leaf=10`, `n_estimators=50`
    *   Training R²: 0.288, RMSE: 8.514
    *   Testing R²: 0.358, RMSE: 8.960

**Usage:**
```bash
python3 /Users/habibw/Documents/consolidated climatological dataset/train_upscaling_model.py
```

## Future Updates

This `GEMINI.md` file will be updated as further modifications or issues arise during the Eddy Covariance data processing.

## Current Status

We are currently working on improving the deep learning model for NEE upscaling. The current focus is on implementing a hybrid CNN-LSTM model to better capture the complex temporal and feature interactions in the data. The script `train_dl_upscaling_model.py` in the `consolidated climatological dataset` directory is being modified to implement this new architecture. The model is being trained on a Mac with Apple Silicon, and the `tensorflow-metal` package has been installed to enable GPU acceleration.