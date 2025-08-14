import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import xgboost as xgb # Import XGBoost

# --- Data Loading and Correlation Functions (re-used) ---
def load_and_prepare_satellite_data(data_dir, file_name, value_name):
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, parse_dates=['system:time_start'], infer_datetime_format=True)
    df = df.rename(columns={'system:time_start': 'DateTime'})
    df_melted = df.melt(id_vars=['DateTime'], var_name='Tower', value_name=value_name)
    df_melted['Year'] = df_melted['DateTime'].dt.year
    df_melted['Week'] = df_melted['DateTime'].dt.isocalendar().week.astype(int)
    df_weekly = df_melted.groupby(['Tower', 'Year', 'Week'])[value_name].mean().reset_index()
    return df_weekly

def load_all_satellite_data(satellite_data_dir):
    ppfd_df = load_and_prepare_satellite_data(satellite_data_dir, 'daily_mean_modis_ppfd.csv', 'MODIS_PPFD')
    evi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_evi_daily.csv', 'EVI')
    ndmi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndmi_daily.csv', 'NDMI')
    ndvi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_ndvi_daily.csv', 'NDVI')
    savi_df = load_and_prepare_satellite_data(satellite_data_dir, 'modis_savi_daily.csv', 'SAVI')
    merged_satellite_df = ppfd_df
    for df_vi in [evi_df, ndmi_df, ndvi_df, savi_df]:
        merged_satellite_df = pd.merge(merged_satellite_df, df_vi, on=['Tower', 'Year', 'Week'], how='outer')
    return merged_satellite_df

def load_lrc_parameters(gurteen_dir):
    lrc_2024_path = os.path.join(gurteen_dir, 'lrc_parameters_2024.csv')
    return pd.read_csv(lrc_2024_path) if os.path.exists(lrc_2024_path) else pd.DataFrame()

def load_weekly_environmental_data(base_directories, year):
    all_env_dfs = []
    env_cols = ['TA', 'VPD', 'SW_IN'] # Environmental columns to load

    for tower_name, tower_dir in base_directories.items():
        processed_file_path = os.path.join(tower_dir, f"processed_hesseflux_{year}.csv")
        if os.path.exists(processed_file_path):
            df_ec = pd.read_csv(processed_file_path, index_col='DateTime', parse_dates=True)
            df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)
            df_ec['Year'] = df_ec.index.year # Add Year column
            df_ec['Tower'] = tower_name # Add tower name for merging
            
            # Select relevant environmental columns and calculate weekly mean
            weekly_env_df = df_ec.groupby(['Tower', 'Year', 'Week'])[env_cols].mean().reset_index()
            all_env_dfs.append(weekly_env_df)
    
    return pd.concat(all_env_dfs, ignore_index=True) if all_env_dfs else pd.DataFrame()

def mitscherlich_lrc(ppfd, a, b, c):
    denom = a + b
    # Handle cases where denom is zero or very close to zero
    if isinstance(denom, pd.Series):
        # For pandas Series, use element-wise check and fill with NaN
        result = pd.Series(np.full_like(ppfd, np.nan), index=ppfd.index)
        valid_indices = ~np.isclose(denom, 0)
        if valid_indices.any():
            exp_arg = (-c[valid_indices] * ppfd[valid_indices]) / denom[valid_indices]
            exp_arg = np.clip(exp_arg, -700, 700)
            result[valid_indices] = 1 - denom[valid_indices] * (1 - np.exp(exp_arg)) + b[valid_indices]
        return result
    else:
        # For single values
        if np.isclose(denom, 0): return np.full_like(ppfd, np.nan)
        exp_arg = np.clip((-c * ppfd) / denom, -700, 700)
        return 1 - denom * (1 - np.exp(exp_arg)) + b

def plot_feature_importances(importances_dict, plot_dir):
    print("\n--- Generating Feature Importance Plots ---")
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False) # Adjusted figsize
    fig.suptitle('XGBoost Feature Importances for LRC Parameters', fontsize=16, y=1.02)

    for i, (param, importances) in enumerate(importances_dict.items()):
        ax = axes[i]
        # Sort features by importance
        sorted_idx = np.array(importances['importance']).argsort()
        ax.barh(np.array(importances['feature'])[sorted_idx], np.array(importances['importance'])[sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'LRC {param} Model')
        ax.tick_params(axis='y', labelsize=8) # Adjust y-axis label size
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'xgboost_feature_importances.png'))
    plt.close()
    print(f"Feature importance plot saved to {plot_dir}")

# --- Main Execution Block ---
if __name__ == '__main__':
    satellite_data_dir = "/Users/habibw/Documents/satellite derived data"
    gurteen_dir = "/Users/habibw/Documents/Gurteen"
    base_directories = {
        "Gurteen": "/Users/habibw/Documents/Gurteen", "Athenry": "/Users/habibw/Documents/Athenry",
        "JC1": "/Users/habibw/Documents/JC1", "JC2": "/Users/habibw/Documents/JC2",
        "Timoleague": "/Users/habibw/Documents/Timoleague"
    }

    satellite_data_weekly = load_all_satellite_data(satellite_data_dir)
    lrc_params = load_lrc_parameters(gurteen_dir)
    weekly_env_data = load_weekly_environmental_data(base_directories, 2024) # Load 2024 environmental data
    
    lrc_params_2024 = lrc_params[lrc_params['Year'] == 2024].copy()
    satellite_data_2024 = satellite_data_weekly[satellite_data_weekly['Year'] == 2024].copy()
    lrc_params_successful = lrc_params_2024[lrc_params_2024['Fit_Status'] == 'Success'].copy()
    lrc_params_successful['Week'] = lrc_params_successful['Week'].astype(int)

    # Merge all data sources
    merged_data = pd.merge(lrc_params_successful, satellite_data_2024, on=['Tower', 'Year', 'Week'], how='inner')
    merged_data = pd.merge(merged_data, weekly_env_data, on=['Tower', 'Year', 'Week'], how='inner')

    # --- Feature Engineering: Lagged Variables and Seasonal Indicators ---
    print("--- Performing Feature Engineering ---")
    # Sort data for correct lagging
    merged_data = merged_data.sort_values(by=['Tower', 'Year', 'Week'])

    # Lagged VIs and Environmental variables
    lag_features = ['EVI', 'NDMI', 'NDVI', 'SAVI', 'TA', 'VPD', 'SW_IN', 'MODIS_PPFD']
    for feature in lag_features:
        merged_data[f'{feature}_lag1'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(1)
        merged_data[f'{feature}_lag2'] = merged_data.groupby(['Tower', 'Year'])[feature].shift(2)

    # Seasonal indicators (sine and cosine of week number)
    merged_data['Week_sin'] = np.sin(2 * np.pi * merged_data['Week'] / 52)
    merged_data['Week_cos'] = np.cos(2 * np.pi * merged_data['Week'] / 52)

    # --- New Feature Engineering: Polynomial and Interaction Terms for TA ---
    print("--- Adding Polynomial and Interaction Terms for TA ---")
    # Polynomial terms for TA and its lags
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            merged_data[f'{ta_col}_sq'] = merged_data[ta_col]**2
            merged_data[f'{ta_col}_cub'] = merged_data[ta_col]**3

    # Interaction terms between TA (and its lags) and VIs
    vi_features = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            for vi_col in vi_features:
                if vi_col in merged_data.columns:
                    merged_data[f'{ta_col}_x_{vi_col}'] = merged_data[ta_col] * merged_data[vi_col]

    # Define features for ML model (VIs + Environmental variables + Lagged + Seasonal + Polynomial + Interactions)
    features = ['EVI', 'NDMI', 'NDVI', 'SAVI', 'TA', 'VPD', 'SW_IN', 'MODIS_PPFD',
                'EVI_lag1', 'NDMI_lag1', 'NDVI_lag1', 'SAVI_lag1', 'TA_lag1', 'VPD_lag1', 'SW_IN_lag1', 'MODIS_PPFD_lag1',
                'EVI_lag2', 'NDMI_lag2', 'NDVI_lag2', 'SAVI_lag2', 'TA_lag2', 'VPD_lag2', 'SW_IN_lag2', 'MODIS_PPFD_lag2',
                'Week_sin', 'Week_cos']
    
    # Add newly created polynomial and interaction features to the list
    for ta_col in ['TA', 'TA_lag1', 'TA_lag2']:
        if ta_col in merged_data.columns:
            features.append(f'{ta_col}_sq')
            features.append(f'{ta_col}_cub')
            for vi_col in vi_features:
                if vi_col in merged_data.columns:
                    features.append(f'{ta_col}_x_{vi_col}')

    lrc_parameters = ['a', 'b', 'c']
    ml_models = {}
    feature_importances_data = {}

    # Drop rows with NaN in any relevant column for ML training (including new lagged features)
    ml_data = merged_data.dropna(subset=lrc_parameters + features)

    if ml_data.empty:
        print("Not enough data after dropping NaNs for ML model training. Exiting.")
        exit()

    print("--- Training Machine Learning Models for LRC Parameters with Hyperparameter Tuning (XGBoost with Feature Engineering) ---")
    param_grid = {
        'n_estimators': [50, 100, 200], 
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0] # Subsample ratio of columns when constructing each tree
    }

    for param in lrc_parameters:
        X = ml_data[features]
        y = ml_data[param]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize GridSearchCV with XGBRegressor
        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
                                   param_grid=param_grid, cv=3, n_jobs=-1, verbose=0, scoring='r2')
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        ml_models[param] = best_model

        # Store feature importances
        feature_importances_data[param] = {
            'feature': X.columns.tolist(),
            'importance': best_model.feature_importances_.tolist()
        }

        # Evaluate model performance on the test set
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  LRC '{param}' Model:")
        print(f"    Best Parameters: {grid_search.best_params_}")
        print(f"    R² = {r2:.3f}, RMSE = {rmse:.3f}")

    # --- Apply ML-derived LRC parameters to predict NEE ---
    print("\n--- Applying ML Models to Predict Weekly NEE ---")
    weekly_ml_results = []
    for tower_name, tower_dir in base_directories.items():
        processed_ec_data_path = os.path.join(tower_dir, "processed_hesseflux_2024.csv")
        if not os.path.exists(processed_ec_data_path): continue
        df_ec = pd.read_csv(processed_ec_data_path, index_col='DateTime', parse_dates=True)
        df_ec['Week'] = df_ec.index.isocalendar().week.astype(int)

        for week in range(1, 53):
            df_ec_target_week = df_ec[df_ec['Week'] == week].copy()
            
            # Get satellite and environmental data for the current week and tower
            # Need to ensure current_week_data has lagged features and seasonal indicators
            current_week_data_full = merged_data[
                (merged_data['Tower'] == tower_name) &
                (merged_data['Week'] == week)
            ]

            if df_ec_target_week.empty or current_week_data_full.empty: continue

            # Prepare features for prediction (ensure all features are present)
            current_features_for_prediction = current_week_data_full[features].iloc[0].to_frame().T

            try:
                # Predict LRC parameters using ML models
                a_pred = ml_models['a'].predict(current_features_for_prediction)[0]
                b_pred = ml_models['b'].predict(current_features_for_prediction)[0]
                c_pred = ml_models['c'].predict(current_features_for_prediction)[0]

                # Predict NEE using Mitscherlich function with ML-derived LRC params
                y_predicted = mitscherlich_lrc(df_ec_target_week['PPFD'].values, a_pred, b_pred, c_pred)
                y_observed = df_ec_target_week['NEE'].values

                valid_indices = ~np.isnan(y_observed) & ~np.isnan(y_predicted)
                if np.sum(valid_indices) > 0:
                    weekly_ml_results.append({
                        'Tower': tower_name, 'Week': week,
                        'Measured_NEE': np.mean(y_observed[valid_indices]),
                        'Modeled_NEE': np.mean(y_predicted[valid_indices])
                    })
            except Exception as e:
                # print(f"Error predicting NEE for {tower_name} Week {week}: {e}")
                continue

    ml_results_df = pd.DataFrame(weekly_ml_results)

    # --- 4. Visualize and Report ---
    print("\n--- Generating Time Series Comparison Plots for ML Models ---")
    plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})
    fig, axes = plt.subplots(len(base_directories), 1, figsize=(12, 16), sharex=True)
    fig.suptitle('ML Model (VIs + Environmental + Lagged + Seasonal + Poly/Interactions, Tuned, XGBoost): Measured vs. Modeled Weekly NEE (2024)', fontsize=16, y=0.95)

    for i, (tower_name, ax) in enumerate(zip(base_directories.keys(), axes.flatten())):
        tower_data = ml_results_df[ml_results_df['Tower'] == tower_name]
        if tower_data.empty: 
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        ax.plot(tower_data['Week'], tower_data['Measured_NEE'], 'o-', label='Measured NEE', markersize=4)
        ax.plot(tower_data['Week'], tower_data['Modeled_NEE'], 's--', label='Modeled NEE', markersize=4)
        
        if not tower_data.empty:
            overall_r2 = r2_score(tower_data['Measured_NEE'], tower_data['Modeled_NEE'])
            overall_rmse = np.sqrt(mean_squared_error(tower_data['Measured_NEE'], tower_data['Modeled_NEE']))
            ax.set_title(f'{tower_name} (R² = {overall_r2:.2f}, RMSE = {overall_rmse:.2f})')
        else:
            ax.set_title(tower_name)

        ax.set_ylabel('Mean NEE')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.xlabel('Week of Year')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_plots_dir = os.path.join(os.path.expanduser("~"), "Documents", "combined_plots")
    os.makedirs(combined_plots_dir, exist_ok=True)
    plt.savefig(os.path.join(combined_plots_dir, 'ml_model_environmental_tuned_xgboost_lagged_seasonal_poly_interactions_weekly_nee_comparison.png'))
    plt.close()
    print(f"ML model (VIs + Environmental + Lagged + Seasonal + Poly/Interactions, Tuned, XGBoost) comparison plot saved to {combined_plots_dir}")

    # --- Feature Importance Plotting ---
    plot_feature_importances(feature_importances_data, combined_plots_dir)