"""
ERA5-Enhanced NEE Upscaling Model

This script develops an advanced machine learning model for NEE upscaling by integrating:
1. ERA5 reanalysis meteorological data
2. Light Response Curve (LRC) parameters 
3. Satellite-derived Vegetation Indices (VIs)
4. Temporal features

The model aims to improve CO2 flux predictions by incorporating comprehensive
atmospheric and biophysical information.

Key Features:
- Integrates multiple data sources (ERA5, LRC, satellite VIs)
- Advanced feature engineering with meteorological variables
- Multiple ML algorithms (XGBoost, Random Forest, Deep Learning)
- Cross-validation with spatial splitting
- Comprehensive model evaluation and interpretation

Author: ICONeEx Analysis Team
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
    TF_AVAILABLE = False

# Plotting
plt.style.use('default')
sns.set_palette("husl")

# Tower coordinates for reference
TOWER_COORDINATES = {
    'Gurteen': (53.2914, -8.2347),
    'Athenry': (53.3031, -8.7389),
    'JC1': (53.2819, -7.9472),
    'JC2': (53.2819, -7.9472),
    'Timoleague': (51.6464, -8.7361),
    'Lullymore': (53.3506, -6.9194),
    'Clarabog': (54.2667, -7.8333)
}

def load_era5_data(era5_dir):
    """
    Load processed ERA5 data for all towers.
    
    Args:
        era5_dir: Directory containing processed ERA5 CSV files
    
    Returns:
        dict: ERA5 data for each tower
    """
    print("Loading ERA5 data...")
    era5_data = {}
    
    for tower in TOWER_COORDINATES.keys():
        era5_file = os.path.join(era5_dir, f"era5_{tower.lower()}_half_hourly.csv")
        
        if os.path.exists(era5_file):
            df = pd.read_csv(era5_file)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            era5_data[tower] = df
            print(f"Loaded ERA5 data for {tower}: {df.shape}")
        else:
            print(f"ERA5 file not found for {tower}: {era5_file}")
    
    return era5_data

def load_lrc_parameters(lrc_file):
    """
    Load Light Response Curve parameters.
    
    Args:
        lrc_file: Path to LRC parameters CSV file
    
    Returns:
        DataFrame: LRC parameters data
    """
    print("Loading LRC parameters...")
    
    if os.path.exists(lrc_file):
        df = pd.read_csv(lrc_file)
        print(f"Loaded LRC parameters: {df.shape}")
        return df
    else:
        print(f"LRC parameters file not found: {lrc_file}")
        return None

def load_satellite_vis(consolidated_file):
    """
    Load satellite vegetation indices from consolidated dataset.
    
    Args:
        consolidated_file: Path to consolidated climatology data
    
    Returns:
        DataFrame: Satellite VI data
    """
    print("Loading satellite vegetation indices...")
    
    if os.path.exists(consolidated_file):
        df = pd.read_csv(consolidated_file)
        
        # Extract VI columns
        vi_columns = [col for col in df.columns if any(vi in col.lower() for vi in ['evi', 'ndvi', 'ndmi', 'savi'])]
        
        if vi_columns:
            # Keep DateTime, Tower, and VI columns
            essential_cols = ['DateTime', 'Tower'] + vi_columns
            df_vis = df[essential_cols].copy()
            df_vis['DateTime'] = pd.to_datetime(df_vis['DateTime'])
            print(f"Loaded satellite VIs: {df_vis.shape}, Variables: {vi_columns}")
            return df_vis
        else:
            print("No vegetation index columns found in consolidated data")
            return None
    else:
        print(f"Consolidated data file not found: {consolidated_file}")
        return None

def load_flux_data(consolidated_file):
    """
    Load NEE flux data from consolidated dataset.
    
    Args:
        consolidated_file: Path to consolidated climatology data
    
    Returns:
        DataFrame: NEE data
    """
    print("Loading NEE flux data...")
    
    if os.path.exists(consolidated_file):
        df = pd.read_csv(consolidated_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Keep essential columns for NEE prediction
        nee_columns = [col for col in df.columns if 'nee' in col.lower()]
        ppfd_columns = [col for col in df.columns if 'ppfd' in col.lower()]
        
        essential_cols = ['DateTime', 'Tower'] + nee_columns + ppfd_columns
        df_flux = df[[col for col in essential_cols if col in df.columns]].copy()
        
        print(f"Loaded NEE flux data: {df_flux.shape}")
        print(f"NEE columns: {nee_columns}")
        return df_flux
    else:
        print(f"Consolidated data file not found: {consolidated_file}")
        return None

def create_temporal_features(df):
    """
    Create temporal features from DateTime.
    
    Args:
        df: DataFrame with DateTime column
    
    Returns:
        DataFrame: DataFrame with additional temporal features
    """
    df = df.copy()
    
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
    
    # Cyclical encoding for seasonal patterns
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    return df

def integrate_all_data(era5_data, lrc_params, satellite_vis, flux_data):
    """
    Integrate all data sources into a single dataset.
    
    Args:
        era5_data: Dictionary of ERA5 data by tower
        lrc_params: LRC parameters DataFrame
        satellite_vis: Satellite VI DataFrame
        flux_data: NEE flux DataFrame
    
    Returns:
        DataFrame: Integrated dataset
    """
    print("\\nIntegrating all data sources...")
    
    integrated_data = []
    
    for tower in TOWER_COORDINATES.keys():
        print(f"Processing {tower}...")
        
        # Start with flux data for this tower
        tower_flux = flux_data[flux_data['Tower'] == tower].copy()
        
        if tower_flux.empty:
            print(f"No flux data for {tower}")
            continue
        
        # Merge with ERA5 data
        if tower in era5_data:
            tower_era5 = era5_data[tower].copy()
            tower_era5['Tower'] = tower
            
            # Merge on DateTime and Tower
            merged_data = pd.merge(tower_flux, tower_era5, on=['DateTime', 'Tower'], how='inner')
            print(f"After ERA5 merge: {merged_data.shape}")
        else:
            merged_data = tower_flux
            print(f"No ERA5 data for {tower}, using flux data only")
        
        # Merge with satellite VIs
        if satellite_vis is not None:
            tower_vis = satellite_vis[satellite_vis['Tower'] == tower].copy()
            if not tower_vis.empty:
                merged_data = pd.merge(merged_data, tower_vis, on=['DateTime', 'Tower'], how='left')
                print(f"After satellite VI merge: {merged_data.shape}")
        
        # Merge with LRC parameters (weekly data)
        if lrc_params is not None:
            tower_lrc = lrc_params[lrc_params['Tower'] == tower].copy()
            if not tower_lrc.empty:
                # Add week of year to main data for merging
                merged_data['WeekOfYear'] = merged_data['DateTime'].dt.isocalendar().week
                
                merged_data = pd.merge(merged_data, tower_lrc, on=['Tower', 'WeekOfYear'], how='left')
                print(f"After LRC parameters merge: {merged_data.shape}")
        
        # Add temporal features
        merged_data = create_temporal_features(merged_data)
        
        integrated_data.append(merged_data)
    
    # Combine all towers
    if integrated_data:
        final_data = pd.concat(integrated_data, ignore_index=True)
        print(f"\\nFinal integrated dataset: {final_data.shape}")
        print(f"Columns: {list(final_data.columns)}")
        return final_data
    else:
        print("No data to integrate")
        return None

def prepare_features_and_target(df, target_col='NEE'):
    """
    Prepare features and target variable for machine learning.
    
    Args:
        df: Integrated dataset
        target_col: Name of target column
    
    Returns:
        tuple: (X, y, feature_names)
    """
    print(f"\\nPreparing features and target ({target_col})...")
    
    # Define feature categories
    era5_features = [col for col in df.columns if any(var in col.lower() for var in 
                    ['temperature', 'pressure', 'wind', 'solar', 'precipitation', 
                     'evaporation', 'boundary', 'vpd', 'humidity'])]
    
    vi_features = [col for col in df.columns if any(vi in col.lower() for vi in 
                  ['evi', 'ndvi', 'ndmi', 'savi'])]
    
    lrc_features = [col for col in df.columns if any(param in col.lower() for param in 
                   ['lrc_a', 'lrc_b', 'lrc_c', 'r_squared'])]
    
    temporal_features = [col for col in df.columns if any(temp in col.lower() for temp in 
                        ['month', 'day', 'hour', 'year', 'week', 'sin', 'cos'])]
    
    ppfd_features = [col for col in df.columns if 'ppfd' in col.lower()]
    
    # Combine all feature categories
    feature_cols = era5_features + vi_features + lrc_features + temporal_features + ppfd_features
    
    # Remove duplicate columns
    feature_cols = list(set(feature_cols))
    
    # Remove target column if it's in features
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    print(f"ERA5 features ({len(era5_features)}): {era5_features[:5]}...")
    print(f"Satellite VI features ({len(vi_features)}): {vi_features}")
    print(f"LRC features ({len(lrc_features)}): {lrc_features}")
    print(f"Temporal features ({len(temporal_features)}): {temporal_features[:5]}...")
    print(f"PPFD features ({len(ppfd_features)}): {ppfd_features}")
    print(f"Total features: {len(feature_cols)}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Dataset after removing missing targets: {X.shape}")
    
    return X, y, feature_cols

def train_era5_enhanced_models(X, y, feature_names, towers_list):
    """
    Train multiple machine learning models with ERA5-enhanced features.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        towers_list: List of tower names for spatial splitting
    
    Returns:
        dict: Trained models and results
    """
    print("\\nTraining ERA5-enhanced NEE upscaling models...")
    
    # Encode tower names for spatial splitting
    tower_encoder = LabelEncoder()
    tower_encoded = tower_encoder.fit_transform(towers_list)
    
    # Split data spatially (by towers) for training and testing
    unique_towers = np.unique(tower_encoded)
    np.random.seed(42)
    
    # Select 70% of towers for training, 30% for testing
    n_train_towers = max(1, int(0.7 * len(unique_towers)))
    train_towers = np.random.choice(unique_towers, n_train_towers, replace=False)
    test_towers = np.setdiff1d(unique_towers, train_towers)
    
    train_mask = np.isin(tower_encoded, train_towers)
    test_mask = np.isin(tower_encoded, test_towers)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training towers: {[tower_encoder.classes_[i] for i in train_towers]}")
    print(f"Testing towers: {[tower_encoder.classes_[i] for i in test_towers]}")
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    results = {}
    models = {}
    
    # 1. XGBoost Model
    print("\\n1. Training XGBoost model...")
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train_imputed, y_train)
    
    xgb_best = xgb_grid.best_estimator_
    y_pred_train_xgb = xgb_best.predict(X_train_imputed)
    y_pred_test_xgb = xgb_best.predict(X_test_imputed)
    
    results['XGBoost'] = {
        'model': xgb_best,
        'train_r2': r2_score(y_train, y_pred_train_xgb),
        'test_r2': r2_score(y_test, y_pred_test_xgb),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_xgb)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_xgb)),
        'best_params': xgb_grid.best_params_,
        'predictions': {'train': y_pred_train_xgb, 'test': y_pred_test_xgb}
    }
    
    # 2. Random Forest Model
    print("\\n2. Training Random Forest model...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train_imputed, y_train)
    
    rf_best = rf_grid.best_estimator_
    y_pred_train_rf = rf_best.predict(X_train_imputed)
    y_pred_test_rf = rf_best.predict(X_test_imputed)
    
    results['Random Forest'] = {
        'model': rf_best,
        'train_r2': r2_score(y_train, y_pred_train_rf),
        'test_r2': r2_score(y_test, y_pred_test_rf),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),
        'best_params': rf_grid.best_params_,
        'predictions': {'train': y_pred_train_rf, 'test': y_pred_test_rf}
    }
    
    # 3. Deep Learning Model (if TensorFlow is available)
    if TF_AVAILABLE:
        print("\\n3. Training Deep Neural Network...")
        
        # Build neural network
        model = Sequential([\n            Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),\n            Dropout(0.3),\n            Dense(256, activation='relu'),\n            Dropout(0.3),\n            Dense(128, activation='relu'),\n            Dropout(0.2),\n            Dense(64, activation='relu'),\n            Dense(1, activation='linear')\n        ])\n        \n        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])\n        \n        # Callbacks\n        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)\n        reduce_lr = ReduceLROnPlateau(patience=10, factor=0.5)\n        \n        # Train model\n        history = model.fit(\n            X_train_scaled, y_train,\n            validation_data=(X_test_scaled, y_test),\n            epochs=200,\n            batch_size=32,\n            callbacks=[early_stopping, reduce_lr],\n            verbose=1\n        )\n        \n        y_pred_train_dl = model.predict(X_train_scaled).flatten()\n        y_pred_test_dl = model.predict(X_test_scaled).flatten()\n        \n        results['Deep Learning'] = {\n            'model': model,\n            'train_r2': r2_score(y_train, y_pred_train_dl),\n            'test_r2': r2_score(y_test, y_pred_test_dl),\n            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_dl)),\n            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_dl)),\n            'history': history,\n            'predictions': {'train': y_pred_train_dl, 'test': y_pred_test_dl}\n        }\n    \n    # Store additional information\n    results['data_info'] = {\n        'train_towers': [tower_encoder.classes_[i] for i in train_towers],\n        'test_towers': [tower_encoder.classes_[i] for i in test_towers],\n        'feature_names': feature_names,\n        'scaler': scaler,\n        'imputer': imputer,\n        'y_train': y_train,\n        'y_test': y_test\n    }\n    \n    return results

def plot_model_results(results, output_dir):
    \"\"\"\n    Create comprehensive plots of model results.\n    \n    Args:\n        results: Dictionary of model results\n        output_dir: Directory to save plots\n    \"\"\"\n    print(\"\\nCreating result plots...\")\n    os.makedirs(output_dir, exist_ok=True)\n    \n    # 1. Performance comparison plot\n    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n    \n    models = [model for model in results.keys() if model != 'data_info']\n    train_r2 = [results[model]['train_r2'] for model in models]\n    test_r2 = [results[model]['test_r2'] for model in models]\n    train_rmse = [results[model]['train_rmse'] for model in models]\n    test_rmse = [results[model]['test_rmse'] for model in models]\n    \n    # R² comparison\n    x_pos = np.arange(len(models))\n    width = 0.35\n    \n    axes[0,0].bar(x_pos - width/2, train_r2, width, label='Training', alpha=0.8)\n    axes[0,0].bar(x_pos + width/2, test_r2, width, label='Testing', alpha=0.8)\n    axes[0,0].set_xlabel('Model')\n    axes[0,0].set_ylabel('R² Score')\n    axes[0,0].set_title('Model R² Performance Comparison')\n    axes[0,0].set_xticks(x_pos)\n    axes[0,0].set_xticklabels(models, rotation=45)\n    axes[0,0].legend()\n    axes[0,0].grid(True, alpha=0.3)\n    \n    # RMSE comparison\n    axes[0,1].bar(x_pos - width/2, train_rmse, width, label='Training', alpha=0.8)\n    axes[0,1].bar(x_pos + width/2, test_rmse, width, label='Testing', alpha=0.8)\n    axes[0,1].set_xlabel('Model')\n    axes[0,1].set_ylabel('RMSE (μmol m⁻² s⁻¹)')\n    axes[0,1].set_title('Model RMSE Performance Comparison')\n    axes[0,1].set_xticks(x_pos)\n    axes[0,1].set_xticklabels(models, rotation=45)\n    axes[0,1].legend()\n    axes[0,1].grid(True, alpha=0.3)\n    \n    # Feature importance (XGBoost)\n    if 'XGBoost' in results:\n        feature_names = results['data_info']['feature_names']\n        importance = results['XGBoost']['model'].feature_importances_\n        \n        # Get top 15 features\n        top_indices = np.argsort(importance)[-15:]\n        top_importance = importance[top_indices]\n        top_features = [feature_names[i] for i in top_indices]\n        \n        axes[1,0].barh(range(len(top_features)), top_importance)\n        axes[1,0].set_yticks(range(len(top_features)))\n        axes[1,0].set_yticklabels(top_features)\n        axes[1,0].set_xlabel('Feature Importance')\n        axes[1,0].set_title('Top 15 Feature Importance (XGBoost)')\n        axes[1,0].grid(True, alpha=0.3)\n    \n    # Predicted vs Observed (best model)\n    best_model = max(models, key=lambda x: results[x]['test_r2'])\n    y_test = results['data_info']['y_test']\n    y_pred_test = results[best_model]['predictions']['test']\n    \n    axes[1,1].scatter(y_test, y_pred_test, alpha=0.6)\n    min_val = min(y_test.min(), y_pred_test.min())\n    max_val = max(y_test.max(), y_pred_test.max())\n    axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)\n    axes[1,1].set_xlabel('Observed NEE (μmol m⁻² s⁻¹)')\n    axes[1,1].set_ylabel('Predicted NEE (μmol m⁻² s⁻¹)')\n    axes[1,1].set_title(f'Predicted vs Observed ({best_model})\\nR² = {results[best_model][\"test_r2\"]:.3f}')\n    axes[1,1].grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig(os.path.join(output_dir, 'era5_enhanced_model_results.png'), dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(f\"Results plot saved to: {output_dir}\")\n\ndef main():\n    \"\"\"\n    Main function to run the ERA5-enhanced NEE upscaling analysis.\n    \"\"\"\n    print(\"🚀 Starting ERA5-Enhanced NEE Upscaling Analysis...\")\n    \n    # Configuration\n    era5_dir = \"/Users/habibw/Documents/ERA5_Data/half_hourly\"\n    consolidated_file = \"/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv\"\n    lrc_file = \"/Users/habibw/Documents/consolidated climatological dataset/all_towers_weekly_lrc_parameters.csv\"\n    output_dir = \"/Users/habibw/Documents/ERA5_Data/ml_results\"\n    \n    # Load all data sources\n    era5_data = load_era5_data(era5_dir)\n    lrc_params = load_lrc_parameters(lrc_file)\n    satellite_vis = load_satellite_vis(consolidated_file)\n    flux_data = load_flux_data(consolidated_file)\n    \n    if flux_data is None:\n        print(\"❌ No flux data available. Cannot proceed.\")\n        return\n    \n    # Integrate all data\n    integrated_data = integrate_all_data(era5_data, lrc_params, satellite_vis, flux_data)\n    \n    if integrated_data is None or integrated_data.empty:\n        print(\"❌ No integrated data available. Cannot proceed.\")\n        return\n    \n    # Prepare features and target\n    X, y, feature_names = prepare_features_and_target(integrated_data, target_col='NEE')\n    towers_list = integrated_data['Tower'].values\n    \n    # Train models\n    results = train_era5_enhanced_models(X, y, feature_names, towers_list)\n    \n    # Create plots\n    plot_model_results(results, output_dir)\n    \n    # Print summary\n    print(\"\\n\" + \"=\"*50)\n    print(\"📊 ERA5-ENHANCED NEE UPSCALING RESULTS\")\n    print(\"=\"*50)\n    \n    for model_name in results.keys():\n        if model_name != 'data_info':\n            result = results[model_name]\n            print(f\"\\n{model_name}:\")\n            print(f\"  Training R²: {result['train_r2']:.3f}\")\n            print(f\"  Testing R²:  {result['test_r2']:.3f}\")\n            print(f\"  Training RMSE: {result['train_rmse']:.3f} μmol m⁻² s⁻¹\")\n            print(f\"  Testing RMSE:  {result['test_rmse']:.3f} μmol m⁻² s⁻¹\")\n    \n    print(f\"\\nResults saved to: {output_dir}\")\n    print(\"✅ Analysis completed successfully!\")\n\nif __name__ == \"__main__\":\n    main()
