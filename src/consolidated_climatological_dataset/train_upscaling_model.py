import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor

def train_upscaling_model(consolidated_data_path, train_towers, test_towers):
    # --- 1. Load and Prepare Data ---
    print("--- Loading consolidated data ---")
    full_data = pd.read_csv(consolidated_data_path)

    print(f"Full data shape after load: {full_data.shape}")
    print(f"Full data columns after load: {full_data.columns.tolist()}")
    print(f"Full data head after load:\n{full_data.head()}")
    print(f"NaNs in full_data after load:\n{full_data.isnull().sum()[full_data.isnull().sum() > 0]}")

    # Ensure 'Week' column is present for temporal features
    full_data['Week'] = full_data['DayOfYear'].apply(lambda x: (x - 1) // 7 + 1)

    # --- 2. Define Features and Target ---
    # Use available VI columns from the consolidated data
    vi_types = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    available_vi_cols = [vi for vi in vi_types if vi in full_data.columns]

    # Features: PPFD, VIs, and temporal features
    feature_cols = ['PPFD'] + available_vi_cols + ['Week'] # Use Week directly for now
    target_col = 'NEE'

    # Drop rows with NaN in any feature or target column
    ml_data = full_data.dropna(subset=feature_cols + [target_col])

    if ml_data.empty:
        print("Not enough data after dropping NaNs for ML model training. Exiting.")
        return

    X = ml_data[feature_cols]
    y = ml_data[target_col]
    towers = ml_data['Tower']

    # --- 3. Split Data by Tower ---
    X_train = X[towers.isin(train_towers)]
    y_train = y[towers.isin(train_towers)]
    X_test = X[towers.isin(test_towers)]
    y_test = y[towers.isin(test_towers)]

    # Also get the tower names for the train and test set to group by later
    towers_train = towers[towers.isin(train_towers)]
    towers_test = towers[towers.isin(test_towers)]

    if X_train.empty or X_test.empty:
        print("Not enough data for train/test split. Check tower names or data availability.")
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 4. Train XGBoost Model with Hyperparameter Tuning ---
    print("--- Training XGBoost Model with Hyperparameter Tuning ---")
    xgb_param_grid = {
        'n_estimators': [50, 100, 200], 
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, verbose=0, scoring='r2')
    xgb_grid_search.fit(X_train, y_train)
    best_xgb_model = xgb_grid_search.best_estimator_

    print(f"XGBoost Best parameters found: {xgb_grid_search.best_params_}")

    # --- Train Random Forest Model with Hyperparameter Tuning ---
    print("--- Training Random Forest Model with Hyperparameter Tuning ---")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 5, 10]
    }

    rf_model = RandomForestRegressor(random_state=42)
    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=0, scoring='r2')
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_

    print(f"Random Forest Best parameters found: {rf_grid_search.best_params_}")

    # --- 5. Evaluate Models ---
    print("--- Evaluating Model Performance ---")

    models = {
        'XGBoost': best_xgb_model,
        'RandomForest': best_rf_model
    }

    for model_name, model_obj in models.items():
        y_pred_train = model_obj.predict(X_train)
        y_pred_test = model_obj.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\n--- {model_name} Performance ---")
        print(f"Training R²: {r2_train:.3f}")
        print(f"Training RMSE: {rmse_train:.3f}")
        print(f"Testing R²: {r2_test:.3f}")
        print(f"Testing RMSE: {rmse_test:.3f}")

        # --- 6. Visualize Results ---
        print(f"--- Generating Performance Plots for {model_name} ---")
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300})

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Measured vs. Modeled Weekly NEE ({model_name})', fontsize=14)

        # Training Data Plot
        axes[0].scatter(y_train, y_pred_train, alpha=0.6)
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r', lw=2)
        axes[0].set_xlabel('Measured NEE')
        axes[0].set_ylabel('Modeled NEE')
        axes[0].set_title(f'Training Data (R²={r2_train:.2f}, RMSE={rmse_train:.2f})')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Testing Data Plot
        axes[1].scatter(y_test, y_pred_test, alpha=0.6)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
        axes[1].set_xlabel('Measured NEE')
        axes[1].set_ylabel('Modeled NEE')
        axes[1].set_title(f'Testing Data (R²={r2_test:.2f}, RMSE={rmse_test:.2f})')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_plot_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", f'ml_nee_upscaling_performance_{model_name.lower()}.png')
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Performance plot saved to: {output_plot_path}")

        # --- Bar Chart: Measured vs. Modeled NEE per Training Tower ---
        print(f"--- Generating Bar Chart for Measured vs. Modeled NEE per Training Tower ({model_name}) ---")
        train_results = pd.DataFrame({'Measured': y_train, 'Modeled': y_pred_train, 'Tower': towers_train})
        avg_train_results = train_results.groupby('Tower')[['Measured', 'Modeled']].mean()

        fig, ax = plt.subplots(figsize=(8, 6))
        avg_train_results.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title(f'Average Measured vs. Modeled NEE for Training Towers ({model_name})')
        ax.set_ylabel('Average NEE')
        ax.set_xlabel('Tower')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        bar_chart_train_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", f'ml_nee_upscaling_bar_chart_train_{model_name.lower()}.png')
        plt.savefig(bar_chart_train_path)
        plt.close()
        print(f"Bar chart for training towers saved to: {bar_chart_train_path}")

        # --- Bar Chart: Measured vs. Modeled NEE per Test Tower ---
        print(f"--- Generating Bar Chart for Measured vs. Modeled NEE per Test Tower ({model_name}) ---")
        test_results = pd.DataFrame({'Measured': y_test, 'Modeled': y_pred_test, 'Tower': towers_test})
        avg_test_results = test_results.groupby('Tower')[['Measured', 'Modeled']].mean()

        fig, ax = plt.subplots(figsize=(8, 6))
        avg_test_results.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title(f'Average Measured vs. Modeled NEE for Test Towers ({model_name})')
        ax.set_ylabel('Average NEE')
        ax.set_xlabel('Tower')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        bar_chart_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", f'ml_nee_upscaling_bar_chart_test_{model_name.lower()}.png')
        plt.savefig(bar_chart_path)
        plt.close()
        print(f"Bar chart for testing towers saved to: {bar_chart_path}")

    # --- Feature Importance Plot (Overall) ---
    print("--- Generating Feature Importance Plot ---")
    # Use the best XGBoost model for overall feature importance as it was the first one tuned
    feature_importances = pd.Series(best_xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 7))
    feature_importances.plot(kind='barh')
    plt.title('XGBoost Feature Importances for Weekly NEE Upscaling (Tuned)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    feature_importance_plot_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", "ml_nee_upscaling_feature_importance_tuned.png")
    plt.savefig(feature_importance_plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {feature_importance_plot_path}")


if __name__ == '__main__':
    consolidated_data_file = "/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv"

    # Randomly selected towers (as per previous instruction)
    train_towers = ["Gurteen", "Athenry", "JC1", "Timoleague", "Clarabog"]
    test_towers = ["JC2", "Lullymore"]

    train_upscaling_model(consolidated_data_file, train_towers, test_towers)
