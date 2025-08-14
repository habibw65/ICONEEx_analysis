import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam



def train_dl_upscaling_model(consolidated_data_path, train_towers, test_towers):
    # --- 1. Load and Prepare Data ---
    print("--- Loading consolidated data ---")
    full_data = pd.read_csv(consolidated_data_path)

    # --- 2. Feature Engineering ---
    # Convert DayOfYear, Hour, Minute to sine/cosine components
    full_data['DayOfYear_sin'] = np.sin(2 * np.pi * full_data['DayOfYear'] / 365)
    full_data['DayOfYear_cos'] = np.cos(2 * np.pi * full_data['DayOfYear'] / 365)
    full_data['Hour_sin'] = np.sin(2 * np.pi * full_data['Hour'] / 24)
    full_data['Hour_cos'] = np.cos(2 * np.pi * full_data['Hour'] / 24)
    full_data['Minute_sin'] = np.sin(2 * np.pi * full_data['Minute'] / 60)
    full_data['Minute_cos'] = np.cos(2 * np.pi * full_data['Minute'] / 60)

    # Define features and target
    vi_types = ['EVI', 'NDMI', 'NDVI', 'SAVI']
    available_vi_cols = [vi for vi in vi_types if vi in full_data.columns]

    feature_cols = ['PPFD'] + available_vi_cols + \
                   ['DayOfYear_sin', 'DayOfYear_cos', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos']
    target_col = 'NEE'

    # Drop rows with NaN in any feature or target column
    ml_data = full_data.dropna(subset=feature_cols + [target_col])

    if ml_data.empty:
        print("Not enough data after dropping NaNs for DL model training. Exiting.")
        return

    X = ml_data[feature_cols]
    y = ml_data[target_col]
    towers = ml_data['Tower']

    # --- 3. Split Data by Tower ---
    X_train = X[towers.isin(train_towers)]
    y_train = y[towers.isin(train_towers)]
    X_test = X[towers.isin(test_towers)]
    y_test = y[towers.isin(test_towers)]

    # Store tower names for plotting
    towers_train = towers[towers.isin(train_towers)]
    towers_test = towers[towers.isin(test_towers)]

    if X_train.empty or X_test.empty:
        print("Not enough data for train/test split. Check tower names or data availability.")
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Data Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Reshape data for CNN-LSTM ---
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # --- 4. Define and Tune Deep Learning Model (CNN-LSTM) ---
    print("--- Defining and Tuning Deep Learning Model (CNN-LSTM) ---")

    param_grid = {
        'batch_size': [32, 64],
        'epochs': [50, 100],
        'learning_rate': [0.001, 0.0001],
        'filters': [32, 64],
        'kernel_size': [3, 5],
        'lstm_units': [50, 100],
        'dropout_rate': [0.3, 0.4],
        'l2_reg': [0.01, 0.001]
    }

    best_score = np.inf
    best_params = {}
    best_model = None

    for batch_size in param_grid['batch_size']:
        for epochs in param_grid['epochs']:
            for learning_rate in param_grid['learning_rate']:
                for filters in param_grid['filters']:
                    for kernel_size in param_grid['kernel_size']:
                        for lstm_units in param_grid['lstm_units']:
                            for dropout_rate in param_grid['dropout_rate']:
                                for l2_reg in param_grid['l2_reg']:
                        
                                    # Create and compile the model
                                    model = Sequential([
                                        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), kernel_regularizer=l2(l2_reg)),
                                        MaxPooling1D(pool_size=2),
                                        LSTM(units=lstm_units, kernel_regularizer=l2(l2_reg)),
                                        Dropout(dropout_rate),
                                        Dense(1)
                                    ])
                                    optimizer = Nadam(learning_rate=learning_rate)
                                    model.compile(optimizer=optimizer, loss='mse')
                                    
                                    # Train the model
                                    history = model.fit(X_train_reshaped, y_train,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      validation_split=0.2, # Use a portion of training data for validation
                                                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                                                      verbose=0)
                                    
                                    # Evaluate the model
                                    val_loss = min(history.history['val_loss'])
                                    
                                    # Check if this is the best model so far
                                    if val_loss < best_score:
                                        best_score = val_loss
                                        best_params = {
                                            'batch_size': batch_size,
                                            'epochs': epochs,
                                            'learning_rate': learning_rate,
                                            'filters': filters,
                                            'kernel_size': kernel_size,
                                            'lstm_units': lstm_units,
                                            'dropout_rate': dropout_rate,
                                            'l2_reg': l2_reg
                                        }
                                        best_model = model

    print(f"Best parameters found: {best_params}")

    # --- 5. Evaluate Model ---
    print("--- Evaluating Model Performance ---")
    y_pred_train = best_model.predict(X_train_reshaped).flatten()
    y_pred_test = best_model.predict(X_test_reshaped).flatten()

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"Training R²: {r2_train:.3f}")
    print(f"Training RMSE: {rmse_train:.3f}")
    print(f"Testing R²: {r2_test:.3f}")
    print(f"Testing RMSE: {rmse_test:.3f}")

    # --- 6. Visualize Results ---
    print("--- Generating Performance Plots ---")
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300})

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Measured vs. Modeled Weekly NEE (Deep Learning MLP - Tuned)', fontsize=14)

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
    output_plot_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", "dl_nee_upscaling_performance_tuned.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Performance plot saved to: {output_plot_path}")

    # --- Bar Chart: Measured vs. Modeled NEE per Training Tower ---
    print("--- Generating Bar Chart for Measured vs. Modeled NEE per Training Tower ---")
    train_results = pd.DataFrame({'Measured': y_train, 'Modeled': y_pred_train, 'Tower': towers_train})
    avg_train_results = train_results.groupby('Tower')[['Measured', 'Modeled']].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    avg_train_results.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Average Measured vs. Modeled NEE for Training Towers')
    ax.set_ylabel('Average NEE')
    ax.set_xlabel('Tower')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    bar_chart_train_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", "dl_nee_upscaling_bar_chart_train_tuned.png")
    plt.savefig(bar_chart_train_path)
    plt.close()
    print(f"Bar chart for training towers saved to: {bar_chart_train_path}")

    # --- Bar Chart: Measured vs. Modeled NEE per Test Tower ---
    print("--- Generating Bar Chart for Measured vs. Modeled NEE per Test Tower ---")
    test_results = pd.DataFrame({'Measured': y_test, 'Modeled': y_pred_test, 'Tower': towers_test})
    avg_test_results = test_results.groupby('Tower')[['Measured', 'Modeled']].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    avg_test_results.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Average Measured vs. Modeled NEE for Test Towers')
    ax.set_ylabel('Average NEE')
    ax.set_xlabel('Tower')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    bar_chart_path = os.path.join(os.path.expanduser("~"), "Documents", "consolidated climatological dataset", "dl_nee_upscaling_bar_chart_test_tuned.png")
    plt.savefig(bar_chart_path)
    plt.close()
    print(f"Bar chart for testing towers saved to: {bar_chart_path}")


if __name__ == '__main__':
    consolidated_data_file = "/Users/habibw/Documents/consolidated climatological dataset/consolidated_half_hourly_climatology_data.csv"

    # Randomly selected towers (as per previous instruction)
    train_towers = ["Gurteen", "Athenry", "JC1", "Timoleague", "Clarabog"]
    test_towers = ["JC2", "Lullymore"]

    train_dl_upscaling_model(consolidated_data_file, train_towers, test_towers)