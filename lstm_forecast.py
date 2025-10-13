import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv('DATA_GEO_Train.csv')

print("Original data shape:", df.shape)
print("Columns:", df.columns.tolist())

# Convert utc_time to datetime
df['utc_time'] = pd.to_datetime(df['utc_time'], format='%m/%d/%Y %H:%M')

# Sort by date and remove duplicates
df = df.sort_values('utc_time')
df = df.drop_duplicates(subset=['utc_time'], keep='first')

print("\nAfter removing duplicates:", df.shape)
print("Date range:", df['utc_time'].min(), "to", df['utc_time'].max())

# Split data: Train on data until end of 9/6/2025, Test on 9/7/2025
train_end = pd.to_datetime('2025-09-06 23:59:59')
train_df = df[df['utc_time'] <= train_end].copy()
test_df = df[df['utc_time'] > train_end].copy()

print(f"\nTrain set: {len(train_df)} samples (until {train_end})")
print(f"Test set: {len(test_df)} samples (9/7/2025)")

# Select columns for prediction (all error columns)
value_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']

# Resample to regular intervals (2 hours based on your data pattern)
train_df = train_df.set_index('utc_time')
train_resampled = train_df[value_cols].resample('2h').mean()
train_resampled = train_resampled.interpolate(method='linear')
train_resampled = train_resampled.dropna()

test_df = test_df.set_index('utc_time')
test_resampled = test_df[value_cols].resample('2h').mean()
test_resampled = test_resampled.interpolate(method='linear')
test_resampled = test_resampled.dropna()

print(f"\nResampled train: {len(train_resampled)} points")
print(f"Resampled test: {len(test_resampled)} points")

# Create Darts TimeSeries objects
train_series = TimeSeries.from_dataframe(
    train_resampled.reset_index(), 
    time_col='utc_time',
    value_cols=value_cols,
    freq='2h'
)

test_series = TimeSeries.from_dataframe(
    test_resampled.reset_index(), 
    time_col='utc_time',
    value_cols=value_cols,
    freq='2h'
)

print(f"\nTrain TimeSeries: {len(train_series)} points")
print(f"Test TimeSeries: {len(test_series)} points")

# Scale the data
scaler = Scaler()
train_scaled = scaler.fit_transform(train_series)
test_scaled = scaler.transform(test_series)

# Create and train LSTM model
model = RNNModel(
    model='LSTM',
    hidden_dim=32,
    n_rnn_layers=2,
    dropout=0.2,
    input_chunk_length=4,
    training_length=20,
    n_epochs=1000,
    batch_size=12,
    random_state=42,
    optimizer_kwargs={'lr': 0.0001},
    pl_trainer_kwargs={"accelerator": "cpu"}
)

print("\n" + "="*60)
print("Training LSTM model on data until 9/6/2025...")
print("="*60)
model.fit(train_scaled, verbose=True)

# Predict for 9/7/2025
print("\n" + "="*60)
print("Predicting for 9/7/2025...")
print("="*60)
forecast_scaled = model.predict(n=len(test_series))

# Inverse transform
forecast = scaler.inverse_transform(forecast_scaled)

# Calculate metrics including MSE and RSE
print("\n" + "="*40)
print("PERFORMANCE METRICS FOR 9/7/2025")
print("="*40)

for i, col in enumerate(value_cols):
    test_col = test_series.univariate_component(i)
    forecast_col = forecast.univariate_component(i)
    
    # Get actual values
    actual_values = test_col.values().flatten()
    predicted_values = forecast_col.values().flatten()
    
    # Calculate existing metrics
    mape_score = mape(test_col, forecast_col)
    rmse_score = rmse(test_col, forecast_col)
    mae_score = mae(test_col, forecast_col)
    
    # Calculate MSE (Mean Squared Error)
    mse_score = np.mean((actual_values - predicted_values) ** 2)
    
    # Calculate RSE (Relative Squared Error)
    mean_actual = np.mean(actual_values)
    sse = np.sum((actual_values - predicted_values) ** 2)  # Sum of squared errors
    sst = np.sum((actual_values - mean_actual) ** 2)  # Total sum of squares
    rse_score = sse / sst if sst != 0 else float('inf')
    
    # Calculate R² for reference
    r2_score = 1 - rse_score if sst != 0 else 0
    
    print(f"\n{col}:")
    print(f"  MAE:   {mae_score:.6f} meters")
    print(f"  MSE:   {mse_score:.6f} meters²")
    print(f"  RMSE:  {rmse_score:.6f} meters")
    print(f"  RSE:   {rse_score:.6f}")
    print(f"  R²:    {r2_score:.6f}")
    print(f"  MAPE:  {mape_score:.2f}%")

print("="*40)

# Create plots
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('LSTM Time Series Forecast - Train until 9/6/2025, Predict 9/7/2025', 
             fontsize=16, fontweight='bold')

for i, col in enumerate(value_cols):
    # Full view
    train_col = train_series.univariate_component(i)
    test_col = test_series.univariate_component(i)
    forecast_col = forecast.univariate_component(i)
    
    # Plot 1: Full timeline
    ax1 = axes[i, 0]
    train_col.plot(ax=ax1, label='Training (until 9/6)', linewidth=1.5)
    test_col.plot(ax=ax1, label='Actual (9/7)', linewidth=2, color='green')
    forecast_col.plot(ax=ax1, label='Predicted (9/7)', linewidth=2, 
                      linestyle='--', color='red', alpha=0.8)
    ax1.axvline(x=train_end, color='black', linestyle=':', linewidth=2, 
                label='Train/Test Split')
    ax1.set_title(f'{col} - Full View', fontweight='bold')
    ax1.set_ylabel('Error (meters)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed on 9/7/2025
    ax2 = axes[i, 1]
    test_col.plot(ax=ax2, label='Actual', linewidth=2.5, color='green', marker='o')
    forecast_col.plot(ax=ax2, label='Predicted', linewidth=2.5, 
                      linestyle='--', color='red', marker='s', alpha=0.8)
    ax2.set_title(f'{col} - 9/7/2025 Detail', fontweight='bold')
    ax2.set_ylabel('Error (meters)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_forecast.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'multivariate_forecast.png'")
plt.show()

# Save detailed results to CSV with all metrics
results_data = {
    'timestamp': test_series.time_index
}

# Store metrics summary
metrics_summary = []

for i, col in enumerate(value_cols):
    test_col = test_series.univariate_component(i)
    forecast_col = forecast.univariate_component(i)
    
    actual_vals = test_col.values().flatten()
    predicted_vals = forecast_col.values().flatten()
    
    results_data[f'actual_{col}'] = actual_vals
    results_data[f'predicted_{col}'] = predicted_vals
    results_data[f'error_{col}'] = (actual_vals - predicted_vals)
    results_data[f'squared_error_{col}'] = (actual_vals - predicted_vals) ** 2
    
    # Calculate all metrics
    mae_val = np.mean(np.abs(actual_vals - predicted_vals))
    mse_val = np.mean((actual_vals - predicted_vals) ** 2)
    rmse_val = np.sqrt(mse_val)
    
    mean_actual = np.mean(actual_vals)
    sse = np.sum((actual_vals - predicted_vals) ** 2)
    sst = np.sum((actual_vals - mean_actual) ** 2)
    rse_val = sse / sst if sst != 0 else float('inf')
    r2_val = 1 - rse_val if sst != 0 else 0
    
    metrics_summary.append({
        'variable': col,
        'MAE': mae_val,
        'MSE': mse_val,
        'RMSE': rmse_val,
        'RSE': rse_val,
        'R²': r2_val
    })

results_df = pd.DataFrame(results_data)
results_df.to_csv('forecast_results_9_7_2025.csv', index=False)
print("\nDetailed results saved to 'forecast_results_9_7_2025.csv'")

# Save metrics summary
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('metrics_summary_9_7_2025.csv', index=False)
print("Metrics summary saved to 'metrics_summary_9_7_2025.csv'")

# Print sample predictions
print("\n" + "="*40)
print("SAMPLE PREDICTIONS FOR 9/7/2025")
print("="*40)
print(results_df.to_string(index=False))
print("="*40)

# Print metrics summary table
print("\n" + "="*40)
print("METRICS SUMMARY TABLE")
print("="*40)
print(metrics_df.to_string(index=False))
print("="*40)