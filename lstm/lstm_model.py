import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('DATA_GEO_Train.csv')

print("Original data shape:", df.shape)
print("Columns:", df.columns.tolist())


print("\n" + "="*60)
print("REMOVING OUTLIERS (points outside +/- 1.5 std dev)")
print("="*60)

initial_rows = len(df)
value_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']

# Remove outliers for each error column
for target_column in value_cols:
    mean = df[target_column].mean()
    std = df[target_column].std()
    lower_bound = mean - 1.5 * std
    upper_bound = mean + 1.5 * std
    
    rows_before = len(df)
    df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    print(f"\n{target_column}:")
    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
    print(f"  Bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
    print(f"  Rows removed: {rows_removed}")

final_rows = len(df)
total_rows_removed = initial_rows - final_rows
print(f"\nTotal rows removed: {total_rows_removed}")
print(f"Final number of rows: {final_rows}")
print("="*60 + "\n")

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
print(f"\nResampling to regular 15-minute intervals for model training...")

# Resample to regular 15-minute intervals to match original data pattern
train_df = train_df.set_index('utc_time')
train_resampled = train_df[value_cols].resample('15min').mean()
train_resampled = train_resampled.interpolate(method='linear')
train_resampled = train_resampled.dropna()

test_df = test_df.set_index('utc_time')
test_resampled = test_df[value_cols].resample('15min').mean()
test_resampled = test_resampled.interpolate(method='linear')
test_resampled = test_resampled.dropna()

print(f"\nResampled to 15min intervals:")
print(f"Train: {len(train_resampled)} points")
print(f"Test: {len(test_resampled)} points")

# Normalize the data using StandardScaler
print("\n" + "="*60)
print("NORMALIZING DATA (Z-score normalization)")
print("="*60)

# Store original statistics for reference
print("\nOriginal data statistics (before normalization):")
print("\nTraining data:")
print(train_resampled.describe())

# Initialize StandardScaler fofitr each column
scalers = {}
train_normalized = train_resampled.copy()
test_normalized = test_resampled.copy()

for col in value_cols:
    scaler = StandardScaler()
    # Fit on training data only
    train_normalized[col] = scaler.fit_transform(train_resampled[[col]])
    # Transform test data to normalized data using training statistics
    test_normalized[col] = scaler.transform(test_resampled[[col]])
    scalers[col] = scaler
    
    print(f"\n{col}:")
    print(f"  Mean (train): {scaler.mean_[0]:.6f}")
    print(f"  Std (train):  {scaler.scale_[0]:.6f}")

print("\nNormalized data statistics (Z-scores):")
print("\nTraining data (should have mean≈0, std≈1):")
print(train_normalized.describe())

print("\nTest data (normalized using training statistics):")
print(test_normalized.describe())

# Use normalized data for TimeSeries creation
train_resampled = train_normalized
test_resampled = test_normalized

print("="*60)

# Create Darts TimeSeries objects
train_series = TimeSeries.from_dataframe(
    train_resampled.reset_index(), 
    time_col='utc_time',
    value_cols=value_cols,
    freq='15min'
)

test_series = TimeSeries.from_dataframe(
    test_resampled.reset_index(), 
    time_col='utc_time',
    value_cols=value_cols,
    freq='15min'
)

print(f"\nTrain TimeSeries: {len(train_series)} points at 15min intervals")
print(f"Test TimeSeries: {len(test_series)} points at 15min intervals")
print(f"Train period: {train_series.start_time()} to {train_series.end_time()}")
print(f"Test period: {test_series.start_time()} to {test_series.end_time()}")

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
    input_chunk_length=12,  # Look back ~3 hours (12 * 15min)
    training_length=24,     # Training sequences of ~6 hours
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

# OPTION 3: Predict with explicit series context and align to test timestamps
print("\n" + "="*60)
print("Predicting for 9/7/2025 with timestamp alignment...")
print("="*60)

# Generate forecast starting from where training ended
forecast_scaled = model.predict(
    n=len(test_series),
    series=train_scaled
)

print(f"\nForecast generated:")
print(f"  Start: {forecast_scaled.start_time()}")
print(f"  End: {forecast_scaled.end_time()}")
print(f"  Length: {len(forecast_scaled)} points")
print(f"\nTest data timestamps:")
print(f"  Start: {test_series.start_time()}")
print(f"  End: {test_series.end_time()}")
print(f"  Length: {len(test_series)} points")

# Inverse transform from Darts Scaler
forecast = scaler.inverse_transform(forecast_scaled)

# Additional denormalization: inverse transform from StandardScaler
print("\nDenormalizing predictions back to original scale...")
forecast_df_denorm = forecast.pd_dataframe() if hasattr(forecast, 'pd_dataframe') else forecast.to_dataframe()

for col in value_cols:
    forecast_df_denorm[col] = scalers[col].inverse_transform(forecast_df_denorm[[col]])

# Also denormalize test series for proper comparison
test_df_denorm = test_series.pd_dataframe() if hasattr(test_series, 'pd_dataframe') else test_series.to_dataframe()
for col in value_cols:
    test_df_denorm[col] = scalers[col].inverse_transform(test_df_denorm[[col]])

# Convert back to TimeSeries
forecast = TimeSeries.from_dataframe(
    forecast_df_denorm.reset_index(),
    time_col=forecast_df_denorm.index.name if forecast_df_denorm.index.name else 'time',
    value_cols=value_cols,
    freq='15min'
)

test_series_original = TimeSeries.from_dataframe(
    test_df_denorm.reset_index(),
    time_col=test_df_denorm.index.name if test_df_denorm.index.name else 'time',
    value_cols=value_cols,
    freq='15min'
)

# Also create denormalized training series for plotting
train_df_original = train_series.pd_dataframe() if hasattr(train_series, 'pd_dataframe') else train_series.to_dataframe()
for col in value_cols:
    train_df_original[col] = scalers[col].inverse_transform(train_df_original[[col]])

train_series_original = TimeSeries.from_dataframe(
    train_df_original.reset_index(),
    time_col=train_df_original.index.name if train_df_original.index.name else 'time',
    value_cols=value_cols,
    freq='15min'
)

print("Denormalization complete - values back in original meters scale")

# Align forecast to test timestamps by converting to pandas and reindexing
print("\nAligning forecast to test timestamps...")

# Convert both to pandas DataFrames
forecast_df = forecast.pd_dataframe() if hasattr(forecast, 'pd_dataframe') else forecast.to_dataframe()
test_df_check = test_series_original.pd_dataframe() if hasattr(test_series_original, 'pd_dataframe') else test_series_original.to_dataframe()

print(f"  Original forecast length: {len(forecast_df)}")
print(f"  Test data length: {len(test_df_check)}")
print(f"  Forecast time range: {forecast_df.index[0]} to {forecast_df.index[-1]}")
print(f"  Test time range: {test_df_check.index[0]} to {test_df_check.index[-1]}")

# Reindex forecast to exactly match test timestamps
forecast_df_aligned = forecast_df.reindex(test_df_check.index, method='nearest', tolerance=pd.Timedelta('15min'))

# Check for any NaN values after reindexing
nan_count = forecast_df_aligned.isna().sum().sum()
if nan_count > 0:
    print(f"\nWarning: {nan_count} NaN values after reindexing. Filling with interpolation...")
    forecast_df_aligned = forecast_df_aligned.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Convert back to TimeSeries
forecast_aligned = TimeSeries.from_dataframe(
    forecast_df_aligned.reset_index(),
    time_col=forecast_df_aligned.index.name if forecast_df_aligned.index.name else 'time',
    value_cols=value_cols,
    freq='15min'
)

print(f"\nAlignment complete:")
print(f"  Aligned forecast length: {len(forecast_aligned)}")
print(f"  Lengths match: {len(forecast_aligned) == len(test_series_original)}")

# Use the aligned forecast for all subsequent calculations
forecast = forecast_aligned

# Verify alignment
print(f"\nVerifying alignment:")
print(f"  Forecast start: {forecast.start_time()}")
print(f"  Test start:     {test_series_original.start_time()}")
print(f"  Forecast end:   {forecast.end_time()}")
print(f"  Test end:       {test_series_original.end_time()}")
print(f"  Match: {forecast.start_time() == test_series_original.start_time() and forecast.end_time() == test_series_original.end_time()}")

# Calculate metrics including MSE and RSE
print("\n" + "="*40)
print("PERFORMANCE METRICS FOR 9/7/2025")
print("="*40)

for i, col in enumerate(value_cols):
    test_col = test_series_original.univariate_component(i)
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

# Shapiro-Wilk Normality Test on Residuals
print("\n" + "="*40)
print("SHAPIRO-WILK NORMALITY TEST")
print("="*40)
print("Testing if prediction errors are normally distributed")
print("H0: Errors are normally distributed")
print("H1: Errors are NOT normally distributed")
print("Significance level: α = 0.05")
print("-"*40)

shapiro_results = []

for i, col in enumerate(value_cols):
    test_col = test_series_original.univariate_component(i)
    forecast_col = forecast.univariate_component(i)
    
    # Calculate residuals (errors)
    residuals = test_col.values().flatten() - forecast_col.values().flatten()
    
    # Perform Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    # Interpretation
    is_normal = "Yes" if shapiro_p > 0.05 else "No"
    
    print(f"\n{col}:")
    print(f"  Shapiro Statistic: {shapiro_stat:.6f}")
    print(f"  P-value:           {shapiro_p:.6f}")
    print(f"  Normal at α=0.05?  {is_normal}")
    
    if shapiro_p > 0.05:
        print(f"  → Cannot reject H0: Errors appear normally distributed")
    else:
        print(f"  → Reject H0: Errors NOT normally distributed")
    
    shapiro_results.append({
        'variable': col,
        'shapiro_statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal_at_0.05': is_normal
    })

print("="*40)

# Create 8 separate plots (2 for each column)
print("\nGenerating 8 separate plots...")

for i, col in enumerate(value_cols):
    train_col = train_series_original.univariate_component(i)
    test_col = test_series_original.univariate_component(i)
    forecast_col = forecast.univariate_component(i)
    
    # Plot 1: Full timeline
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    train_col.plot(ax=ax1, label='Training (until 9/6)', linewidth=1.5)
    test_col.plot(ax=ax1, label='Actual (9/7)', linewidth=2, color='green')
    forecast_col.plot(ax=ax1, label='Predicted (9/7)', linewidth=2, 
                      linestyle='--', color='red', alpha=0.8)
    ax1.axvline(x=train_end, color='black', linestyle=':', linewidth=2, 
                label='Train/Test Split')
    ax1.set_title(f'{col} - Full View', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Error (meters)', fontsize=12)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'forecast_{col.replace(" ", "_").replace("(", "").replace(")", "")}_full_view.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: forecast_{col.replace(' ', '_').replace('(', '').replace(')', '')}_full_view.png")
    plt.close()
    
    # Plot 2: Zoomed on 9/7/2025
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    test_col.plot(ax=ax2, label='Actual', linewidth=2.5, color='green', marker='o')
    forecast_col.plot(ax=ax2, label='Predicted', linewidth=2.5, 
                      linestyle='--', color='red', marker='s', alpha=0.8)
    ax2.set_title(f'{col} - 9/7/2025 Detail', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Error (meters)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'forecast_{col.replace(" ", "_").replace("(", "").replace(")", "")}_detail_9_7.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: forecast_{col.replace(' ', '_').replace('(', '').replace(')', '')}_detail_9_7.png")
    plt.close()

print("\nAll 8 plots generated and saved successfully!")

# Save detailed results to CSV with all metrics
results_data = {
    'timestamp': test_series_original.time_index
}

# Store metrics summary
metrics_summary = []

for i, col in enumerate(value_cols):
    test_col = test_series_original.univariate_component(i)
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
results_df.to_csv('forecast_results_9_7_2025_aligned.csv', index=False)
print("\nDetailed results saved to 'forecast_results_9_7_2025_aligned.csv'")

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('metrics_summary_9_7_2025_aligned.csv', index=False)
print("Metrics summary saved to 'metrics_summary_9_7_2025_aligned.csv'")

shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv('shapiro_wilk_test_results_aligned.csv', index=False)
print("Shapiro-Wilk test results saved to 'shapiro_wilk_test_results_aligned.csv'")

print("\n" + "="*40)
print("SAMPLE PREDICTIONS FOR 9/7/2025")
print("="*40)
print(results_df.head(20).to_string(index=False))
print("="*40)

print("\n" + "="*40)
print("METRICS SUMMARY TABLE")
print("="*40)
print(metrics_df.to_string(index=False))
print("="*40)

print("\n" + "="*40)
print("SHAPIRO-WILK TEST SUMMARY")
print("="*40)
print(shapiro_df.to_string(index=False))
print("="*40)