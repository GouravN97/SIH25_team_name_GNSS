import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest
import matplotlib.pyplot as plt
import os
import subprocess
import glob
import warnings
import argparse
import platform
import sys

# --- MODULE 1: DATA PREPROCESSING ---
def preprocess_data(input_file, target_column, cleaned_output_file='geodata.csv'):
    """
    Handles all data preparation: renaming the target column to 'OT',
    removing outliers, and saving a new, cleaned file.
    """
    print("--- STEP 1: Preprocessing Data ---")
    try:
        df = pd.read_csv(input_file)
        
        # Rename the specified target column to 'OT' for the model
        df = df.rename(columns={target_column: 'OT'})
        
        # Ensure the 'date' column exists for the model
        if 'date' not in df.columns:
            # Assuming the first column is the date column if not named 'date'
            date_col_name = df.columns[0]
            df = df.rename(columns={date_col_name: 'date'})
            print(f"  - Renamed column '{date_col_name}' to 'date'.")
        
        print(f"  - Renamed target column '{target_column}' to 'OT'.")
        initial_rows = len(df)
        
        # Remove outliers on the 'OT' column
        mean = df['OT'].mean()
        std = df['OT'].std()
        lower_bound, upper_bound = mean - 1.5 * std, mean + 1.5 * std
        
        df_cleaned = df[(df['OT'] >= lower_bound) & (df['OT'] <= upper_bound)]
        final_rows = len(df_cleaned)
        
        print(f"  - Original rows: {initial_rows}")
        print(f"  - Rows after outlier removal: {final_rows} ({initial_rows - final_rows} removed)")
        
        # Save to the new file, which the model will use
        df_cleaned.to_csv(cleaned_output_file, index=False)
        print(f"  - Cleaned data saved to '{cleaned_output_file}'.\n")
        return cleaned_output_file
        
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'.")
        return None
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the input file.")
        return None

# --- MODULE 2: TRAINING AND TESTING ---
def run_model_training_and_testing(cleaned_data_file):
    """
    Executes the appropriate script (shell or batch) based on the operating system
    to train the model and generate prediction files.
    """
    print("--- STEP 2: Training and Testing Model ---")
    
    # Detect operating system
    system = platform.system()
    print(f"  - Detected OS: {system}")
    
    if system == "Windows":
        # Use batch script for Windows
        script_path = os.path.join('scripts', 'PatchTST', 'geodata8thday.bat')
        
        # Convert to absolute path for Windows
        script_path_abs = os.path.abspath(script_path)
        
        if not os.path.exists(script_path_abs):
            print(f"  - Error: Batch script not found at '{script_path_abs}'")
            print("  - Please create a geodata8thday.bat file for Windows.")
            return None
        
        print(f"  - Using batch script: {script_path_abs}")
        
        # Run the batch script with absolute path
        try:
            original_dir = os.getcwd()
            
            process = subprocess.run([script_path_abs, cleaned_data_file], 
                                   capture_output=True, 
                                   text=True, 
                                   shell=True,
                                   cwd=original_dir)
        except Exception as e:
            print(f"  - Error executing batch script: {e}")
            return None
            
    else:  # Linux, macOS, or other Unix-like systems
        # Use shell script for Unix-like systems
        script_path = os.path.join('scripts', 'PatchTST', 'geodata8thday.sh')
        
        if not os.path.exists(script_path):
            print(f"  - Error: Shell script not found at '{script_path}'")
            return None
        
        print(f"  - Using shell script: {script_path}")
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        # Run the shell script
        try:
            process = subprocess.run(['sh', script_path, cleaned_data_file], 
                                   capture_output=True, 
                                   text=True)
        except Exception as e:
            print(f"  - Error executing shell script: {e}")
            return None
    
    # Check if the process was successful
    if process.returncode != 0:
        print("  - Error during model training/testing:")
        print("  - STDERR:")
        print(process.stderr)
        if process.stdout:
            print("  - STDOUT:")
            print(process.stdout)
        
        # Check if checkpoint exists
        checkpoint_dirs = glob.glob(os.path.join('checkpoints', '*'))
        if checkpoint_dirs:
            print(f"\n  - Available checkpoint directories:")
            for d in checkpoint_dirs:
                checkpoint_file = os.path.join(d, 'checkpoint.pth')
                exists = "✓" if os.path.exists(checkpoint_file) else "✗"
                print(f"    {exists} {d}")
        
        return None
    
    print("  - Model training and prediction generation complete.\n")
    
    # Find the results directory created by the script
    results_dir = os.path.join('results', '7_1_PatchTST_custom_ftMS_sl7_ll48_pl1_dm128_nh16_el2_dl1_df256_fc1_ebtimeF_dtTrue_Final_0')
    
    if not os.path.exists(results_dir):
        print(f"  - Warning: Expected results directory not found at '{results_dir}'")
        print("  - Searching for results directories...")
        # Try to find any results directory
        results_pattern = os.path.join('results', '*', 'pred.npy')
        matching_files = glob.glob(results_pattern)
        if matching_files:
            results_dir = os.path.dirname(matching_files[0])
            print(f"  - Found results at: {results_dir}")
        else:
            print("  - No results found!")
            return None
    
    return results_dir

# --- MODULE 3: VISUALIZATION ---
def analyze_and_plot_results(results_dir, target_column='OT'):
    """
    Generates the final plot, saving it to a dedicated 'graphs' folder 
    with a sanitized filename.
    """
    print("--- STEP 3: Analyzing Results and Plotting ---")
    
    pred_file = os.path.join(results_dir, 'pred.npy')

    if not os.path.exists(pred_file):
        print(f"  - Error: 'pred.npy' not found in {results_dir}")
        return

    # Load the prediction array
    predictions_array = np.load(pred_file)
    
    # The array shape is (num_samples, pred_len, num_features)
    # We flatten it to get a single list of all predicted values
    flattened_predictions = predictions_array.flatten()
    
    # Create a Pandas DataFrame
    df1 = pd.DataFrame(flattened_predictions, columns=['Predicted'])
    
    # Save the DataFrame to a CSV file
    output_csv_path1 = os.path.join(results_dir, 'pred.csv')
    df1.to_csv(output_csv_path1, index=False)
    
    print(f"  - Successfully converted .npy file to CSV!")
    print(f"  - File saved at: {output_csv_path1}")

    # Plotting
    print("  - Generating plot...")
    
    # Define the graphs directory and create it if it doesn't exist
    graphs_dir = os.path.join('.', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot 'Predicted' values in blue
    plt.plot(df1.index, df1['Predicted'], label='Predicted', color='steelblue', linewidth=2)
    
    # Customize the plot to look professional
    plt.title(f"Predicted Values for {target_column}", fontsize=16)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value (m)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Sanitize the target_column to create a valid filename
    # Remove special characters that might cause issues
    safe_target = "".join(c for c in target_column if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe_filename = f'plot_{safe_target}.png'
    plot_save_path = os.path.join(graphs_dir, safe_filename)
    
    # Save the figure
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    
    print(f"  - Plot saved successfully to: {plot_save_path} 📈\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end pipeline for GNSS error forecasting.')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to the input CSV data file (e.g., Data_GEO_Train.csv).')
    parser.add_argument('--target_column', type=str, required=True, 
                       help="Name of the target column to predict (e.g., 'x_error (m)').")
    
    args = parser.parse_args()

    # --- RUN THE FULL PIPELINE ---
    cleaned_file = preprocess_data(args.input_file, args.target_column)
    
    if cleaned_file:
        results_directory = run_model_training_and_testing(cleaned_file)
        if results_directory:
            analyze_and_plot_results(results_directory, args.target_column)
            print("--- Program Finished Successfully! ---")
        else:
            print("--- Program Failed: Model training/testing failed. ---")
            sys.exit(1)
    else:
        print("--- Program Failed: Data preprocessing failed. ---")
        sys.exit(1)