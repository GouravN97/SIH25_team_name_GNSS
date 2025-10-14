import pandas as pd

def remove_outliers_and_save(input_filepath, output_filepath, target_column='OT'):
    """
    Loads a CSV file, removes outliers from a specified column based on the 3-sigma rule,
    and saves the cleaned data to a new CSV file.

    Args:
        input_filepath (str): Path to the original CSV file.
        output_filepath (str): Path to save the new, cleaned CSV file.
        target_column (str): The name of the column to use for outlier detection.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_filepath)
        print(f"--- Processing file: {input_filepath} ---")
        initial_rows = len(df)
        print(f"Original number of rows: {initial_rows}")

        # --- Outlier Removal Logic ---
        # Calculate mean and standard deviation of the target column
        mean = df[target_column].mean()
        std = df[target_column].std()

        # Define the upper and lower bounds for the 2-sigma rule
        lower_bound = mean - 1.5 * std
        upper_bound = mean + 1.5 * std

        # Filter the DataFrame to keep only the inliers
        df_cleaned = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]

        final_rows = len(df_cleaned)
        rows_removed = initial_rows - final_rows

        print(f"Rows removed based on '{target_column}' outliers: {rows_removed}")
        print(f"Final number of rows: {final_rows}")

        # --- Save the New File ---
        # Do not overwrite the original file
        df_cleaned.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to: {output_filepath}\n")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_filepath}'. Please check the path.")
    except KeyError:
        print(f"Error: The column '{target_column}' was not found in '{input_filepath}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Define the input files and the names for the new output files
    
    # Process the original dataset
    remove_outliers_and_save(input_filepath='geodata_outliers.csv',
                             output_filepath='geodata.csv')
