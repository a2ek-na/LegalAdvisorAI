import pandas as pd
import sys
import os

def validate_data(df):
    """
    Performs basic validation on the dataframe.
    Checks for the presence of required columns.
    """
    print("Performing initial data validation...")
    
    required_columns = ['Description', 'Offense', 'Punishment', 'Section']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: The CSV file is missing the following required columns: {missing_columns}")
        # exiting witha non-zero status code to indicate 
        sys.exit(1)
    
    print("Validation successful: All required columns are present.")
    print(f"Loaded a total of {len(df)} records from the CSV file.")

def main():
    """
    Main function to ingest and validate the raw IPC data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'ipc_sections.csv')

    print(f"Attempting to read data from: {raw_data_path}")

    try:
        ipc_df = pd.read_csv(raw_data_path)
        validate_data(ipc_df)
        print("\nFirst 5 rows of the dataset:")
        print(ipc_df.head())

    except FileNotFoundError:
        print(f"Error: The file was not found at {raw_data_path}")
        print("Please ensure 'ipc_sections.csv' is in the 'data/raw' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()