import pandas as pd
import os
import re
import json
import sys

def clean_text(text):
    """
    Applies a sequence of cleaning operations to the input text.
    1. Converts to lowercase.
    2. Removes punctuation.
    3. Removes newline characters.
    4. Strips leading/trailing and reduces multiple internal whitespace to one.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string types (like NaN)
    
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = " ".join(text.split())
    return text

def main():
    """
    Main function to preprocess the raw data and generate the JSON corpus.
    """

    #Importing Paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'ipc_sections.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    output_json_path = os.path.join(processed_data_path, 'ipc_corpus.json')

    print(f"Loading raw data from: {raw_data_path}")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        sys.exit(1)

    # NEW STEP: Remove Duplicate Sections ---
    print(f"Original record count: {len(df)}")
    df.drop_duplicates(subset=['Section'], keep='first', inplace=True)
    print(f"Record count after removing duplicates: {len(df)}")


    print("Applying text cleaning and preprocessing...")
    df['cleaned_description'] = df['Description'].apply(clean_text)
    
    df['Offense'] = df['Offense'].apply(clean_text)
    df['Punishment'] = df['Punishment'].apply(clean_text)

    print("Combining text fields to create the corpus...")
    df['corpus_text'] = df['cleaned_description'] + " " + df['Offense'] + " " + df['Punishment']

    print("Structuring data for JSON output...")
  
    corpus_df = pd.DataFrame({
        'id': df['Section'],
        'text': df['corpus_text']
    })

    corpus_list = corpus_df.to_dict(orient='records')

    os.makedirs(processed_data_path, exist_ok=True)
    
    print(f"Saving processed corpus to: {output_json_path}")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_list, f, indent=4, ensure_ascii=False)
        print("Successfully created 'ipc_corpus.json'.")
        print(f"Total records processed: {len(corpus_list)}")
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()