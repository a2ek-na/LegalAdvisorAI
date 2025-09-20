import pandas as pd
import os
import re
import json
import sys

def cleanText(text):
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
    #Importing Paths.
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    # __file__ -> name of the Root Folder
    projectRoot = os.path.dirname(scriptDir)
    rawDataPath = os.path.join(projectRoot, 'data', 'raw', 'ipc_sections.csv')
    processedDataPath = os.path.join(projectRoot, 'data', 'processed')
    outputJsonPath = os.path.join(processedDataPath, 'ipc_corpus.json')

    print(f"Loading raw data from: {rawDataPath}")
    try:
        df = pd.read_csv(rawDataPath)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {rawDataPath}")
        sys.exit(1)

    # NEW STEP: Remove Duplicate Sections ---
    print(f"Original record count: {len(df)}")
    df.drop_duplicates(subset=['Section'], keep='first', inplace=True)
    print(f"Record count after removing duplicates: {len(df)}")


    print("Applying text cleaning and preprocessing...")
    df['cleaned_description'] = df['Description'].apply(cleanText)
    df['Offense'] = df['Offense'].apply(cleanText)
    # df['Punishment'] = df['Punishment'].apply(cleanText)

    print("Combining text fields to create the corpus...")
    df['corpus_text'] = df['cleaned_description'] + " " + df['Offense']

    print("Structuring data for JSON output...")
    corpus_df = pd.DataFrame({
        'id': df['Section'],
        'text': df['corpus_text']
    })

    corpus_list = corpus_df.to_dict(orient='records')

    os.makedirs(processedDataPath, exist_ok=True)
    
    print(f"Saving processed corpus to: {outputJsonPath}")
    try:
        with open(outputJsonPath, 'w', encoding='utf-8') as f:
            json.dump(corpus_list, f, indent=4, ensure_ascii=False)
        print("Successfully created 'ipc_corpus.json'.")
        print(f"Total records processed: {len(corpus_list)}")
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()