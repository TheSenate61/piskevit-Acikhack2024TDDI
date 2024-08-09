from datasets import load_dataset
from deep_translator import GoogleTranslator
import pandas as pd
from tqdm import tqdm

# Load the dataset
dataset = load_dataset('PKU-Alignment/PKU-SafeRLHF-30K')  # replace with the actual dataset name

# Initialize the translator
translator = GoogleTranslator(source='en', target='tr')

def translate_text(text):
    return translator.translate(text)

def translate_rows(rows, columns):
    translated_rows = []
    for row in tqdm(rows, desc="Translating rows"):
        translated_row = row.copy()  # Create a copy to avoid modifying the original row
        for column in columns:
            if column in translated_row and translated_row[column]:
                translated_row[column] = translate_text(translated_row[column])
        translated_rows.append(translated_row)
    return translated_rows

def translate_and_save_filtered_rows(dataset, start_row=300, n=20000, save_interval=100):
    # Define the columns to translate
    columns_to_translate = ['prompt', 'response_0', 'response_1']
    
    # Translate train and test sets
    for split in ['train', 'test']:
        if split in dataset:
            split_data = dataset[split]
            
            # Select the rows from start_row to start_row + n
            data_list = split_data.select(range(start_row, min(start_row + n, len(split_data)))).to_pandas()

            # Filter out rows where both safety columns are False
            filtered_data = data_list[~((data_list['is_response_0_safe'] == False) & (data_list['is_response_1_safe'] == False))]

            # Convert the filtered rows to a list of dictionaries
            filtered_rows = filtered_data.to_dict(orient='records')

            # Initialize a list to hold translated rows
            all_translated_rows = []

            # Translate and save the filtered rows in chunks
            for i in tqdm(range(0, len(filtered_rows), save_interval), desc="Processing in chunks"):
                chunk = filtered_rows[i:i + save_interval]
                translated_chunk = translate_rows(chunk, columns_to_translate)
                all_translated_rows.extend(translated_chunk)
                
                # Convert the list of translated rows to a DataFrame
                translated_df = pd.DataFrame(translated_chunk)
                
                # Save the translated DataFrame to a CSV file
                translated_df.to_csv(f"translated_{split}_filtered_rows_part_{i//save_interval + 82}.csv", index=False)
            
            # Optionally save the complete translated DataFrame
            all_translated_df = pd.DataFrame(all_translated_rows)
            all_translated_df.to_csv(f"translated_{split}_filtered_rows_complete.csv", index=False)

# Translate the filtered rows and save as CSV files
translate_and_save_filtered_rows(dataset, start_row=8000, n=20000, save_interval=100)
