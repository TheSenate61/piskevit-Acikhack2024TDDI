import os
import pandas as pd
import pyarrow.parquet as pq

from deep_translator import GoogleTranslator
from datasets import load_dataset
from tqdm import tqdm
import nltk
import polars as pl
from rich.progress import Progress

os.environ['POLARS_MAX_THREADS'] = '4'

def df_to_parquet(df, target_dir, chunk_size=1000000, **parquet_wargs):
    """Writes pandas DataFrame to parquet format with pyarrow.

    Args:
        df: DataFrame
        target_dir: local directory where parquet files are written to
        chunk_size: number of rows stored in one chunk of parquet file. Defaults to 1000000.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for i in tqdm(range(0, len(df), chunk_size)):
        slc = df.iloc[i: i + chunk_size]
        chunk = int(i / chunk_size)
        fname = os.path.join(target_dir, f"part_{chunk:04d}.parquet")
        slc.to_parquet(fname, engine="pyarrow", **parquet_wargs)

def from_hf_dataset_into_parquets():
    print("Loading dataset...")
    dataset = load_dataset('declare-lab/CategoricalHarmfulQA', split='en')

    print('Dataset:', dataset.shape)
    df = dataset.to_pandas()
    df['question_id'] = range(0, len(df))

    print('Converting into parquet...')
    df_to_parquet(df, './data/CategoricalHarmfulQA', chunk_size=100)

def translate(text):
    def _translate(text):
        return GoogleTranslator(source='en', target='tr').translate(text)

    try:
        if len(text) > 4999:
            sentences = nltk.tokenize.sent_tokenize(text)
            first_half = ' '.join(sentences[:len(sentences) // 2])
            second_half = ' '.join(sentences[len(sentences) // 2:])
            return _translate(first_half) + _translate(second_half)
        return _translate(text)
    except KeyboardInterrupt:
        print('KeyboardInterrupt occurred. Exiting...')
        exit(1)
    except Exception as e:
        return f'<NOT-TRANSLATED>{text}'

def qa_google_translation(row):
    return {
        'question_id': row['question_id'],
        'Category': translate(row['Category']),
        'Subcategory': translate(row['Subcategory']),
        'Question': translate(row['Question'])
    }

def translate_parquet(parquet, target_path, num_proc=6):
    load_dataset('parquet', data_files=parquet, split='train', num_proc=num_proc)\
        .map(qa_google_translation, batched=False, num_proc=num_proc) \
        .to_pandas()\
        .to_parquet(target_path, engine='pyarrow')

def category_translation(categories: list[str]) -> list[str]:
    translations = [translate(category) for category in categories]
    progress.update(child_task_id, advance=0.33 * len(categories))
    return translations

def subcategory_translation(subcategories: list[str]) -> list[str]:
    translations = [translate(subcategory) for subcategory in subcategories]
    progress.update(child_task_id, advance=0.33 * len(subcategories))
    return translations

def question_translation(questions: list[str]) -> list[str]:
    translations = [translate(question) for question in questions]
    progress.update(child_task_id, advance=0.34 * len(questions))
    return translations

if __name__ == '__main__':
    from_hf_dataset_into_parquets()
    parquet_files = list(map(lambda idx: f'{idx:04d}', range(0, 6)))  # Adjust the range based on the actual number of parts

    src_format = './data/CategoricalHarmfulQA/part_{file_name}.parquet'
    tgt_format = './data/CategoricalHarmfulQA/translated/part_{file_name}.parquet'

    if not os.path.exists('./data/CategoricalHarmfulQA/translated'):
        os.makedirs('./data/CategoricalHarmfulQA/translated')

    with Progress() as progress:
        main_task_id = progress.add_task('Translating parquet files...', total=len(parquet_files))
        for file_name in parquet_files:
            src_file = src_format.format(file_name=file_name)
            if not os.path.exists(src_file):
                print(f'{src_file} not found. Skipping...')
                continue
            num_groups = pl.scan_parquet(src_file).collect().get_column('Category').unique().len()
            child_task_id = progress.add_task(f'Translating {src_file}...', total=num_groups)

            group = pl.scan_parquet(src_file)\
                .group_by('Category', maintain_order=True)\
                .agg(
                    pl.col('Category').map_elements(category_translation, return_dtype=list[str]).alias('translated_category'),
                    pl.col('Subcategory').map_elements(subcategory_translation, return_dtype=list[str]).alias('translated_subcategory'),
                    pl.col('Question').map_elements(question_translation, return_dtype=list[str]).alias('translated_question'),
                    pl.col('question_id'),
            )\
                .collect()

            # Ensure columns to be dropped exist
            columns_to_drop = [col for col in ['Category', 'Subcategory', 'Question'] if col in group.columns]
            df = group.explode(pl.exclude('Category'))\
                .drop(*columns_to_drop) \
                .rename({'translated_category': 'Category', 'translated_subcategory': 'Subcategory', 'translated_question': 'Question'})\
                .to_pandas()\
                .to_parquet(tgt_format.format(file_name=file_name), engine='pyarrow')

            progress.update(main_task_id, advance=1)

            for task in progress.tasks:
                if task.finished:
                    progress.update(task.id, visible=False)