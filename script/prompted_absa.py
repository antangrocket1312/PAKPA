import argparse
import pandas as pd
import json
import openai
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
from utils.preprocessing import preprocessing_comment_data
from multiprocessing import Pool
import re
import ast
import spacy

nlp = spacy.load('en_core_web_sm')
openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_reviews_file", type=str, default='input_reviews.pkl',
                        help="The name of the input reviews file placed under 'yelp' or 'space'.")
    parser.add_argument("--output_file_name", type=str, default='reviews_absa_processed.pkl',
                        help="The name of the output reviews file to be saved under 'yelp' or 'space'.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset for inference. Either 'yelp' or 'space'")

    args = parser.parse_args()
    input_reviews_file = args.input_reviews_file
    output_file_name = args.output_file_name
    dataset = args.dataset

    # # Load dataset
    df = pd.read_pickle(f"./data/{dataset}/{input_reviews_file}")

    sampled_comment_df = df.explode(['sentences'])

    # ## Preprocess
    sampled_comment_df = preprocessing_comment_data(sampled_comment_df)

    # # Inference
    num_workers = 5
    root_path = f"./data/{dataset}/processed_absa_reviews"
    if dataset == 'space':
        sampled_comment_df['category'] = 'Hotel'
    inputs = [(root_path,
               domain,
               sampled_comment_df[sampled_comment_df['category'] == domain].reset_index(drop=True)
               )
              for domain in sampled_comment_df['category'].unique()]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompt_based_absa, inputs)
    print("TIME ELAPSED", time.time() - start_time)
    processed_df = pd.concat(data)

    # ## Post-processing
    # ### Fix ' character in the response
    mask = processed_df['absa_extractions'].str.contains(": *\'([^':,]*\'+[^':,]*)+\' *,")
    processed_df.loc[mask, 'absa_extractions'] = processed_df.loc[mask, 'absa_extractions'].apply(
        lambda x: re.sub(r"(: *)\'((?:[^':,]*\'+[^':,]*)+)\'( *,)", r'\1"\2"\3', x))

    # ### Correct JSON
    mask = processed_df['absa_extractions'].str.startswith(
        "[{'aspect': 'breads', 'cakes', 'pies', 'desserts','sentiment': 'positive'}]")
    for fix_id in processed_df[mask].index.tolist():
        fix_row = processed_df.loc[fix_id]
        processed_df.loc[fix_id, 'absa_extractions'] = fix_json(fix_row['absa_extractions'])

    # ### Normalize JSON into columns
    processed_df['absa_extractions'] = processed_df['absa_extractions'].str.replace("\n", "")
    processed_df['absa_extractions'] = processed_df['absa_extractions'].apply(lambda x: ast.literal_eval(x))
    processed_df = processed_df.explode(['absa_extractions'])
    json_struct = json.loads(processed_df.to_json(orient="records"))
    processed_df = pd.json_normalize(json_struct)
    processed_df.columns = [col.replace('absa_extractions.', 'prompt_') for col in processed_df.columns]
    processed_df = processed_df.drop(columns=['absa_extractions'])
    processed_df = processed_df[pd.notnull(processed_df['prompt_aspect'])]

    # ### Standardization
    processed_df['prompt_aspect_lemm'] = processed_df['prompt_aspect'].apply(
        lambda aspect: " ".join([token.lemma_ for token in nlp(f'{aspect.lower()}')])
    )
    processed_df.to_pickle(f"./data/{dataset}/{output_file_name}")
