import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils.prompting import *
from utils.preprocessing import preprocessing_comment_data
from multiprocessing import Pool
import re
import ast
import spacy
import argparse
import pandas as pd
import json
import openai
import time

nlp = spacy.load('en_core_web_sm')
openai.api_key = "EMPTY"  # Not support yet
openai.api_base = "http://localhost:8000/v1"
model = "vicuna-7b-v1.3"


def get_absa_completion(comment, offset=1, model="vicuna-7b-v1.3"):
    absa_prompt = get_prompt("absa_few_shot") + "\n\nReview sentence: %s\n"
    prompt = absa_prompt % comment
    comment_len = len(comment.split())
    max_tokens = int(comment_len) * 5
    max_tokens = max(50, max_tokens) * offset

    return get_completion(model, prompt, max_tokens)


def prompt_based_absa(root_path, domain, domain_df, save_step=10):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    absa_extractions = []

    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, ": ", "Loaded saved ABSA file. Done")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")
        absa_extractions = last_domain_df['absa_extractions'].tolist()
        start = last_index
        print(domain, "Loaded saved ABSA file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        comment = row['sentences']
        comment = re.sub('^\n+', '', comment)

        offset = 1
        answers = []
        prev_response = ""
        while len(answers) == 0:
            try:
                response = get_absa_completion(comment, offset=offset)
            except:
                answers = ['[]']
                break
            response = response.replace("\_", "_")

            answers = re.findall('Answer: (\[.*\])', response)
            offset += 1
            #             print(offset)
            if response == prev_response or offset >= 5:
                answers = ['[]']
                break
            prev_response = response

        absa_extraction = answers[0]
        absa_extractions += [absa_extraction]

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'absa_extractions', absa_extractions)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'absa_extractions', absa_extractions)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


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
