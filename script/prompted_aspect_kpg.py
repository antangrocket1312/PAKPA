import argparse
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
import os
import ast
import spacy
from utils.prompting import *
# pd.set_option('display.max_colwidth', None)
import time
from multiprocessing import Pool
from tqdm import tqdm
from os import listdir
import openai
import os
import warnings

warnings.filterwarnings("ignore")


def get_single_aspect_comments(row):
    selected_cols = [col for col in row.index if type(row[col]) is list]
    comment_filtered_df = pd.DataFrame(row).T.explode(selected_cols)
    comment_filtered_df = comment_filtered_df[comment_filtered_df['num_of_aspects'] == 1]

    row['single_aspect_comments'] = comment_filtered_df['sentences'].tolist()
    row['single_aspect_comment_ids'] = comment_filtered_df['id'].tolist()

    return row


# def get_absa_completion(comment, offset=1, model="vicuna-7b-v1.3"):
def get_aspect_kpg_completion(truncated_comments, comment_aspects, model="gpt-3.5-turbo"):
    aspect_kpg_prompt = get_prompt("aspect_kpg_one_shot") + '\n\nComments: """%s"""\nAspects: """%s"""'
    prompt = aspect_kpg_prompt % (truncated_comments, comment_aspects)

    retries = 10
    while retries > 0:
        try:
            response = get_chat_completion(model, prompt, 20)
            return response
        except Exception as e:
            if e:
                if "exceeded your current quota" in str(e).lower():
                    raise e
                print(e)
                print('Timeout error, retrying...')
                retries -= 1
                if "limit reached for" in str(e).lower():
                    time.sleep(30)
                else:
                    time.sleep(5)
            else:
                raise e

    print('API is not responding, moving on...')
    return None


def prompt_aspect_kpg(root_path, domain, domain_df, save_step=10):
    src_path = f"{root_path}/{domain}"

    Path(src_path).mkdir(parents=True, exist_ok=True)
    abstractive_kps = []
    single_aspect_comments_list = []
    single_aspect_comment_ids_list = []

    # Progress recovery & continue
    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, ": ", "Loaded saved KPG file. Done")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")

        abstractive_kps = last_domain_df['generated_kp'].tolist()
        single_aspect_comments_list = last_domain_df['single_aspect_comments'].tolist()
        single_aspect_comment_ids_list = last_domain_df['single_aspect_comment_ids'].tolist()

        start = last_index
        print(domain, "Loaded saved KPG file. Continuing")
    else:
        print(domain, "Start new process.")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        new_row = get_single_aspect_comments(row)
        single_aspect_comments = new_row['single_aspect_comments']
        comment_aspects = list(sorted(set(new_row['aspects_lemm'])))
        single_aspect_comment_ids = new_row['single_aspect_comment_ids']

        single_aspect_comments_list += [single_aspect_comments]
        single_aspect_comment_ids_list += [single_aspect_comment_ids]

        if len(single_aspect_comments) == 0:
            abstractive_kps += ["Key Point: No comments provided"]
        else:
            abstractive_kp = get_aspect_kpg_completion(single_aspect_comments, comment_aspects)
            abstractive_kps += [abstractive_kp]
            time.sleep(0.5)

        if (i + 1) % save_step == 0:
            save_df = domain_df.iloc[:i + 1]
            save_df.insert(0, 'generated_kp', abstractive_kps)
            save_df['single_aspect_comments'] = single_aspect_comments_list
            save_df['single_aspect_comment_ids'] = single_aspect_comment_ids_list
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'generated_kp', abstractive_kps)
    new_domain_df['single_aspect_comments'] = single_aspect_comments_list
    new_domain_df['single_aspect_comment_ids'] = single_aspect_comment_ids_list

    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df


def explode_horizontal(sub_df):
    new_df = sub_df['comments'].explode().reset_index().drop(columns=['index']).T
    new_df.insert(0, 'coverage', len(sub_df['comments'].iloc[0]))
    new_df.insert(0, 'aspects_terms', [sub_df['aspects_lemm'].iloc[0]])
    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_comment_clusters_file", type=str, default='aspect_sentiment_clusters.pkl',
                        help="The name of the file containing comments clusters, grouped by their aspects and "
                             "sentiments, placed under 'yelp' or 'space'.")
    parser.add_argument("--output_file_name", type=str, default='kpg_summaries.pkl',
                        help="The name of the output file containing the final KP summaries.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset for inference. Either 'yelp' or 'space'")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo',
                        help="The LLM of OpenAI used for KPG")
    parser.add_argument("--openai_api_key", type=str, required=True,
                        help="The API key, use for prompting OpenAI's endpoint models for KPG")
    parser.add_argument("--num_workers", type=int, default=5, help="The number of workers for the KPG task")

    args = parser.parse_args()
    input_comment_clusters_file = args.input_comment_clusters_file
    output_file_name = args.output_file_name
    dataset = args.dataset
    model = args.model  # Threshold for merging similar aspect terms into clusters
    openai_api_key = args.openai_api_key
    num_workers = args.num_workers

    # # Read Aspect Sentiment Comment Clusters
    df = pd.read_pickle(f"./data/{dataset}/{input_comment_clusters_file}")
    if dataset == 'space':
        df['category'] = "Hotels"
        df = df.rename(columns={'entity_id': 'business_id', 'entity_name': 'business_name'})
        df['comment_ids'] = df['id']
    elif dataset == 'yelp':
        df = df.drop(columns=['categories_list'])

    root_path = f"./data/yelp/kpg_cache"
    inputs = [(root_path,
               domain,
               df[df['category'] == domain].reset_index(drop=True)
               )
              for domain in df['category'].unique()]
    start_time = time.time()
    with Pool(num_workers) as processor:
        data = processor.starmap(prompt_aspect_kpg, inputs)
    print("TIME ELAPSED", time.time() - start_time)
    processed_summ_df = pd.concat(data)

    # ## Post-processing
    processed_summ_df = processed_summ_df.rename(columns={
        'truncated_comments': 'single_aspect_comments',
        'truncated_comment_ids': 'single_aspect_comment_ids'
    })
    if dataset == 'space':
        processed_summ_df = processed_summ_df.rename(columns={
            'entity_id': 'business_id',
            'entity_name': 'business_name'
        })

    import re

    # Extracting KPs from the LLM response
    processed_summ_df['generated_kp'] = processed_summ_df['generated_kp'].apply(
        lambda x: re.findall("(.+: )*(.+\.*)\n*", x)[-1][-1])

    # Save the final KP summaries
    processed_summ_df.to_pickle(f"./data/{dataset}/{output_file_name}")

    # ## Output CSV
    processed_summ_df['aspects_lemm_comments'] = processed_summ_df['aspects_lemm']
    processed_summ_df = processed_summ_df.groupby(
        ['topic', 'business_id', 'business_name', 'cluster_sentiment', 'generated_kp']).agg(
        {'aspects_lemm_comments': lambda x: np.hstack(x.tolist()),
         'aspects_lemm': lambda x: list(dict.fromkeys(np.hstack(x.tolist()))),
         'comments': lambda x: list(dict.fromkeys(np.hstack(x.tolist()))),
         'single_aspect_comments': lambda x: list(dict.fromkeys(np.hstack(x.tolist()))),
         }).reset_index()
    kp_df = processed_summ_df.groupby(
        ['topic', 'business_id', 'business_name', 'cluster_sentiment', 'generated_kp']).apply(
        explode_horizontal).reset_index().drop(columns=['level_5'])
    kp_df = kp_df.sort_values(by=['topic', 'business_name', 'cluster_sentiment', 'coverage'],
                              ascending=[True, True, True, False])
    kp_df.to_csv(f"./data/{dataset}/{output_file_name}".replace("pkl", "csv"), index=False)
