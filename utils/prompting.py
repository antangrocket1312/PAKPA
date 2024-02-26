import random
import re
from tqdm import tqdm
from os import listdir
import openai
import os
import pandas as pd
import time
from pathlib import Path
import random
import re
from tqdm import tqdm

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompt")


def get_prompt(name):
    with open(os.path.join(PROMPT_DIR, name + ".txt")) as f:
        return "".join([line for line in f])


def get_absa_completion(comment, offset=1, model="vicuna-7b-v1.3"):
    absa_prompt = get_prompt("absa_few_shot") + "\n\nReview sentence: %s\n"
    comment_len = len(comment.split())
    max_tokens = int(comment_len) * 5
    response = openai.Completion.create(
        model=model,
        prompt=absa_prompt % comment,
        max_tokens=max(50, max_tokens) * offset,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


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


fix_prompt = """Fix the input text, delimited by triple quote, into the correct JSON format.
JSON input: [{'aspect': 'wraps','salads', 'burgers','mango margarita','sentiment': 'positive'}]
Corrected JSON: [{'aspect': 'wraps', 'sentiment': 'positive'}, {'aspect': 'salads', 'sentiment': 'positive'}, {'aspect': 'burgers', 'sentiment': 'positive'}, {'aspect': 'mango margarita', 'sentiment': 'positive'}]

JSON input: %s
"""


def get_fix_completion(comment, offset=1, model="vicuna-7b-v1.3"):
    max_tokens = int(len(comment) * 1.2)
    response = openai.Completion.create(
        model=model,
        prompt=fix_prompt % comment,
        max_tokens=max(50, max_tokens) * offset,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


def fix_json(comment):
    offset = 1
    answers = []
    prev_response = ""
    while len(answers) == 0:
        response = get_fix_completion(comment, offset=offset)
        response = response.replace("\_", "_")
        answers = re.findall('Corrected JSON: (\[.*\])', response)
        offset += 1
        if response == prev_response or offset >= 5:
            answers = ['[]']
            break
        prev_response = response

    corrected_json = answers[0]

    return corrected_json
