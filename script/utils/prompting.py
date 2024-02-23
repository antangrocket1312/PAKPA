import pandas as pd
from pathlib import Path
import random
import re
from tqdm import tqdm
from os import listdir
import openai
import os
import pandas as pd
import time


def get_completion(prompt, comment, offset=1, model="vicuna-7b-v1.3"):
    comment_len = len(comment.split())
    max_tokens = int(comment_len) * 5
#     max_tokens = int(comment_len) * 6
    messages = [{"role": "user", "content": prompt}]
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max(50, max_tokens) * offset,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


def prompt_based_absa(root_path, prompt, domain, domain_df):
    src_path = f"{root_path}/{domain}"
    Path(src_path).mkdir(parents=True, exist_ok=True)
    absa_extractions = []

    file_names = listdir(src_path)
    postfix = [re.split("[_.]", name)[1]
               for name in listdir(src_path)
               ]
    start = 0
    if 'done' in postfix:
        print(domain, "ALREADY DONE")
        new_domain_df = pd.read_pickle(f"{src_path}/{domain}_done.pkl")
        return new_domain_df
    elif len(postfix) > 0:
        last_index = max([int(idx) for idx in postfix if idx != 'done'])
        last_domain_df = pd.read_pickle(f"{src_path}/{domain}_{last_index}.pkl")
        absa_extractions = last_domain_df['absa_extractions'].tolist()
        start = last_index
        print(domain, "CONTINUING FROM ", start)
    else:
        print(domain, "START NEW PROCESS")

    for i, (_, row) in tqdm(enumerate(domain_df.iterrows()), total=domain_df.shape[0]):
        if i < start:
            continue

        #     for i, (_, row) in enumerate(domain_df.sample(5, random_state=42).iterrows()):
        comment = row['sentences']
        #         comment = comment.replace("\n", "")
        comment = re.sub('^\n+', '', comment)

        offset = 1
        answers = []
        prev_response = ""
        while len(answers) == 0:
            #             response = get_completion(prompt %(comment), comment, offset=offset)
            try:
                response = get_completion(prompt % (comment), comment, offset=offset)
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

        if (i + 1) % 5 == 0:
            save_df = domain_df.iloc[:i + 1]
            #             display(save_df)
            #             print(matching_labels_list)
            #             print(i, absa_extractions)
            save_df.insert(0, 'absa_extractions', absa_extractions)
            save_df.to_pickle(f"{src_path}/{domain}_{i + 1}.pkl")

    new_domain_df = domain_df.iloc[:i + 1]
    new_domain_df.insert(0, 'absa_extractions', absa_extractions)
    new_domain_df.to_pickle(f"{src_path}/{domain}_done.pkl")
    return new_domain_df