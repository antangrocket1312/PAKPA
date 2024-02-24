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
from pathlib import Path
import random
import re
from tqdm import tqdm

absa_prompt = """You will be provided with a review sentence delimited by triple quotes.
A review sentence usually covers the customer opinions expressed on different aspects of a product or service.

You were tasked to perform Aspect-based Sentiment Analysis to extract the user sentiments expressed on different aspects in the review.
Formally, we define subtask of extracting the aspects it corresponding sentiments as Aspect Extraction and Aspect Sentiment Classification:
- Aspect Extraction: Identifying aspect targets in opinionated text, i.e., in detecting the specific aspects of a product or service the opinion holder is either praising or complaining about. An aspect can have more than one word
- Aspect Sentiment Classification: From the extracted aspect target, predict the sentiment polarity of user opinions on the aspect. The sentiment polarity value can be: "positive", "neutral", and "negative".

Provide the answer in JSON format with the following keys: aspect, sentiment

Review sentence: \"\"\"Movies cost $ 14 , and there is no student discount at this location .\"\"\"
Answer: [{'aspect': 'student discount', 'sentiment': 'negative'}]

Review sentence: \"\"\"Our tour guide was knowledgeable about the property and about all things Frank Lloyd Wright .\"\"\"
Answer: [{'aspect': 'tour guide', 'sentiment': 'positive'}]

Review sentence: \"\"\"BMW Henderson made my purchase easy and stress free .\"\"\"
Answer: [{'aspect': 'purchase', 'sentiment': 'positive'}]

Review sentence: \"\"\"I had a male therapist and he was amazing !\"\"\"
Answer: [{'aspect': 'male therapist', 'sentiment': 'positive'}]

Review sentence: \"\"\"We were pleasantly surprised with how exceptional our stay was at this hotel on a Saturday night .\"\"\"
Answer: [{'aspect': 'stay', 'sentiment': 'positive'}]

Review sentence: \"\"\"The food was excellent but the portion is so small .\"\"\"
Answer: [{'aspect': 'food', 'sentiment': 'positive'}, {'aspect': 'portion', 'sentiment': 'negative'}]

Review sentence: \"\"\"The food has great taste but the price is too high and the service is super slow .\"\"\"
Answer: [{'aspect': 'food', 'sentiment': 'positive'}, {'aspect': 'price', 'sentiment': 'negative'}, {'aspect': 'service', 'sentiment': 'negative'}]

Review sentence: \"\"\"Once we retuned home , the wait for the shuttle was less than 5 minutes and got us to our car within 10 minutes max .\"\"\"
Answer: [{'aspect': 'wait', 'sentiment': 'positive'}]

Review sentence: \"\"\"Blood on master shower curtain was the disgusting top of our experience .\"\"\"
Answer: [{'aspect': 'shower curtain', 'sentiment': 'negative'}]

Review sentence: \"\"\"The hotel restaurant had a fabulous breakfast buffet .\"\"\"
Answer: [{'aspect': 'breakfast buffet', 'sentiment': 'positive'}]

Review sentence: \"\"\"What a great array of appetizers they have on their menu and between these hours all are half price .\"\"\"
Answer: [{'aspect': 'appetizers', 'sentiment': 'positive'}, {'aspect': 'price', 'sentiment': 'positive'}]

Review sentence: \"\"\"Don't let this bad location mess it up for your brand and for everyone who needs a reasonable priced hotel to stay in .\"\"\"
Answer: [{'aspect': 'location', 'sentiment': 'negative'}, {'aspect': 'prices', 'sentiment': 'positive'}]

Review sentence: \"\"\"Unfortunately, with our show tickets, we didn't have time to sample any desserts .\"\"\"
Answer: [{'aspect': 'dessert', 'sentiment': 'neutral'}]

Review sentence: \"\"\"The food did take a few extra minutes to come, but the cute waiters' jokes and friendliness made up for it .\"\"\"
Answer: [{'aspect': 'food', 'sentiment': 'neutral'}, {'aspect': 'waiters', 'sentiment': 'positive'}]

Review sentence: \"\"\"Be sure to accompany your food with one of their fresh juice concoctions .\"\"\"
Answer: [{'aspect': 'food', 'sentiment': 'neutral'}, {'aspect': 'fresh juice concoctions', 'sentiment': 'positive'}]

Review sentence: \"\"\"During busy hrs, i recommend that you make a reservation .\"\"\"
Answer: [{'aspect': 'reservation', 'sentiment': 'neutral'}]

Review sentence: \"\"\"The menu, which changes seasonally, shows both regional and international influences .\"\"\"
Answer: [{'aspect': 'menu', 'sentiment': 'neutral'}]

Review sentence: \"\"\"Our waitress had apparently never tried any of the food, and there was no one to recommend any wine.\"\"\"
Answer: [{'aspect': 'waitress', 'sentiment': 'negative'}, {'aspect': 'food', 'sentiment': 'neutral'}, {'aspect': 'wine', 'sentiment': 'neutral'}]

Review sentence: %s
"""


def get_absa_completion(comment, offset=1, model="vicuna-7b-v1.3"):
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
