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


def get_completion(model, prompt, max_tokens, temperature=0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


def get_chat_completion(model, prompt, max_tokens, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


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
