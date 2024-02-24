import re
import numpy as np


def split_list_in_reviews_into_sentences(sampled_comment_df):
    # Find
    min_list_size = 1
    mask = sampled_comment_df['sentences'].apply(lambda s: len(re.findall("\n[-+]", s)) >= min_list_size)

    # Process
    sampled_comment_df.loc[~mask, 'sentences'] = sampled_comment_df.loc[~mask, 'sentences'].apply(lambda x: [x])
    sampled_comment_df.loc[mask, 'sentences'] = sampled_comment_df[mask]['sentences'].apply(lambda s:
                                                            [sent.strip("\n").strip()
                                                             for sent in re.split("\n[\+\-]+", s)
                                                             if len(sent.strip("\n").strip()) > 0 and
                                                             sent.strip("\n").strip()[-1] != ':'])
    sampled_comment_df = sampled_comment_df.explode(['sentences'])
    return sampled_comment_df


def split_newlines_in_reviews_into_sentences(sampled_comment_df):
    # Find
    min_consecutive_newline = 2
    mask = sampled_comment_df['sentences'].apply(
        lambda s: re.match("^.+\n{%d,}" % min_consecutive_newline, s.strip("\n").strip()) != None)

    # Process
    sampled_comment_df.loc[~mask, 'sentences'] = sampled_comment_df.loc[~mask, 'sentences'].apply(lambda x: [x])
    sampled_comment_df.loc[mask, 'sentences'] = sampled_comment_df[mask]['sentences'].apply(
        lambda s: [sent.strip("\n").strip()
                   for sent in re.split("\n{%d,}" % min_consecutive_newline, s)
                   if len(sent.strip("\n").strip()) > 0 and sent.strip("\n").strip()[-1] != ':'])
    sampled_comment_df = sampled_comment_df.explode(['sentences'])
    return sampled_comment_df


def preprocessing_comment_data(sampled_comment_df):
    preprocessed_sampled_comment_df = sampled_comment_df.copy()
    preprocessed_sampled_comment_df = split_list_in_reviews_into_sentences(preprocessed_sampled_comment_df)
    preprocessed_sampled_comment_df = split_newlines_in_reviews_into_sentences(preprocessed_sampled_comment_df)

    # Remove sent with less than 10 characters
    preprocessed_sampled_comment_df = preprocessed_sampled_comment_df[preprocessed_sampled_comment_df['sentences'].str.len() >= 10]

    return preprocessed_sampled_comment_df
