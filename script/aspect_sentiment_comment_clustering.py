import argparse
import pandas as pd
import numpy as np
# from tqdm import tqdm
import time
# from tqdm import tqdm
import statistics
import time
import spacy
from tqdm.contrib.concurrent import process_map  # or thread_map

nlp = spacy.load('en_core_web_lg')


def cal_spacy_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

    # Merge to the cluster with the highest score


def deduplicate(inputs):
    """ Group similar aspect terms of a business in a greedy fashion."""
    # Deduplication
    buss_aspects = inputs[0]
    sent_buss_df = inputs[1]

    filtered = []
    for aspect in buss_aspects:
        find_merge = False

        similarity_to_other_clusters = []
        # Get best cluster
        for aspects_cluster in filtered:
            average_cosine = average_similarity_to_cluster(aspect, aspects_cluster, sent_buss_df)
            similarity_to_other_clusters += [average_cosine]

        sorted_cluster_indices = np.argsort(similarity_to_other_clusters)[::-1]

        if len(sorted_cluster_indices) > 0:
            optimal_cluster_index = sorted_cluster_indices[0]
            if similarity_to_other_clusters[optimal_cluster_index] >= threshold:
                aspects_other = filtered[optimal_cluster_index]
                aspects_other.append(aspect)
                find_merge = True

        if not find_merge:
            filtered.append([aspect])

    aspect_clusters_df = pd.DataFrame()
    aspect_clusters_df['aspects_lemm'] = filtered
    aspect_clusters_df = aspect_clusters_df.reset_index().explode(['aspects_lemm']).rename(
        columns={'index': 'cluster_id'})

    return filtered, sent_buss_df.merge(aspect_clusters_df, on=['aspects_lemm'])


def average_similarity_to_cluster(kp, kps_other, sent_buss_df):
    """ Calculate average cosine similarity of an AK to a cluster """
    total_similarity = []
    for kp_other in kps_other:
        total_similarity += [calculate_similarity(kp, kp_other, sent_buss_df)]

    return statistics.mean(total_similarity)


def calculate_similarity(text1, text2, sent_buss_df):
    """ Determine if two extractions are the same or not
    Args:
        other (Extraction object)
    Returns:
        True or False
    Rule:
        Consider two extractions as the same if their w2v cosine similarity
        is above the specified threshold:
            ext1 == ext2, if cosine(ext1.emb, ext2.emb) >= threshold
    """
    similarity = cal_spacy_similarity(text1, text2)
    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_reviews_absa_file", type=str, default='reviews_absa_processed.pkl',
                        help="The name of the reviews file already processed for ABSA, placed under 'yelp' or 'space'.")
    parser.add_argument("--output_file_name", type=str, default='aspect_sentiment_clusters.pkl',
                        help="The name of the output file containing comment clusters.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset for inference. Either 'yelp' or 'space'")
    parser.add_argument("--similarity_threshold", type=float, default=0.55, help="threshold for merging similar "
                                                                                 "aspect terms into clusters")
    parser.add_argument("--num_worker", type=int, default=5, help="The number of workers for the clustering task")

    args = parser.parse_args()
    input_reviews_absa_file = args.input_reviews_absa_file
    output_file_name = args.output_file_name
    dataset = args.dataset
    threshold = args.similarity_threshold # Threshold for merging similar aspect terms into clusters
    num_worker = args.num_worker

    # # Read ABSA-processed Reviews
    df = pd.read_pickle(f"./data/{dataset}/{input_reviews_absa_file}")
    df.columns = [col.replace('prompt_', '').replace('aspect', 'aspects').replace('sentiment', 'sentiments') for col in
                  df.columns]
    df = df[df['sentiments'] != 'neutral']

    # Aggregating
    col_agg = {col: lambda x: x.iloc[0] for col in df.columns if col not in ['review_id', 'user_id', 'business_id',
                                                                             'text', 'sentences',
                                                                             'aspect', 'sentiment', 'aspect_lemm']}
    sent_list_agg = {col: lambda x: x.tolist() for col in df.columns if
                     col in ['aspects', 'sentiments', 'aspects_lemm']}
    col_agg.update(sent_list_agg)
    df = df.groupby(['review_id', 'user_id', 'business_id', 'text', 'sentences'], sort=False, as_index=False).agg(
        col_agg).reset_index(drop=True)

    # Indexing
    df = df.groupby(['review_id', 'user_id', 'business_id', 'text'],
                    sort=False, as_index=False).apply(lambda grp: grp.reset_index(drop=True)).reset_index()
    df = df.rename(columns={'text': 'review_content'})
    df['id'] = df['review_id'].astype(str) + "######" + df['level_1'].astype(str)

    # Preprocessing
    df_scored = df
    df_scored['num_of_aspects'] = df_scored['aspects_lemm'].apply(lambda x: len(x))

    # # Aspect Sentiment Clustering
    sent_df = df_scored.explode(['aspects', 'sentiments', 'aspects_lemm'])
    col_agg = {col: lambda x: x.iloc[0] for col in df.columns if
               col in ['business_name', 'business_id', 'categories', 'categories_list', 'category']}
    sent_list_agg = {col: lambda x: x.tolist() for col in df.columns if
                     col not in ['cluster_id', 'business_name', 'business_id', 'categories', 'categories_list',
                                 'category']}
    col_agg.update(sent_list_agg)

    # Form input for multiprocessing
    inputs = []
    for category in sorted(df_scored['category'].unique()):
        for business_id in sorted(df_scored[df_scored['category'] == category]['business_id'].unique()):
            for sentiment in ['positive', 'negative']:
                sent_buss_df = sent_df[(sent_df['business_id'] == business_id) & (sent_df['sentiments'] == sentiment)]
                sent_buss_df = sent_buss_df[
                    sent_buss_df.apply(lambda row: row['aspects'].lower() in row['sentences'].lower(), axis=1)]

                # Sort aspects by their occurrences in the particular business
                sorted_aspects_index = sent_buss_df['aspects_lemm'].value_counts()
                buss_aspects = sorted_aspects_index.index.tolist()
                inputs += [(buss_aspects, sent_buss_df)]

    # Perform clustering
    start_time = time.time()
    clusters_info = process_map(deduplicate, inputs[:2], max_workers=num_workers)
    print("TIME ELAPSED", time.time() - start_time)

    # Process the results
    dfs = []
    for business_sentiment_cluster_info in clusters_info:
        sent_buss_clustered_df = business_sentiment_cluster_info[1]

        # Number of sentences in a cluster must be > the number of aspects"
        sent_buss_clustered_df = sent_buss_clustered_df.groupby(['cluster_id']).filter(
            lambda grp: len(grp) > len(grp['aspects_lemm'].unique()))

        # Get the final clustered df of comments by aspects
        aspect_clusters_df = sent_buss_clustered_df.groupby(['cluster_id']).agg(col_agg)
        aspect_clusters_df['cluster_sentiment'] = aspect_clusters_df['sentiments'].iloc[0][0]
        dfs += [aspect_clusters_df]

    summ_df = pd.concat(dfs)
    summ_df.to_pickle(f"./data/{dataset}/{output_file_name}.pkl")
