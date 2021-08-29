"""
This scripts downloads automatically the TED2020 corpus: https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md
This corpus contains transcripts from TED and TEDx talks, translated to 100+ languages. 
For downloading other parallel data, see get_parallel_data_{}.py scripts
"""
import os
import sentence_transformers.util

# This function downloads TED2020 if it does not exist
def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)


# Here we define train train and dev corpora
train_corpus = "datasets/ted2020.tsv.gz"  # Transcripts of TED talks, crawled 2020
