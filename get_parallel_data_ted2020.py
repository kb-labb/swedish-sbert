"""
This scripts downloads automatically the TED2020 corpus: https://github.com/UKPLab/sentence-transformers/blob/master/docs/datasets/TED2020.md
This corpus contains transcripts from TED and TEDx talks, translated to 100+ languages. 
For downloading other parallel data, see get_parallel_data_{}.py scripts
"""
import os
import sentence_transformers.util
import csv
import gzip
from tqdm.autonotebook import tqdm

dev_sentences = 1000

# Define the language codes you would like to extend the model to
source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
target_languages = set(
    ["sv"]
)  # We want to extend the model to these new languages. For language codes, see the header of the train file

# This function downloads a corpus if it does not exist
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
sts_corpus = "datasets/STS2017-extended.zip"  # Extended STS2017 dataset for more languages
parallel_sentences_folder = "parallel-sentences/"

# Check if the file exists. If not, they are downloaded
download_corpora([train_corpus, sts_corpus])


# Create parallel files for the selected language combinations
os.makedirs(parallel_sentences_folder, exist_ok=True)
train_files = []
dev_files = []
files_to_create = []
for source_lang in source_languages:
    for target_lang in target_languages:
        output_filename_train = os.path.join(
            parallel_sentences_folder,
            "TED2020-{}-{}-train.tsv.gz".format(source_lang, target_lang),
        )
        output_filename_dev = os.path.join(
            parallel_sentences_folder, "TED2020-{}-{}-dev.tsv.gz".format(source_lang, target_lang)
        )
        train_files.append(output_filename_train)
        dev_files.append(output_filename_dev)
        if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
            files_to_create.append(
                {
                    "src_lang": source_lang,
                    "trg_lang": target_lang,
                    "fTrain": gzip.open(output_filename_train, "wt", encoding="utf8"),
                    "fDev": gzip.open(output_filename_dev, "wt", encoding="utf8"),
                    "devCount": 0,
                }
            )

if len(files_to_create) > 0:
    print(
        "Parallel sentences files {} do not exist. Create these files now".format(
            ", ".join(map(lambda x: x["src_lang"] + "-" + x["trg_lang"], files_to_create))
        )
    )
    with gzip.open(train_corpus, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line in tqdm(reader, desc="Sentences"):
            for outfile in files_to_create:
                src_text = line[outfile["src_lang"]].strip()
                trg_text = line[outfile["trg_lang"]].strip()

                if src_text != "" and trg_text != "":
                    if outfile["devCount"] < dev_sentences:
                        outfile["devCount"] += 1
                        fOut = outfile["fDev"]
                    else:
                        fOut = outfile["fTrain"]

                    fOut.write("{}\t{}\n".format(src_text, trg_text))

    for outfile in files_to_create:
        outfile["fTrain"].close()
        outfile["fDev"].close()
