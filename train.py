"""
This script contains an example on how to extend an existent sentence embedding model to Swedish.
Given a (monolingual) teacher model you would like to extend to a new language, 
which is specified by the teacher_model_name variable. 
We train a bilingual student model to imitate the English teacher model (variable student_model_name)
in Swedish.
For training, you need parallel sentence data (sentences from two or more languages that are translated and aligned). 
You need tab-seperated files (.tsv) with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further column(s) contain the corresponding translations for languages you want to extend to.
Further information can be found in the paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://arxiv.org/abs/2004.09813
"""

import os
import logging
import sentence_transformers.util
import gzip
import numpy as np
import torch.nn as nn

from sentence_transformers import SentenceTransformer, LoggingHandler, models, evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


teacher_model_name = "all-mpnet-base-v2"  # Our monolingual teacher model, we want to convert to multiple languages
student_model_name = (
    "KB/bert-base-swedish-cased"  # Multilingual base model we use to imitate the teacher model
)

max_seq_length = 384  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64  # Batch size for training
inference_batch_size = 64  # Batch size at inference
max_sentences_per_language = 3000000  # Max number of parallel sentences (per datafile) for train
train_max_sentence_length = 2600 # Maximum length (characters) for parallel training sentences

num_epochs = 2  # Train for x epochs
num_warmup_steps = 5000  # Warumup steps

num_evaluation_steps = 1000  # Evaluate performance after every xxxx steps
dev_sentences = 1500  # Number of parallel sentences to be used for development


source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
target_languages = set(["sv"])  # We want to extend the model to Swedish.
output_path = (
    "output/no-normalize-"
    + "-".join(sorted(list(source_languages)) + sorted(list(target_languages)))
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Here we define train train and dev corpora
parallel_sentences_folder = "parallel-sentences/"


# Create parallel files for the selected language combinations
os.makedirs(parallel_sentences_folder, exist_ok=True)
train_files = [
    "parallel-sentences/ccs_synthetic.tsv.gz",
    "parallel-sentences/TED2020-en-sv-train.tsv.gz",
    "parallel-sentences/Tatoeba-eng-swe-train.tsv.gz",
    # "parallel-sentences/OpenSubtitles-en-sv.tsv.gz",
    "parallel-sentences/paragraphs-en-sv.tsv.gz",
    # "parallel-sentences/JW300-en-sv.tsv.gz",
    # "parallel-sentences/DGT-TM-en-sv.tsv",
    # "parallel-sentences/ELITR-ECA-en-sv.tsv.gz",
    # "parallel-sentences/EMEA-en-sv.tsv.gz",
    "parallel-sentences/Europarl-en-sv.tsv.gz",
]
dev_files = [
    "parallel-sentences/TED2020-en-sv-dev.tsv.gz",
    "parallel-sentences/Tatoeba-eng-swe-dev.tsv.gz",
]

logger.info("Load teacher model")

# Remove l2 normalization layer from all-mpnet-base-v2
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

teacher_model = SentenceTransformer(teacher_model_name)
teacher_model[2] = Identity()

logger.info("Create student model from scratch")
word_embedding_model = models.Transformer(student_model_name, max_seq_length=max_seq_length)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# logger.info("Resume student model training")
# student_model = SentenceTransformer("output/original-en-sv-2022-12-20_16-52-40")


###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(
    student_model=student_model,
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
    use_embedding_cache=True,
)
for train_file in train_files:
    train_data.load_data(
        train_file,
        max_sentences=max_sentences_per_language,
        max_sentence_length=train_max_sentence_length,
    )

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=student_model)


#### Evaluate cross-lingual performance on different tasks #####
evaluators = []  # evaluators has a list of different evaluator classes we call periodically

for dev_file in dev_files:
    logger.info("Create evaluator for " + dev_file)
    src_sentences = []
    trg_sentences = []
    with gzip.open(dev_file, "rt", encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if splits[0] != "" and splits[1] != "":
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = evaluation.MSEEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = evaluation.TranslationEvaluator(
        src_sentences,
        trg_sentences,
        name=os.path.basename(dev_file),
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_trans_acc)


# Train the model
student_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluation.SequentialEvaluator(
        evaluators, main_score_function=lambda scores: np.mean(scores)
    ),
    epochs=num_epochs,
    warmup_steps=num_warmup_steps,
    evaluation_steps=num_evaluation_steps,
    output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 8e-6, "eps": 1e-6},
)
