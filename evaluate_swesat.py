import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

df = pd.read_csv(
    "swesat-synonyms.txt",
    sep="\t",
    header=None,
    index_col=False,
    skiprows=42,
    skipfooter=2,
    names=["id", "question_word", "A", "B", "C", "D", "E",],
)

df = pd.melt(
    df,
    id_vars=["id", "question_word"],
    value_vars=["A", "B", "C", "D", "E"],
    var_name="answer_alternative",
    value_name="answer_word",
)

df[["answer_word", "label"]] = df["answer_word"].str.split("/", 1, expand=True)
df = df.sort_values(["id", "answer_alternative"]).reset_index(drop=True)

# This label is (mistakenly) missing in SuperLim, so we correct and add it manually
df.loc[(df["id"] == "h21ba03") & (df["answer_word"] == "beslutsam"), "label"] = 0
df["label"] = df["label"].astype("int64")


# Get embeddings and calculate similarities
question_embeddings = model.encode(df["question_word"].tolist(), normalize_embeddings=True)
answer_embeddings = model.encode(df["answer_word"].tolist(), normalize_embeddings=True)

similarity_matrix = question_embeddings @ answer_embeddings.T
similarities = np.diag(similarity_matrix)

df["similarities"] = similarities

# Predict 1 for the index with maximum similarity score within a question id
pred_max_idx = df[["id", "similarities"]].groupby("id").idxmax(axis=0)
df["pred"] = 0
df.loc[pred_max_idx["similarities"], "pred"] = 1


# Subset only the correct labels in order to calculate accuracy
df_correct_labels = df[df["label"] != 0]  # There are some labels == 2 (see dataset documentation)
correct_predictions = (df_correct_labels["label"] != 0) & (df_correct_labels["pred"] == 1)
accuracy = sum(correct_predictions) / len(df_correct_labels)

print(f"SweSAT accuracy: {accuracy:.4f}")
