import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

df = pd.read_csv("faq_mismatched.tsv", sep="\t")

df_list = []
for category, group in df.groupby("category_id"):
    question_embeddings = model.encode(group["question"].tolist(), normalize_embeddings=True)
    candidate_embeddings = model.encode(
        group["candidate_answer"].tolist(), normalize_embeddings=True
    )

    similarities = question_embeddings @ candidate_embeddings.T

    # Choose candidate answer with highest similarity to given question as prediction
    group["prediction"] = (
        group["candidate_answer"]
        .reset_index(drop=True)
        .reindex(np.argmax(similarities, axis=1).tolist())
        .tolist()
    )

    df_list.append(group)


df_pred = pd.concat(df_list)[["category_id", "source", "question", "correct_answer", "prediction"]]
sum(df_pred["correct_answer"] == df_pred["prediction"]) / len(df_pred)

# Accuracy per source (Försäkringskassan, Skatteverket, etc...)
for source, group in df_pred.groupby("source"):
    accuracy = sum(group["correct_answer"] == group["prediction"]) / len(group)
    print(f"{source}: {accuracy}")


group_lengths = []
for category, group in df_pred.groupby("category_id"):
    group_lengths.append(len(group))

# Expected accuracy if guessing same candidate answer for all questions within a category
expected_acc = 1 / np.mean(group_lengths)
print(f"Expected naive guess accuracy: {expected_acc:.4f}")
