from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv(
    "sweparaphrase-dev-165.csv",
    sep="\t",
    header=None,
    names=[
        "original_id",
        "source",
        "type",
        "sentence_swe1",
        "sentence_swe2",
        "score",
        "sentence1",
        "sentence2",
    ],
)

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

sentences1 = df["sentence_swe1"].tolist()
sentences2 = df["sentence_swe2"].tolist()

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine similarity after normalizing
embeddings1 /= embeddings1.norm(dim=-1, keepdim=True)
embeddings2 /= embeddings2.norm(dim=-1, keepdim=True)

cosine_scores = embeddings1 @ embeddings2.t()
sentence_pair_scores = cosine_scores.diag()

df["model_score"] = sentence_pair_scores.cpu().tolist()
print(df[["score", "model_score"]].corr(method="spearman"))
print(df[["score", "model_score"]].corr(method="pearson"))
