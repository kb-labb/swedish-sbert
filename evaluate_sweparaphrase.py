from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv(
    "sweparaphrase_dev.tsv",
    sep="\t",
)

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

sentences1 = df["Sentence 1"].tolist()
sentences2 = df["Sentence 2"].tolist()

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine similarity after normalizing
embeddings1 /= embeddings1.norm(dim=-1, keepdim=True)
embeddings2 /= embeddings2.norm(dim=-1, keepdim=True)

cosine_scores = embeddings1 @ embeddings2.t()
sentence_pair_scores = cosine_scores.diag()

df["model_score"] = sentence_pair_scores.cpu().tolist()
print(df[["Score", "model_score"]].corr(method="spearman"))
print(df[["Score", "model_score"]].corr(method="pearson"))
