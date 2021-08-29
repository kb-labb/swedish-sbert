import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

df = pd.read_csv("supersim_superlim/data/gold_similarity.tsv", sep="\t")
df = df.rename(columns=lambda x: x.strip())  # Strip whitespace from column headers

# Get embeddings and calculate similarities
word1_embeddings = model.encode(df["Word 1"].tolist(), normalize_embeddings=True)
word2_embeddings = model.encode(df["Word 2"].tolist(), normalize_embeddings=True)

similarity_matrix = word1_embeddings @ word2_embeddings.T
similarities = np.diag(similarity_matrix)

df["similarities"] = similarities

print(df[["Average", "similarities"]].corr(method="spearman"))
