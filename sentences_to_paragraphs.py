import pandas as pd
import csv
import torch
from tqdm import tqdm


df_dgt = pd.read_csv(
    "parallel-sentences/DGT-TM-en-sv.tsv",
    sep="\t",
    header=None,
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
)

df_parl = pd.read_csv(
    "parallel-sentences/Europarl-en-sv.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
)

df_emea = pd.read_csv(
    "parallel-sentences/EMEA-en-sv.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
)

df_elitr = pd.read_csv(
    "parallel-sentences/ELITR-ECA-en-sv.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
)

df_jw = pd.read_csv(
    "parallel-sentences/JW300-en-sv.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
    on_bad_lines="skip",
)

df_ted = pd.read_csv(
    "parallel-sentences/TED2020-en-sv-train.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    names=["en", "sv"],
    quoting=csv.QUOTE_NONE,
    engine="python",
    on_bad_lines="skip",
)

df = pd.concat(
    [df_dgt, df_parl, df_emea, df_elitr, df_jw, df_ted],
    keys=["dgt", "parl", "emea", "elitr", "jw", "ted"],
).reset_index(level=0)
df = df.rename(columns={"level_0": "dataset"})
df[["en", "sv"]] = df[["en", "sv"]].astype(str)
df["nr_words"] = df["en"].str.split().str.len()

sv_paragraphs = []
en_paragraphs = []
dataset_paragraphs = []

for group, df_group in df.groupby("dataset"):
    cum_words = 0
    max_words = torch.nn.init.trunc_normal_(torch.Tensor([0]), mean=270, std=110, a=60, b=330)
    max_words = int(max_words.round())
    sv_paragraph = ""
    en_paragraph = ""

    for i, row in tqdm(df_group.iterrows(), total=len(df_group)):
        cum_words += row["nr_words"]
        if row["nr_words"] > max_words:
            sv_paragraphs.append(row["sv"])
            en_paragraphs.append(row["en"])
            dataset_paragraphs.append(group)
            sv_paragraph = ""
            en_paragraph = ""
            cum_words = 0
        elif cum_words > max_words:
            sv_paragraphs.append(sv_paragraph)
            en_paragraphs.append(en_paragraph)
            dataset_paragraphs.append(group)
            sv_paragraph = ""
            en_paragraph = ""
            cum_words = 0
        else:
            sv_paragraph += " " + row["sv"]
            en_paragraph += " " + row["en"]
        if i % 50000 == 0:
            print(
                f"Nr sv paragraphs: {len(sv_paragraphs)}, nr en paragraphs: {len(en_paragraphs)}"
            )


df_long = pd.DataFrame({"en": en_paragraphs, "sv": sv_paragraphs, "dataset": dataset_paragraphs})

# NKFD normalization
df_long["en"] = df_long["en"].str.normalize("NFKD")
df_long["sv"] = df_long["sv"].str.normalize("NFKD")

# df_long.to_parquet("parallel-sentences/paragraphs-en-sv.parquet")
df_long[["en", "sv"]].to_csv(
    "parallel-sentences/paragraphs-en-sv.tsv.gz",
    sep="\t",
    header=None,
    compression="gzip",
    index=False,
)