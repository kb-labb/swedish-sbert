## Swedish Sentence BERT

Swedish sentence BERT (KB-SBERT) was trained to emulate a strong English sentence embedding model called [paraphrase-mpnet-base-v2](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models). The model achieved a Pearson correlation coefficient of 0.918 and a Spearman's rank correlation coefficient of 0.911 on the SweParaphrase test set. 

An article explaining the data and the model in further detail can be found on the [KBLab blog](https://kb-labb.github.io/posts/2021-08-23-a-swedish-sentence-transformer/).

### Training 

We trained on 14.6 million sentences from different parallel corpus sources. Model was trained for 50 hours on a single A100 GPU with 40GB memory. 

To replicate the training, first download data by running all `get_parallel_data_{}.py` files. Make sure all resulting data files are created correctly in the folder `parallel-sentences/`. 

Training should simply be a matter of running 

```{python}
python train.py
```

Adjust the variables in the script if GPU memory is an issue (lower `batch_size` and `max_seq_length`). 

A training script for launching the training as a SLURM job is available in `slumrp_gpu.sh`. 

### Evaluation

The model was evaluated on 4 different SuperLim test sets. Run code in each of the `evaluate_{}.py` scripts to recreate the results. If you train your own model, you can load that model by pointing to the model directory, e.g:

```{python}
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("output/make-multilingual-en-sv-2021-08-05_19-59-51")
```

Download the data for evaluation here: https://spraakbanken.gu.se/en/resources/superlim 

### Acknowledgements

Big thanks to the [`sentence-transformers`](https://www.sbert.net/) package. Code was adapted and modified from their [training examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/multilingual).

We gratefully acknowledge the HPC RIVR consortium ([www.hpc-rivr.si](www.hpc-rivr.si)) and EuroHPC JU ([eurohpc-ju.europa.eu](eurohpc-ju.europa.eu)) for funding this research by providing computing resources of the HPC system Vega at the Institute of Information Science ([www.izum.si](www.izum.si)).