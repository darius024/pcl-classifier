# PCL Classifier — SemEval 2022 Task 4, Subtask 1

Binary classification of Patronising and Condescending Language (PCL) towards vulnerable communities in news articles.

---

## For Markers: Key Files at a Glance

| What you need | Where to find it |
|---|---|
| **Dev set predictions** (`dev.txt`) | [`dev.txt`](dev.txt) — root of repo, 2,094 lines |
| **Test set predictions** (`test.txt`) | [`test.txt`](test.txt) — root of repo, 3,832 lines |
| **Best model code** | [`BestModel/model.ipynb`](BestModel/model.ipynb) |
| **Best model weights** | [`BestModel/roberta-base/`](BestModel/roberta-base/) and [`BestModel/distilbert/`](BestModel/distilbert/) |
| **EDA notebook** | [`2_eda/eda.ipynb`](2_eda/eda.ipynb) |
| **Baseline notebook** | [`3_baseline/RoBERTa_baseline.ipynb`](3_baseline/RoBERTa_baseline.ipynb) |
| **Error analysis** | [`5_evaluation/evaluation.ipynb`](5_evaluation/evaluation.ipynb) |

`dev.txt` and `test.txt` each contain one prediction per line (`0` = No PCL, `1` = PCL).

---

## Repository Structure

```
pcl-classifier/
├── dev.txt                   # Dev set predictions (2,094 lines)
├── test.txt                  # Test set predictions (3,832 lines)
│
├── BestModel/
│   ├── model.ipynb           # Full training pipeline for the submitted model
│   ├── roberta-base/         # Saved weights for RoBERTa backbone
│   └── distilbert/           # Saved weights for DistilBERT backbone
│
├── data/
│   ├── dontpatronizeme_pcl.tsv       # Full annotated dataset
│   ├── train_semeval_parids-labels.csv
│   ├── dev_semeval_parids-labels.csv
│   ├── task4_test.tsv               # Unlabelled official test set
│   └── dontpatronizeme_categories.tsv
│
├── 1_literature/
│   ├── NLP_Specification.pdf
│   └── DontPatronizeMe_ResearchPaper.pdf
│
├── 2_eda/
│   ├── eda.ipynb             # Exploratory data analysis (Exercise 2)
│   └── figures/              # Saved EDA plots
│
├── 3_baseline/
│   └── RoBERTa_baseline.ipynb
│
├── 4_model/
│   ├── model.ipynb           # Model experimentation notebook
│   └── dont_patronize_me.py  # Data loading helper (used by all notebooks)
│
└── 5_evaluation/
    ├── evaluation.ipynb      # Error analysis and local evaluation (Exercise 5.2)
    ├── figures/              # Confusion matrices, PR curves, per-keyword metrics
    └── cache/                # Cached probability outputs for faster re-runs
```

---

## Model Approach

### Problem and Baseline

The task is a binary classification problem with a severe class imbalance: only ~9.5% of paragraphs in the training set are labelled PCL (`orig_label >= 2`). The official RoBERTa-base baseline achieves F1 = 0.48 on the dev set and 0.49 on the test set.

### Proposed Approach: Dual-Backbone Ensemble

The submitted system is an **ensemble** of two fine-tuned transformers:

- **RoBERTa-base** (dropout=0.0, batch size 4, gradient accumulation 4, seed=7)
- **DistilBERT-base-uncased** (dropout=0.3, batch size 8, gradient accumulation 2, seed=7)

Predictions are combined via **weighted probability averaging** (50% RoBERTa, 50% DistilBERT) with a decision threshold of 0.50.

**Core design decisions:**

1. **Input enrichment via metadata prepending**  
   Each paragraph is prepended with its associated keyword and country: `<e>keyword</e> <e>country</e> text`. This gives the model contextual signals about the targeted vulnerable community.

2. **Weighted Random Sampler (WRS)**  
   To address the 9.5:1 class imbalance, a `WeightedRandomSampler` oversamples PCL examples in every batch, preventing the model from collapsing to the majority class.

3. **Grouped Layer-wise Learning Rate Decay (LLRD)**  
   Lower transformer layers receive a smaller learning rate to preserve pre-trained representations; upper layers and the classifier head adapt more aggressively.

4. **Cosine annealing with linear warmup**  
   Warmup over ~6% of total steps, then cosine decay for stable convergence.

5. **Early stopping on validation F1**  
   A stratified 10% hold-out of the training set is used for validation. Training stops when validation F1 does not improve for 3 consecutive epochs.

6. **Multi-seed training + threshold tuning**  
   Each backbone was trained with three seeds (42, 7, 123). The checkpoint with the best held-out dev F1 was selected. The decision threshold was tuned on the validation set to maximise F1 of the positive class.

### Results

| System | Val F1 | Dev Set F1 | Threshold |
|---|---|---|---|
| Solo RoBERTa (seed=7) | 0.5795 | 0.5955 | 0.75 |
| Solo DistilBERT (seed=7) | 0.5818 | 0.5463 | 0.50 |
| **Ensemble (w=0.50/0.50)** | **0.5882** | **0.6069** | **0.50** |

The ensemble outperforms both solo models by combining complementary strengths and reducing variance.

### Results vs. Baseline

| Set | Baseline F1 | Submitted F1 |
|---|---|---|
| Dev (official) | 0.48 | **0.6069** (+0.1269) |
| Test (official, hidden) | 0.49 | TBD (leaderboard) |

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers` (Hugging Face)
- `torch`, `pandas`, `numpy`, `scikit-learn`, `seaborn`, `tqdm`

For full reproducibility, the notebook uses `4_model/dont_patronize_me.py` for data loading; ensure it is on the Python path when running from other directories.
