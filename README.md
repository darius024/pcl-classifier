# PCL Classifier — SemEval 2022 Task 4, Subtask 1

Binary classification of Patronising and Condescending Language (PCL) towards vulnerable communities.

---

## For Markers: Key Files at a Glance

| What you need | Where to find it |
|---|---|
| **Dev set predictions** (`dev.txt`) | [`dev.txt`](dev.txt) — root of repo, 2,094 lines |
| **Test set predictions** (`test.txt`) | [`test.txt`](test.txt) — root of repo, 3,832 lines |
| **Best model code** | [`BestModel/model.ipynb`](BestModel/model.ipynb) |
| **Best model weights** | [`BestModel/roberta-base/`](BestModel/roberta-base/) |
| **EDA notebook** | [`2_eda/eda.ipynb`](2_eda/eda.ipynb) |
| **Baseline notebook** | [`3_baseline/RoBERTa_baseline.ipynb`](3_baseline/RoBERTa_baseline.ipynb) |
| **Error analysis** | [`5_evaluation/evaluation.ipynb`](5_evaluation/evaluation.ipynb) |

`dev.txt` and `test.txt` each contain one prediction per line (`0` = No PCL, `1` = PCL).

---

## Repository Structure

```
pcl-classifier/
├── dev.txt                   <- Dev set predictions (2,094 lines)
├── test.txt                  <- Test set predictions (3,832 lines)
|
├── BestModel/
|   ├── model.ipynb           <- Full training pipeline for the submitted model
|   ├── roberta-base/         <- Saved weights of the best model (RoBERTa, submitted)
|   └── deberta-base/         <- Saved weights of the DeBERTa model (ensemble experiment)
|
├── data/
|   ├── dontpatronizeme_pcl.tsv       <- Full annotated dataset
|   ├── train_semeval_parids-labels.csv
|   ├── dev_semeval_parids-labels.csv
|   ├── task4_test.tsv                <- Unlabelled official test set
|   └── dontpatronizeme_categories.tsv
|
├── 1_literature/
|   ├── NLP_Specification.pdf
|   └── DontPatronizeMe_ResearchPaper.pdf
|
├── 2_eda/
|   ├── eda.ipynb             <- Exploratory data analysis (Exercise 2)
|   └── figures/              <- Saved EDA plots
|
├── 3_baseline/
|   └── RoBERTa_baseline.ipynb
|
├── 4_model/
|   ├── model.ipynb           <- Earlier model experimentation notebook
|   ├── dont_patronize_me.py  <- Data loading helper (used by all notebooks)
|   └── SavedModels/          <- Intermediate checkpoints from experiments
|
├── 5_evaluation/
|   ├── evaluation.ipynb      <- Error analysis and local evaluation (Exercise 5.2)
|   └── figures/              <- All figures and CSVs produced by evaluation.ipynb
|
└── SavedModels/              <- Multi-seed checkpoints (local only — excluded from git)
```

---

## Model Approach

### Problem and Baseline

The task is a binary classification problem with a severe class imbalance: only ~9.5% of paragraphs in the training set are labelled PCL (orig\_label >= 2). The official RoBERTa-base baseline achieves F1 = 0.48 on the dev set and 0.49 on the test set.

### Proposed Approach

The submitted system is a fine-tuned **`roberta-base`** model, selected after running a broader ensemble experiment. The core design decisions are:

**1. Input Enrichment via Metadata Prepending**
Each paragraph is prepended with its associated keyword and country using special entity-boundary tokens (`<e>keyword</e> <e>country</e> text`). This gives the model lightweight contextual signals about the targeted community without changing the architecture.

**2. Weighted Random Sampler (WRS)**
To address the 9.5:1 class imbalance, a `WeightedRandomSampler` is used during training so that PCL examples are oversampled in every batch. This prevents the model collapsing to the majority class.

**3. Grouped Layer-wise Learning Rate Decay (LLRD)**
Rather than applying a uniform learning rate across all transformer layers, lower layers (which encode general linguistic knowledge) are assigned lower learning rates, while the top layers and the classifier head receive higher rates. This preserves pre-trained representations while allowing task-specific layers to adapt more aggressively.

**4. Cosine Annealing Scheduler with Linear Warmup**
A warmup of 10% of total training steps followed by cosine annealing was used to stabilise early training and allow smooth convergence.

**5. Label Smoothing**
A label smoothing factor of 0.1 was applied to the cross-entropy loss on the DeBERTa variant to reduce overconfidence and improve generalisation on the minority class.

**6. Multi-seed Training + Decision Threshold Tuning**
Each backbone was trained across three random seeds (42, 7, 123). After training, the classification threshold was tuned on the internal validation set (stratified 90/10 split of the official train set) to maximise F1 of the positive class rather than defaulting to 0.5. The best RoBERTa threshold was **t = 0.60**.

**7. Ensemble Experiment (Weighted Probability Averaging)**
A weighted ensemble of `roberta-base` and `microsoft/deberta-v3-base` was evaluated. Predictions from the best seed of each model were combined via a weighted average of output probabilities. The best ensemble weight was w = 0.95 (RoBERTa) / 0.05 (DeBERTa).

### Final Decision

| System | Val F1 | Dev Set F1 | Threshold |
|---|---|---|---|
| Solo RoBERTa (seed=123) | 0.6289 | **0.5344** | 0.60 |
| DeBERTa (seed=123) | 0.5635 | 0.4327 | 0.50 |
| Ensemble (w=0.95/0.05) | 0.6289 | 0.5344 | 0.61 |

The ensemble offered no measurable improvement over solo RoBERTa at the best-found weight, so **solo RoBERTa (seed=123) was submitted**. Both `dev.txt` and `test.txt` are generated from this checkpoint.

### Results vs. Baseline

| Set | Baseline F1 | Submitted F1 |
|---|---|---|
| Dev (official) | 0.48 | **0.5344** |
| Test (official, hidden) | 0.49 | TBD (leaderboard) |
