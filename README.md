# PCL Classifier — SemEval 2022 Task 4, Subtask 1

Binary classification of Patronising and Condescending Language (PCL) towards vulnerable communities.

---

## For Markers: Key Files at a Glance

| What you need | Where to find it |
|---|---|
| **Dev set predictions** (`dev.txt`) | [`dev.txt`](dev.txt) — root of repo, 2,094 lines |
| **Test set predictions** (`test.txt`) | [`test.txt`](test.txt) — root of repo, 3,832 lines |
| **Best model code** | [`BestModel/model.ipynb`](BestModel/model.ipynb) |
| **Best model weights (RoBERTa)** | [`BestModel/roberta-base/`](BestModel/roberta-base/) |
| **Best model weights (DistilBERT)** | [`BestModel/distilbert/`](BestModel/distilbert/) |
| **EDA notebook** | [`2_eda/eda.ipynb`](2_eda/eda.ipynb) |
| **Baseline notebook** | [`3_baseline/RoBERTa_baseline.ipynb`](3_baseline/RoBERTa_baseline.ipynb) |
| **Error analysis & evaluation** | [`5_evaluation/evaluation.ipynb`](5_evaluation/evaluation.ipynb) |

`dev.txt` and `test.txt` each contain one prediction per line (`0` = No PCL, `1` = PCL).

---

## Repository Structure

```
pcl-classifier/
├── dev.txt                    # Dev set predictions (2,094 lines, one 0/1 per line)
├── test.txt                   # Test set predictions (3,832 lines, one 0/1 per line)
│
├── BestModel/
│   ├── model.ipynb            # Full training pipeline for the submitted ensemble
│   ├── roberta-base/          # Saved weights — roberta-base (drop0-bs4ga4, seed=7)
│   └── distilbert/            # Saved weights — distilbert-base-uncased (drop03-bs8ga2, seed=7)
│
├── data/
│   ├── dontpatronizeme_pcl.tsv           # Full annotated corpus (10,469 examples)
│   ├── train_semeval_parids-labels.csv   # Official train split IDs + task-2 category labels
│   ├── dev_semeval_parids-labels.csv     # Official dev split IDs + task-2 category labels
│   ├── task4_test.tsv                    # Unlabelled official test set (3,832 rows)
│   └── dontpatronizeme_categories.tsv   # Fine-grained PCL category annotations
│
├── 1_literature/
│   ├── NLP_Specification.pdf
│   └── DontPatronizeMe_ResearchPaper.pdf
│
├── 2_eda/
│   ├── eda.ipynb              # Exploratory data analysis (Exercise 2)
│   └── figures/               # All EDA plots saved here (class distribution, token lengths, n-grams, etc.)
│
├── 3_baseline/
│   └── RoBERTa_baseline.ipynb # Official baseline reproduction (F1 = 0.48)
│
├── 4_model/
│   ├── model.ipynb            # Full model experimentation and hyperparameter search
│   └── dont_patronize_me.py   # Data loading helper used by all notebooks
│
├── 5_evaluation/
│   ├── evaluation.ipynb       # Local evaluation and error analysis (Exercise 5.2)
│   ├── figures/               # Saved plots and CSVs (confusion matrix, PR curve, per-keyword metrics, etc.)
│   └── cache/                 # Cached model probability arrays (speeds up re-runs)
│
└── SavedModels/               # All intermediate multi-seed checkpoints from hyperparameter search
    ├── hyperparam_search.log  # Full training log for all runs
    ├── roberta-base/          # roberta-base seeds 42, 7, 123
    ├── roberta-drop0-bs4ga4/  # Best RoBERTa config — seeds 42, 7, 123
    ├── roberta-drop0-bs8ga2/
    ├── roberta-drop05-bs4ga4/
    ├── roberta-drop05-bs8ga2/
    ├── distilbert-drop01-bs4ga4/
    ├── distilbert-drop01-bs8ga2/
    ├── distilbert-drop03-bs8ga2/ # Best DistilBERT config — seeds 42, 7, 123
    └── deberta-base/          # DeBERTa experiments (not used in final submission)
```

---

## Model Approach

### Problem and Baseline

Binary classification with a severe class imbalance: only ~9.5% of paragraphs (794 out of 8,375 training examples) are labelled PCL (`orig_label >= 2`). The official RoBERTa-base baseline achieves F1 = 0.48 on the dev set.

### Proposed Approach: Dual-Backbone Ensemble

The submitted system is a **50/50 weighted probability ensemble** of `roberta-base` and `distilbert-base-uncased`, both fine-tuned with the following techniques:

**1. Input Enrichment via Metadata Prepending**  
Each paragraph is prepended with its keyword and country using special tokens:
```
<e>keyword</e> <e>country</e> {paragraph text}
```
This gives the model lightweight contextual signals about the targeted community and geopolitical setting without changing the architecture.

**2. Weighted Random Sampler (WRS)**  
A `WeightedRandomSampler` oversamples PCL examples in every batch to counteract the 9.5:1 class imbalance and prevent the model collapsing to the majority class.

**3. Grouped Layer-wise Learning Rate Decay (LLRD)**  
Lower transformer layers (encoding general linguistic knowledge) receive smaller learning rates; upper layers and the classifier head receive higher rates. This preserves pre-trained representations while letting task-specific layers adapt more aggressively.

**4. Cosine Annealing Scheduler with Linear Warmup**  
Linear warmup over the first ~6% of training steps followed by cosine decay, promoting stable early training and smooth convergence.

**5. Multi-Seed Training + Per-Seed Threshold Tuning**  
Each backbone was trained with three random seeds (42, 7, 123). The decision threshold was tuned per seed on an internal validation set (stratified 10% hold-out of the official train set) to maximise F1 on the positive class. The best checkpoints for both models correspond to **seed = 7**.

**6. Ensemble via Weighted Probability Averaging**  
The class-1 softmax probabilities from the best RoBERTa and DistilBERT checkpoints are averaged with equal weights (0.50 / 0.50). A threshold of 0.50 is applied to the averaged score to produce the final binary prediction.

### Final Results

| System | Val F1 | Dev Set F1 | Threshold |
|---|---|---|---|
| Solo RoBERTa (drop0-bs4ga4, seed=7) | 0.5795 | 0.5955 | 0.75 |
| Solo DistilBERT (drop03-bs8ga2, seed=7) | 0.5818 | 0.5463 | 0.50 |
| **Ensemble (w=0.50/0.50)** | **0.5882** | **0.6069** | **0.50** |

The ensemble outperforms both individual models and was selected as the final submission.

### Results vs. Baseline

| Set | Baseline F1 | Submitted F1 | Improvement |
|---|---|---|---|
| Dev (official) | 0.48 | **0.6069** | +0.1269 |
| Test (official, hidden) | 0.49 | TBD (leaderboard) | — |
