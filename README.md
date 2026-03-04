# PCL Classifier

## Task Name

**SemEval 2022 Task 4, Subtask 1**: Binary Classification of Patronising and Condescending Language (PCL)

## Task Purpose

Classify text paragraphs as either **PCL** (1) or **No PCL** (0). PCL is language that is patronising or condescending towards vulnerable communities (e.g. refugees, homeless people, poor families). It is often subtle and used unconsciously, making it a challenging NLP task. The goal is to build a model that outperforms the RoBERTa-base baseline (F1 = 0.48 on dev, 0.49 on test).

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `1_literature/` | Coursework spec and PCL research paper |
| `2_eda/` | Exploratory data analysis notebook and figures |
| `3_baseline/` | RoBERTa-base baseline notebook |
| `4_model/` | Model training notebook, helper module, SavedModels |
| `5_evaluation/` | Error analysis and local evaluation |
| `BestModel/` | Best model + code for submission |
| `data/` | Train/dev splits and data files |
| `dev.txt` | Dev set predictions (one 0 or 1 per line) |
| `test.txt` | Test set predictions (one 0 or 1 per line) |

## Steps to Achieve the Task

1. **Data acquisition** — Download the Don't Patronize Me! dataset (`dontpatronizeme_pcl.tsv`) and the official train/dev/test splits.
2. **Exploratory data analysis** — Analyse the dataset for class balance, linguistic patterns, and noise.
3. **Data preprocessing** — Clean the data and prepare it for model training.
4. **Baseline** — Train or evaluate the RoBERTa-base baseline as a reference.
5. **Propose and implement approach** — Design a strategy to beat the baseline (e.g. fine-tuning, data augmentation, architecture changes).
6. **Train model** — Implement and train your best model.
7. **Evaluate** — Generate predictions for dev and test sets; perform error analysis.
8. **Submit** — Produce `dev.txt` and `test.txt` with one prediction (0 or 1) per line.
