"""
improved_distilbert.py
======================
Trains two improved DistilBERT variants derived from the ablation study,
then compares every combination against the existing BestModel ensemble.

Run from the BestModel/ directory:
    python improved_distilbert.py

Model A – "Best Combined"
    All four components that individually improved over the ablation baseline:
    no metadata prepending, no WRS, no grouped LLRD, no warmup.
    Rationale: every component tested alone showed positive delta; stacking
    them targets the maximum additive gain.

Model B – "High-Recall Companion"
    Designed to complement the existing RoBERTa backbone in the ensemble.
    Keeps metadata + LLRD (preserve representation quality), drops WRS +
    warmup (both boost recall), and uses lower dropout (0.1) to reduce
    over-regularisation on the tiny positive class (~794 examples).
    A high-recall partner compensates for RoBERTa's tendency to miss FNs.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, html, random, warnings, gc, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_cosine_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

warnings.filterwarnings("ignore")

# Script lives at repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
SAVE_ROOT = os.path.join(REPO_ROOT, "SavedModels")

# dont_patronize_me.py lives in 4_model/
sys.path.insert(0, os.path.join(REPO_ROOT, "4_model"))
from dont_patronize_me import DontPatronizeMe

# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading (shared across all models)
# ─────────────────────────────────────────────────────────────────────────────
dpm = DontPatronizeMe(DATA_DIR, os.path.join(DATA_DIR, "task4_test.tsv"))
dpm.load_task1()
full_df = dpm.train_task1_df.copy()
full_df["par_id"]     = full_df["par_id"].astype(str)
full_df["orig_label"] = full_df["orig_label"].astype(int)

train_ids = pd.read_csv(os.path.join(DATA_DIR, "train_semeval_parids-labels.csv"))
dev_ids   = pd.read_csv(os.path.join(DATA_DIR, "dev_semeval_parids-labels.csv"))
train_ids["par_id"] = train_ids["par_id"].astype(str)
dev_ids["par_id"]   = dev_ids["par_id"].astype(str)

train_pool = full_df[full_df["par_id"].isin(train_ids["par_id"])].reset_index(drop=True)
test_df    = full_df[full_df["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)

train_df, val_df = train_test_split(
    train_pool, test_size=0.10, stratify=train_pool["label"], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

print(f"Train : {len(train_df):,}  PCL={train_df['label'].sum()} ({train_df['label'].mean():.1%})")
print(f"Val   : {len(val_df):,}   PCL={val_df['label'].sum()} ({val_df['label'].mean():.1%})")
print(f"Test  : {len(test_df):,}  PCL={test_df['label'].sum()} ({test_df['label'].mean():.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Input text builders
# ─────────────────────────────────────────────────────────────────────────────
def build_with_metadata(row):
    return f"<e>{row['keyword']}</e> <e>{row['country']}</e> {html.unescape(str(row['text']))}"

def build_no_metadata(row):
    return html.unescape(str(row["text"]))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────────────────────────────────────
MAX_LENGTH = 250

class PCLDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item

# ─────────────────────────────────────────────────────────────────────────────
# 5. Grouped LLRD optimiser (same as BestModel notebook)
# ─────────────────────────────────────────────────────────────────────────────
def get_grouped_llrd_params(model, base_lr, lambda_val, weight_decay, n_groups=3):
    """Grouped layer-wise LR decay for DistilBERT."""
    encoder_layers = model.distilbert.transformer.layer
    embeddings     = model.distilbert.embeddings
    n_layers       = len(encoder_layers)
    group_size     = n_layers // n_groups

    def layer_lr(idx):
        group_idx = min(idx // group_size, n_groups - 1)
        return base_lr * (lambda_val ** (group_idx - 1))

    param_groups = []
    seen = set()

    def add_group(params_iter, lr):
        params = [p for p in params_iter if id(p) not in seen and p.requires_grad]
        for p in params:
            seen.add(id(p))
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})

    add_group(embeddings.parameters(), base_lr / lambda_val)
    for i, layer in enumerate(encoder_layers):
        add_group(layer.parameters(), layer_lr(i))
    top_lr = base_lr * lambda_val
    for attr in ("pooler", "pre_classifier", "classifier"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if m is not None:
                add_group(m.parameters(), top_lr)

    return param_groups

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_probs_and_labels(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    return np.array(all_probs), np.array(all_labels)

def get_probs_unlabelled(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            all_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.array(all_probs)

def eval_monitor(model, loader):
    """Return (F1@0.5, best-threshold F1, best-threshold) for training monitoring."""
    probs, labels = get_probs_and_labels(model, loader)
    f1_05 = f1_score(labels, (probs >= 0.5).astype(int), pos_label=1, zero_division=0)
    best_f1, best_thr = f1_05, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return f1_05, best_f1, best_thr

def tune_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), round(best_f1, 4)

def load_saved_probs(model_dir, dataset, batch_size=16):
    """Load a saved checkpoint and return positive-class probabilities."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def collate_fn(batch):
        texts  = [b["text"] for b in batch]
        labels = [b["label"] for b in batch] if "label" in batch[0] else None
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)
    model  = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labs = batch.pop("labels", None)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            all_probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            if labs is not None:
                all_labels.extend(labs.numpy())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(all_probs), (np.array(all_labels) if all_labels else None)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Model configurations
# ─────────────────────────────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Model A: "Best Combined"                                                │
# │  All four components that individually improved in the ablation study. │
# │  No metadata, no WRS, no LLRD, no warmup.                              │
# └─────────────────────────────────────────────────────────────────────────┘
MODEL_A_CFG = {
    "tag"          : "distilbert-best-combined",
    "MODEL_NAME"   : "distilbert-base-uncased",
    "DROPOUT"      : 0.3,
    "BASE_LR"      : 2e-5,
    "WEIGHT_DECAY" : 0.01,
    "BATCH_SIZE"   : 8,
    "GRAD_ACCUM"   : 2,
    # --- ablation switches ---
    "use_metadata" : False,   # removed: +0.034 delta in ablation
    "use_wrs"      : False,   # removed: +0.043 delta (recall 0.61→0.70)
    "use_llrd"     : False,   # removed: +0.025 delta
    "use_warmup"   : False,   # removed: +0.040 delta (recall 0.61→0.74)
}

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Model B: "High-Recall Companion"                                        │
# │  Designed for ensemble diversity: pairs with precision-biased RoBERTa. │
# │  Keeps metadata + LLRD (quality); drops WRS + warmup (recall boost);   │
# │  uses lower dropout (0.1) to reduce over-regularisation.               │
# └─────────────────────────────────────────────────────────────────────────┘
MODEL_B_CFG = {
    "tag"          : "distilbert-high-recall",
    "MODEL_NAME"   : "distilbert-base-uncased",
    "DROPOUT"      : 0.1,    # less regularisation → better minority class recall
    "BASE_LR"      : 2e-5,
    "WEIGHT_DECAY" : 0.01,
    "BATCH_SIZE"   : 8,
    "GRAD_ACCUM"   : 2,
    # --- ablation switches ---
    "use_metadata" : True,   # kept: keyword/country context aids calibration
    "use_wrs"      : False,  # removed: proven recall booster (+0.043 individually)
    "use_llrd"     : True,   # kept: protects lower-layer representations
    "use_warmup"   : False,  # removed: proven recall booster (+0.040 individually)
}

CONFIGS = [MODEL_A_CFG, MODEL_B_CFG]
SEEDS       = [7, 42, 123]
N_EPOCHS    = 10
LAMBDA      = 1.6
N_GROUPS    = 3
EVAL_STEPS  = 50
PATIENCE    = 15

# ─────────────────────────────────────────────────────────────────────────────
# 8. Training loop
# ─────────────────────────────────────────────────────────────────────────────
train_results = []
best_seed     = {}   # tag → {seed, f1, dir}

for cfg in CONFIGS:
    tag = cfg["tag"]
    print(f"\n{'#'*65}")
    print(f"  Training: {tag}")
    print(f"  metadata={cfg['use_metadata']}  wrs={cfg['use_wrs']}  "
          f"llrd={cfg['use_llrd']}  warmup={cfg['use_warmup']}  "
          f"dropout={cfg['DROPOUT']}")
    print(f"{'#'*65}")

    for seed in SEEDS:
        set_seed(seed)
        run_name = f"{tag}/seed{seed}"
        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"{'='*60}")

        # ── Input texts ──────────────────────────────────────────────────
        builder   = build_with_metadata if cfg["use_metadata"] else build_no_metadata
        tr_texts  = train_df.apply(builder, axis=1).tolist()
        vl_texts  = val_df.apply(builder, axis=1).tolist()
        te_texts  = test_df.apply(builder, axis=1).tolist()
        tr_labels = train_df["label"].tolist()
        vl_labels = val_df["label"].tolist()
        te_labels = test_df["label"].tolist()

        # ── Tokenizer ────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"])
        if cfg["use_metadata"]:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<e>", "</e>"]}
            )

        def collate_fn(batch, tok=tokenizer):
            texts  = [b["text"] for b in batch]
            labels = [b["label"] for b in batch] if "label" in batch[0] else None
            enc = tok(texts, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
            if labels is not None:
                enc["labels"] = torch.tensor(labels, dtype=torch.long)
            return enc

        tr_ds = PCLDataset(tr_texts, tr_labels)
        vl_ds = PCLDataset(vl_texts, vl_labels)
        te_ds = PCLDataset(te_texts, te_labels)

        # ── DataLoader: WRS or standard shuffle ──────────────────────────
        if cfg["use_wrs"]:
            labels_arr = np.array(tr_labels)
            class_counts = np.bincount(labels_arr)
            weights = 1.0 / np.sqrt(class_counts[labels_arr].astype(float))
            # Confidence-dampen ambiguous positives (orig_label==2)
            orig = train_df["orig_label"].values
            weights[orig == 2] *= 0.5
            sampler = WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.float),
                num_samples=len(tr_ds), replacement=True
            )
            train_loader = DataLoader(tr_ds, batch_size=cfg["BATCH_SIZE"],
                                      sampler=sampler, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(tr_ds, batch_size=cfg["BATCH_SIZE"],
                                      shuffle=True, collate_fn=collate_fn)

        val_loader  = DataLoader(vl_ds, batch_size=cfg["BATCH_SIZE"] * 2,
                                 shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(te_ds, batch_size=cfg["BATCH_SIZE"] * 2,
                                 shuffle=False, collate_fn=collate_fn)

        # ── Model ────────────────────────────────────────────────────────
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["MODEL_NAME"], num_labels=2,
            ignore_mismatched_sizes=True,
            dropout=cfg["DROPOUT"],
            attention_dropout=cfg["DROPOUT"],
            seq_classif_dropout=cfg["DROPOUT"],
        )
        if cfg["use_metadata"]:
            model.resize_token_embeddings(len(tokenizer))
        model.float().to(DEVICE)

        # ── Optimiser ────────────────────────────────────────────────────
        if cfg["use_llrd"]:
            param_groups = get_grouped_llrd_params(
                model, cfg["BASE_LR"], LAMBDA, cfg["WEIGHT_DECAY"], N_GROUPS
            )
            optimizer = AdamW(param_groups, eps=1e-6)
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=cfg["BASE_LR"],
                weight_decay=cfg["WEIGHT_DECAY"],
                eps=1e-6,
            )

        # ── Scheduler ────────────────────────────────────────────────────
        total_steps  = (len(train_loader) // cfg["GRAD_ACCUM"]) * N_EPOCHS
        warmup_steps = int(0.06 * total_steps) if cfg["use_warmup"] else 0
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        print(f"  Steps: {total_steps}  |  Warmup: {warmup_steps}")

        # ── Training ─────────────────────────────────────────────────────
        best_val_f1      = -1.0
        best_state_dict  = None
        patience_counter = 0
        global_step      = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            running_loss  = 0.0
            early_stopped = False

            for step, batch in enumerate(train_loader):
                labels = batch.pop("labels").to(DEVICE)
                batch  = {k: v.to(DEVICE) for k, v in batch.items()}
                loss   = model(**batch, labels=labels).loss / cfg["GRAD_ACCUM"]
                loss.backward()
                running_loss += loss.item() * cfg["GRAD_ACCUM"]

                if (step + 1) % cfg["GRAD_ACCUM"] == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % EVAL_STEPS == 0:
                        _, step_f1, step_thr = eval_monitor(model, val_loader)
                        model.train()
                        if step_f1 > best_val_f1:
                            best_val_f1      = step_f1
                            best_state_dict  = {k: v.cpu().clone()
                                                for k, v in model.state_dict().items()}
                            patience_counter = 0
                            print(f"  step {global_step:>5}  val F1*={step_f1:.4f} "
                                  f"@{step_thr:.2f}  ✓ best")
                        else:
                            patience_counter += 1
                            if patience_counter >= PATIENCE:
                                early_stopped = True
                                break

                if early_stopped:
                    break

            avg_loss = running_loss / max(step + 1, 1)
            _, epoch_f1, epoch_thr = eval_monitor(model, val_loader)
            model.train()
            suffix = "  [early stop]" if early_stopped else ""
            print(f"  Ep {epoch}  loss={avg_loss:.4f}  val F1*={epoch_f1:.4f} "
                  f"@{epoch_thr:.2f}{suffix}")
            if early_stopped:
                break

        # ── Save best checkpoint ──────────────────────────────────────────
        save_dir = os.path.join(SAVE_ROOT, run_name)
        os.makedirs(save_dir, exist_ok=True)
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        train_results.append({"tag": tag, "seed": seed, "best_val_f1": best_val_f1})
        print(f"  >> {run_name}  best val F1* = {best_val_f1:.4f}  saved → {save_dir}")

        if tag not in best_seed or best_val_f1 > best_seed[tag]["f1"]:
            best_seed[tag] = {"seed": seed, "f1": best_val_f1, "dir": save_dir}

        del model, optimizer, scheduler, tokenizer
        if best_state_dict is not None:
            del best_state_dict
        gc.collect()
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Print all-run summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n── All runs ─────────────────────────────────────────────────────")
for r in train_results:
    marker = " ★" if best_seed[r["tag"]]["seed"] == r["seed"] else ""
    print(f"  {r['tag']:35s}  seed={r['seed']}  "
          f"val F1*={r['best_val_f1']:.4f}{marker}")

print("\n── Best checkpoint per model ─────────────────────────────────────")
for tag, info in best_seed.items():
    print(f"  {tag}  seed={info['seed']}  val F1*={info['f1']:.4f}  {info['dir']}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Load existing RoBERTa (from BestModel/) for ensemble comparison
# ─────────────────────────────────────────────────────────────────────────────
ROBERTA_DIR     = os.path.join(REPO_ROOT, "BestModel", "roberta-base")
ORIG_DISTIL_DIR = os.path.join(REPO_ROOT, "BestModel", "distilbert")
BESTMODEL_DIR   = os.path.join(REPO_ROOT, "BestModel")

print("\n── Loading existing checkpoints ──────────────────────────────────")
for label, path in [("RoBERTa (BestModel)",   ROBERTA_DIR),
                    ("DistilBERT (BestModel)", ORIG_DISTIL_DIR)]:
    exists = os.path.isdir(path)
    print(f"  {label}: {'OK' if exists else 'NOT FOUND'}  ({path})")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Collect val-set probabilities for threshold/weight tuning
# ─────────────────────────────────────────────────────────────────────────────
# Build val/test datasets with metadata for RoBERTa (it was trained with metadata)
val_dataset_meta  = PCLDataset(val_df.apply(build_with_metadata,  axis=1).tolist(),
                               val_df["label"].tolist())
test_dataset_meta = PCLDataset(test_df.apply(build_with_metadata, axis=1).tolist(),
                               test_df["label"].tolist())
# Plain text for Model A (no metadata)
val_dataset_plain  = PCLDataset(val_df.apply(build_no_metadata,  axis=1).tolist(),
                                val_df["label"].tolist())
test_dataset_plain = PCLDataset(test_df.apply(build_no_metadata, axis=1).tolist(),
                                test_df["label"].tolist())

print("\nCollecting val-set probabilities …")

# RoBERTa (metadata)
roberta_val_probs, val_labels = load_saved_probs(ROBERTA_DIR, val_dataset_meta)

# Original DistilBERT (metadata)
orig_distil_val_probs, _ = load_saved_probs(ORIG_DISTIL_DIR, val_dataset_meta)

# Model A (no metadata)
model_a_tag = MODEL_A_CFG["tag"]
model_a_val_probs, _ = load_saved_probs(
    best_seed[model_a_tag]["dir"], val_dataset_plain
)

# Model B (metadata)
model_b_tag = MODEL_B_CFG["tag"]
model_b_val_probs, _ = load_saved_probs(
    best_seed[model_b_tag]["dir"], val_dataset_meta
)

# ─────────────────────────────────────────────────────────────────────────────
# 12. Tune thresholds and ensemble weights on val set
# ─────────────────────────────────────────────────────────────────────────────
def best_solo_threshold(probs, labels):
    return tune_threshold(probs, labels)

def best_ensemble_params(probs_a, probs_b, labels):
    """Grid-search w and t on the val set; return (w, t, val_f1)."""
    best_w, best_t, best_f1 = 0.5, 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        combined = w * probs_a + (1 - w) * probs_b
        for t in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(labels, (combined >= t).astype(int), pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_w, best_t, best_f1 = float(w), float(t), f1
    return round(best_w, 2), round(best_t, 2), round(best_f1, 4)

print("Tuning thresholds and ensemble weights on val set …")

# Solo thresholds
t_roberta,      vf1_roberta      = best_solo_threshold(roberta_val_probs,    val_labels)
t_orig_distil,  vf1_orig_distil  = best_solo_threshold(orig_distil_val_probs, val_labels)
t_model_a,      vf1_model_a      = best_solo_threshold(model_a_val_probs,    val_labels)
t_model_b,      vf1_model_b      = best_solo_threshold(model_b_val_probs,    val_labels)

# Ensemble weights: RoBERTa + orig DistilBERT (existing)
w_orig, t_ens_orig, vf1_ens_orig = best_ensemble_params(
    roberta_val_probs, orig_distil_val_probs, val_labels
)
# Ensemble weights: RoBERTa + Model A
w_a, t_ens_a, vf1_ens_a = best_ensemble_params(
    roberta_val_probs, model_a_val_probs, val_labels
)
# Ensemble weights: RoBERTa + Model B
w_b, t_ens_b, vf1_ens_b = best_ensemble_params(
    roberta_val_probs, model_b_val_probs, val_labels
)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("Collecting test-set probabilities …")

roberta_test_probs,     test_labels = load_saved_probs(ROBERTA_DIR,     test_dataset_meta)
orig_distil_test_probs, _           = load_saved_probs(ORIG_DISTIL_DIR, test_dataset_meta)
model_a_test_probs,     _           = load_saved_probs(
    best_seed[model_a_tag]["dir"], test_dataset_plain
)
model_b_test_probs,     _           = load_saved_probs(
    best_seed[model_b_tag]["dir"], test_dataset_meta
)

def test_f1_at(probs, labels, threshold):
    return round(float(f1_score(labels, (probs >= threshold).astype(int),
                                pos_label=1, zero_division=0)), 4)

def test_prf_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return (
        round(float(precision_score(labels, preds, pos_label=1, zero_division=0)), 4),
        round(float(recall_score(labels,    preds, pos_label=1, zero_division=0)), 4),
        round(float(f1_score(labels,        preds, pos_label=1, zero_division=0)), 4),
    )

tf1_roberta      = test_f1_at(roberta_test_probs,     test_labels, t_roberta)
tf1_orig_distil  = test_f1_at(orig_distil_test_probs, test_labels, t_orig_distil)
tf1_model_a      = test_f1_at(model_a_test_probs,     test_labels, t_model_a)
tf1_model_b      = test_f1_at(model_b_test_probs,     test_labels, t_model_b)

ens_orig_probs = w_orig * roberta_test_probs + (1 - w_orig) * orig_distil_test_probs
ens_a_probs    = w_a    * roberta_test_probs + (1 - w_a)    * model_a_test_probs
ens_b_probs    = w_b    * roberta_test_probs + (1 - w_b)    * model_b_test_probs

tf1_ens_orig = test_f1_at(ens_orig_probs, test_labels, t_ens_orig)
tf1_ens_a    = test_f1_at(ens_a_probs,    test_labels, t_ens_a)
tf1_ens_b    = test_f1_at(ens_b_probs,    test_labels, t_ens_b)

# Detailed P/R/F1 for each ensemble
p_orig, r_orig, f_orig = test_prf_at(ens_orig_probs, test_labels, t_ens_orig)
p_a,    r_a,    f_a    = test_prf_at(ens_a_probs,    test_labels, t_ens_a)
p_b,    r_b,    f_b    = test_prf_at(ens_b_probs,    test_labels, t_ens_b)

# ─────────────────────────────────────────────────────────────────────────────
# 14. Final comparison (mirrors BestModel/model.ipynb output)
# ─────────────────────────────────────────────────────────────────────────────
SEP = "─" * 72
print(f"\n{SEP}")
print("── Final comparison ──")
print(f"{SEP}")
print(f"  {'Model':<42} {'val F1*':>9}  {'test F1':>9}  {'t':>5}")
print(f"  {'-'*42}  {'-'*9}  {'-'*9}  {'-'*5}")

# ── Solo models ──────────────────────────────────────────────────────────────
print(f"  {'roberta-base  (BestModel, seed=7)':<42} {'N/A':>9}  {tf1_roberta:>9.4f}  {t_roberta:>5.2f}")
print(f"  {'distilbert    (BestModel, seed=7)':<42} {'N/A':>9}  {tf1_orig_distil:>9.4f}  {t_orig_distil:>5.2f}")
print(f"  {'Model A  (best-combined, seed='+str(best_seed[model_a_tag]['seed'])+')':<42} {vf1_model_a:>9.4f}  {tf1_model_a:>9.4f}  {t_model_a:>5.2f}")
print(f"  {'Model B  (high-recall,   seed='+str(best_seed[model_b_tag]['seed'])+')':<42} {vf1_model_b:>9.4f}  {tf1_model_b:>9.4f}  {t_model_b:>5.2f}")

# ── Ensembles ─────────────────────────────────────────────────────────────────
print()
print(f"  {'ORIGINAL ENSEMBLE (RoBERTa + orig DistilBERT)':<42} {vf1_ens_orig:>9.4f}  {tf1_ens_orig:>9.4f}  "
      f"{t_ens_orig:>5.2f}  (w={w_orig:.2f}/{1-w_orig:.2f})  "
      f"P={p_orig:.4f} R={r_orig:.4f}")
print(f"  {'NEW  ENSEMBLE A (RoBERTa + Model A)':<42} {vf1_ens_a:>9.4f}  {tf1_ens_a:>9.4f}  "
      f"{t_ens_a:>5.2f}  (w={w_a:.2f}/{1-w_a:.2f})  "
      f"P={p_a:.4f} R={r_a:.4f}")
print(f"  {'NEW  ENSEMBLE B (RoBERTa + Model B)':<42} {vf1_ens_b:>9.4f}  {tf1_ens_b:>9.4f}  "
      f"{t_ens_b:>5.2f}  (w={w_b:.2f}/{1-w_b:.2f})  "
      f"P={p_b:.4f} R={r_b:.4f}")

# ── Best overall ──────────────────────────────────────────────────────────────
candidates = {
    "ORIGINAL ENSEMBLE": (vf1_ens_orig, tf1_ens_orig),
    "NEW ENSEMBLE A":    (vf1_ens_a,    tf1_ens_a),
    "NEW ENSEMBLE B":    (vf1_ens_b,    tf1_ens_b),
}
best_name = max(candidates, key=lambda k: candidates[k][1])
print(f"\n  >>> Best on test: {best_name}  (test F1={candidates[best_name][1]:.4f})")
print(f"{SEP}\n")
