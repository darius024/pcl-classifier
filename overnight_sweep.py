"""
overnight_sweep.py  —  PCL Classifier hyperparameter sweep

Trains multiple model configs (RoBERTa + DistilBERT) across several seeds.
Best checkpoint per config is saved to SavedModels/<tag>/seed<seed>/.
All output is mirrored to a timestamped log file in the working directory.

Run:
    python overnight_sweep.py
    # or background it:
    nohup python overnight_sweep.py &
"""

import gc
import html
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ── Tee: write stdout to both terminal and log file ───────────────────────────

class _Tee:
    """Duplicate writes to two streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


SCRIPT_DIR = Path(__file__).parent
LOG_PATH   = SCRIPT_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.log"
_log_file  = open(LOG_PATH, "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

print(f"Logging to: {LOG_PATH}")
print(f"Started:    {datetime.now():%Y-%m-%d %H:%M:%S}\n")


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = SCRIPT_DIR / "data"
SAVE_ROOT  = SCRIPT_DIR / "SavedModels"

sys.path.insert(0, str(SCRIPT_DIR / "4_model"))
from dont_patronize_me import DontPatronizeMe  # noqa: E402


# ── Sweep configuration ───────────────────────────────────────────────────────
#
# Each entry is one (model × dropout × batch_size × grad_accum) combination.
# Every entry is trained on all seeds in SEEDS.

SEEDS = [42, 7, 123]

SWEEP_CONFIGS = [
    # ── RoBERTa: dropout=0.0 ─────────────────────────────────────────────
    {
        "tag":          "roberta-drop0-bs8ga2",
        "MODEL_NAME":   "roberta-base",
        "BASE_LR":      1e-5,
        "DROPOUT":      0.0,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   8,
        "GRAD_ACCUM":   2,
    },
    {
        "tag":          "roberta-drop0-bs4ga4",
        "MODEL_NAME":   "roberta-base",
        "BASE_LR":      1e-5,
        "DROPOUT":      0.0,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   4,
        "GRAD_ACCUM":   4,
    },
    # ── RoBERTa: dropout=0.5 ─────────────────────────────────────────────
    {
        "tag":          "roberta-drop05-bs8ga2",
        "MODEL_NAME":   "roberta-base",
        "BASE_LR":      1e-5,
        "DROPOUT":      0.5,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   8,
        "GRAD_ACCUM":   2,
    },
    {
        "tag":          "roberta-drop05-bs4ga4",
        "MODEL_NAME":   "roberta-base",
        "BASE_LR":      1e-5,
        "DROPOUT":      0.5,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   4,
        "GRAD_ACCUM":   4,
    },
    # ── DistilBERT: dropout=0.1 ───────────────────────────────────────────
    {
        "tag":          "distilbert-drop01-bs8ga2",
        "MODEL_NAME":   "distilbert-base-uncased",
        "BASE_LR":      2e-5,
        "DROPOUT":      0.1,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   8,
        "GRAD_ACCUM":   2,
    },
    {
        "tag":          "distilbert-drop01-bs4ga4",
        "MODEL_NAME":   "distilbert-base-uncased",
        "BASE_LR":      2e-5,
        "DROPOUT":      0.1,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   4,
        "GRAD_ACCUM":   4,
    },
    # ── DistilBERT: dropout=0.3 ───────────────────────────────────────────
    {
        "tag":          "distilbert-drop03-bs8ga2",
        "MODEL_NAME":   "distilbert-base-uncased",
        "BASE_LR":      2e-5,
        "DROPOUT":      0.3,
        "WEIGHT_DECAY": 0.01,
        "LABEL_SMOOTH": 0.0,
        "BATCH_SIZE":   8,
        "GRAD_ACCUM":   2,
    },
]

# ── Shared training hyper-params ──────────────────────────────────────────────

N_EPOCHS     = 8
LAMBDA       = 1.6
N_GROUPS     = 3
WARMUP_RATIO = 0.10
PATIENCE     = 15
EVAL_STEPS   = 50
MAX_LENGTH   = 250


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}\n")


# ── Data loading ──────────────────────────────────────────────────────────────

def build_input_text(row: pd.Series) -> str:
    return f"<e>{row['keyword']}</e> <e>{row['country']}</e> {html.unescape(row['text'])}"


def load_splits():
    dpm = DontPatronizeMe(str(DATA_DIR), str(DATA_DIR / "task4_test.tsv"))
    dpm.load_task1()
    full_df = dpm.train_task1_df.copy()
    full_df["par_id"]     = full_df["par_id"].astype(str)
    full_df["orig_label"] = full_df["orig_label"].astype(int)

    train_ids = pd.read_csv(DATA_DIR / "train_semeval_parids-labels.csv")
    dev_ids   = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")
    train_ids["par_id"] = train_ids["par_id"].astype(str)
    dev_ids["par_id"]   = dev_ids["par_id"].astype(str)

    train_pool = full_df[full_df["par_id"].isin(train_ids["par_id"])].reset_index(drop=True)
    test_df    = full_df[full_df["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_pool, test_size=0.10, stratify=train_pool["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    for df in [train_df, val_df, test_df]:
        df["input_text"] = df.apply(build_input_text, axis=1)

    print(f"Train: {len(train_df):,}  PCL: {train_df['label'].sum()} ({train_df['label'].mean():.1%})")
    print(f"Val:   {len(val_df):,}   PCL: {val_df['label'].sum()} ({val_df['label'].mean():.1%})")
    print(f"Test:  {len(test_df):,}  PCL: {test_df['label'].sum()} ({test_df['label'].mean():.1%})\n")

    return train_df, val_df, test_df


# ── Dataset ───────────────────────────────────────────────────────────────────

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


def make_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    labels_array   = train_df["label"].values
    num_positive   = labels_array.sum()
    ratio_positive = num_positive / len(labels_array)
    ratio_negative = 1.0 - ratio_positive
    w_pos = 1.0 / np.sqrt(ratio_positive)
    w_neg = 1.0 / np.sqrt(ratio_negative)

    CONF = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0, 4: 1.0}
    orig = train_df["orig_label"].astype(int).values
    weights = np.where(labels_array == 1, w_pos, w_neg) * np.array([CONF[l] for l in orig])

    print(f"WRS  pos weight={w_pos:.3f}  neg weight={w_neg:.3f}")
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=len(train_df),
        replacement=True,
    )


# ── Grouped LLRD ──────────────────────────────────────────────────────────────

def get_grouped_llrd_params(model, base_lr, lambda_val, weight_decay, n_groups=3):
    """Layer-wise learning rate decay split into n_groups from bottom to top."""
    if hasattr(model, "deberta"):
        encoder_layers = model.deberta.encoder.layer
        embeddings     = model.deberta.embeddings
        backbone       = "deberta"
    elif hasattr(model, "roberta"):
        encoder_layers = model.roberta.encoder.layer
        embeddings     = model.roberta.embeddings
        backbone       = "roberta"
    elif hasattr(model, "distilbert"):
        encoder_layers = model.distilbert.transformer.layer
        embeddings     = model.distilbert.embeddings
        backbone       = "distilbert"
    elif hasattr(model, "bert"):
        encoder_layers = model.bert.encoder.layer
        embeddings     = model.bert.embeddings
        backbone       = "bert"
    else:
        return [{"params": model.parameters(), "lr": base_lr}]

    n_layers   = len(encoder_layers)
    group_size = max(n_layers // n_groups, 1)

    def layer_lr(idx):
        g = min(idx // group_size, n_groups - 1)
        return base_lr * (lambda_val ** (g - 1))

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

    if backbone == "deberta":
        top_lr = base_lr * lambda_val
        for attr in ("rel_embeddings", "LayerNorm"):
            if hasattr(model.deberta.encoder, attr):
                add_group(getattr(model.deberta.encoder, attr).parameters(), top_lr)

    top_lr = base_lr * lambda_val
    for attr in ("pooler", "pre_classifier", "classifier"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if m is not None:
                add_group(m.parameters(), top_lr)

    total = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"LLRD: {len(param_groups)} groups | {total:,} trainable params")
    for i, g in enumerate(param_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  Group {i:2d}: lr={g['lr']:.2e}  params={n:,}")

    return param_groups


# ── Model loading helpers ─────────────────────────────────────────────────────

def load_model(model_name: str, dropout: float, n_labels: int = 2):
    """Load a sequence-classification model with the given dropout."""
    is_deberta    = "deberta" in model_name
    is_distilbert = "distilbert" in model_name

    kwargs = {"ignore_mismatched_sizes": True}
    if is_deberta:
        kwargs["attn_implementation"] = "eager"

    if is_distilbert:
        # DistilBERT uses different dropout config keys
        kwargs["dropout"]             = dropout
        kwargs["attention_dropout"]   = dropout
        kwargs["seq_classif_dropout"] = dropout
    else:
        kwargs["hidden_dropout_prob"]            = dropout
        kwargs["attention_probs_dropout_prob"]   = dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels, **kwargs
    )
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            probs  = torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    return probs, labels, f1_score(labels, preds, pos_label=1, zero_division=0)


def eval_monitor(model, loader, thresholds=np.arange(0.1, 0.91, 0.01)):
    probs, labels, f1_05 = evaluate(model, loader, threshold=0.5)
    best_thr, best_f1 = 0.5, f1_05
    for thr in thresholds:
        f1_t = f1_score(labels, (probs >= thr).astype(int), pos_label=1, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thr = f1_t, float(thr)
    return f1_05, best_f1, best_thr


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_run(cfg, seed, train_dataset, val_dataset, sampler):
    set_seed(seed)
    run_name = f"{cfg['tag']}/seed{seed}"
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  {run_name}  ({cfg['MODEL_NAME']}  seed={seed})")
    print(bar)

    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    def collate_fn(batch, tok=tokenizer):
        texts   = [b["text"] for b in batch]
        labels  = [b["label"] for b in batch] if "label" in batch[0] else None
        encoded = tok(texts, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
        if labels is not None:
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["BATCH_SIZE"],
        sampler=sampler, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["BATCH_SIZE"] * 2,
        shuffle=False, collate_fn=collate_fn,
    )

    model = load_model(cfg["MODEL_NAME"], cfg["DROPOUT"])
    model.resize_token_embeddings(len(tokenizer))
    model.float().to(DEVICE)

    param_groups = get_grouped_llrd_params(
        model, cfg["BASE_LR"], LAMBDA, cfg["WEIGHT_DECAY"], N_GROUPS
    )
    optimizer    = AdamW(param_groups, eps=1e-6)
    total_steps  = (len(train_loader) // cfg["GRAD_ACCUM"]) * N_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.CrossEntropyLoss(label_smoothing=cfg["LABEL_SMOOTH"])

    print(f"  Steps: {total_steps}  |  Warmup: {warmup_steps}")

    best_val_f1      = -1.0
    best_state_dict  = None
    patience_counter = 0
    global_step      = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        running_loss  = 0.0
        early_stopped = False

        progress = tqdm(train_loader, desc=f"  {run_name} Ep {epoch}/{N_EPOCHS}",
                        leave=False, file=sys.__stdout__)

        for step, batch in enumerate(progress):
            labels = batch.pop("labels").to(DEVICE)
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            loss   = loss_fn(model(**batch).logits, labels) / cfg["GRAD_ACCUM"]
            loss.backward()
            running_loss += loss.item() * cfg["GRAD_ACCUM"]

            if (step + 1) % cfg["GRAD_ACCUM"] == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % EVAL_STEPS == 0 and global_step >= warmup_steps:
                    _, step_f1, step_thr = eval_monitor(model, val_loader)
                    model.train()
                    if step_f1 > best_val_f1:
                        best_val_f1      = step_f1
                        best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                        tqdm.write(
                            f"  step {global_step:>5}  val F1*={step_f1:.4f} @{step_thr:.2f}  ✓ best"
                        )
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            early_stopped = True
                            break

            if early_stopped:
                break

        avg_loss = running_loss / max(step + 1, 1)
        _, ep_f1, ep_thr = eval_monitor(model, val_loader)
        model.train()
        suffix = "  [early stop]" if early_stopped else ""
        print(f"  Ep {epoch}  loss={avg_loss:.4f}  val F1*={ep_f1:.4f} @{ep_thr:.2f}{suffix}")

        if early_stopped:
            break

    # Save best checkpoint
    save_dir = SAVE_ROOT / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"  >> Saved to {save_dir}")

    print(f"  >> {run_name} best val F1* = {best_val_f1:.4f}")

    del model, optimizer, scheduler, tokenizer, best_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    train_df, val_df, test_df = load_splits()
    sampler       = make_sampler(train_df)
    train_dataset = PCLDataset(train_df["input_text"].tolist(), train_df["label"].tolist())
    val_dataset   = PCLDataset(val_df["input_text"].tolist(),   val_df["label"].tolist())

    all_results = []
    best_per_tag = {}

    for cfg in SWEEP_CONFIGS:
        tag = cfg["tag"]
        tag_results = []

        for seed in SEEDS:
            f1 = train_one_run(cfg, seed, train_dataset, val_dataset, sampler)
            run_name = f"{tag}/seed{seed}"
            tag_results.append({"run": run_name, "tag": tag, "seed": seed, "val_f1": f1})
            all_results.append(tag_results[-1])

            if tag not in best_per_tag or f1 > best_per_tag[tag]["val_f1"]:
                best_per_tag[tag] = {
                    "seed": seed, "val_f1": f1,
                    "dir":  str(SAVE_ROOT / run_name),
                }

        # Mini-summary after each config
        print(f"\n── {tag} summary ──")
        for r in tag_results:
            star = " ★" if r["seed"] == best_per_tag[tag]["seed"] else ""
            print(f"  seed={r['seed']}  val F1*={r['val_f1']:.4f}{star}")

    # ── Final summary table ──────────────────────────────────────────────────
    divider = "─" * 70
    print(f"\n\n{'═'*70}")
    print("  SWEEP COMPLETE — FULL RESULTS")
    print(f"{'═'*70}")
    print(f"  {'Run':<42}  {'Val F1*':>8}")
    print(divider)
    for r in sorted(all_results, key=lambda x: -x["val_f1"]):
        print(f"  {r['run']:<42}  {r['val_f1']:>8.4f}")

    print(f"\n{divider}")
    print("  BEST SEED PER CONFIG")
    print(divider)
    for tag, info in sorted(best_per_tag.items(), key=lambda x: -x[1]["val_f1"]):
        print(f"  {tag:<38}  seed={info['seed']}  F1*={info['val_f1']:.4f}")
        print(f"    saved → {info['dir']}")

    print(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Log file: {LOG_PATH}")


if __name__ == "__main__":
    main()
