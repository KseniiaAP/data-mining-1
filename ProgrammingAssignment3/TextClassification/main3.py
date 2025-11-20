import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

HERE = os.path.dirname(__file__)
TRAIN_JSON = os.path.join(HERE, "train.json")
VAL_JSON   = os.path.join(HERE, "validation.json")
TEST_JSON  = os.path.join(HERE, "test.json")
OUT_DIR = HERE

# 3(a) Define label set (11 emotions)
LABELS = ["anger","anticipation","disgust","fear","joy","love",
          "optimism","pessimism","sadness","surprise","trust"]

def load_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return pd.DataFrame()
    if txt.startswith("["):
        data = json.loads(txt)
        return pd.DataFrame(data)

    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# 3(b) Load train/validation/test sets from JSON files
for p in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Not found {os.path.basename(p)}. Copy train.json, validation.json, test.json to the folder TextClassification."
        )

df_train = load_json_any(TRAIN_JSON)
df_val   = load_json_any(VAL_JSON)
df_test  = load_json_any(TEST_JSON)

# 3(b) Ensure label columns exist and are 0/1 integers
for lab in LABELS:
    if lab not in df_train.columns:
        raise ValueError(f"The placemark column is missing in the dataset: {lab}")

    for d in (df_train, df_val, df_test):
        d[lab] = d[lab].astype(int)

# 3(b) Extract raw text column and cast to string
TEXT_COL = "Tweet" if "Tweet" in df_train.columns else "text"
for d in (df_train, df_val, df_test):
    d[TEXT_COL] = d[TEXT_COL].astype(str)

def df_to_xy(df):
    x = df[TEXT_COL].tolist()
    y = df[LABELS].values.astype(int)
    return x, y

x_train, y_train = df_to_xy(df_train)
x_val,   y_val   = df_to_xy(df_val)
x_test,  y_test  = df_to_xy(df_test)

# 3(c) Text preprocessing with BERT tokenizer
MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 3(c) Wrap data into custom PyTorch Dataset for Trainer API
class TweetDS(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):

        enc = tokenizer(self.texts[i], truncation=True, padding=False, max_length=128)
        enc = {k: torch.tensor(v) for k, v in enc.items()}

        enc["labels"] = torch.tensor(self.labels[i]).float()
        return enc

train_ds = TweetDS(x_train, y_train)
val_ds   = TweetDS(x_val, y_val)
test_ds  = TweetDS(x_test, y_test)

# 3(d) BERT base model for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS), problem_type="multi_label_classification"
)

collator = DataCollatorWithPadding(tokenizer)

# 3(e) Define evaluation metrics
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred, threshold=0.5):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)

    labels = np.asarray(labels)
    if labels.ndim == 1 and preds.ndim == 2:
        labels = labels.reshape(preds.shape)
    labels_bin = (labels >= 0.5).astype(np.int64)

    strict = (preds == labels_bin).all(axis=1).mean()

    any_match = (np.logical_and(preds == 1, labels_bin == 1).sum(axis=1) > 0).mean()

    return {
        "strict_all_match_acc": float(strict),
        "any_label_match_acc": float(any_match)
    }

# 3(f) Fine-tuning setup: TrainingArguments
args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "bert_out"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="strict_all_match_acc",
    greater_is_better=True,
)

# 3(f) Trainer: combines model, data, tokenizer, collator and metrics
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, threshold=0.5),
)

# 3(g) Train the model and evaluate on validation set
train_result = trainer.train()
_ = trainer.evaluate()

# 3(h) Collect training & validation losses for plotting learning curves
log = trainer.state.log_history
epochs, tr_losses, val_losses = [], [], []
for row in log:
    if "loss" in row and "epoch" in row:
        epochs.append(row["epoch"])
        tr_losses.append(row["loss"])
    if "eval_loss" in row and "epoch" in row:
        val_losses.append(row["eval_loss"])

plt.figure()
if tr_losses:
    plt.plot(range(1, len(tr_losses)+1), tr_losses, label="train loss")
if val_losses:
    plt.plot(range(1, len(val_losses)+1), val_losses, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("BERT fine-tuning - losses (5 epochs)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bert_losses.png"), dpi=180)

# Final evaluation on the test set
test_metrics = trainer.evaluate(test_ds)
print("[3] strict_all_match_acc:", test_metrics["eval_strict_all_match_acc"])
print("[3] any_label_match_acc:", test_metrics["eval_any_label_match_acc"])
