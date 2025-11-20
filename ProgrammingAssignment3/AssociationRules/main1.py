import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "Grocery_Items_21.csv")  # твой номер
OUT_DIR = HERE

# 1(a-b)
raw = pd.read_csv(CSV_PATH, header=None, sep=None, engine="python")
transactions = []
for _, row in raw.iterrows():
    items = []
    for cell in row.dropna().astype(str):
        cell = cell.replace(";", ",")
        items.extend([x.strip() for x in cell.split(",") if x.strip()])
    transactions.append(items)

# 1(c)
n_txn = len(transactions)
all_items = [it for txn in transactions for it in txn]
unique_items = sorted(set(all_items))
n_unique = len(unique_items)
counts = pd.Series(all_items).value_counts()
most_popular_item = counts.idxmax()
most_popular_in_txn = sum(most_popular_item in txn for txn in transactions)

print(f"[1(c)] unique items: {n_unique}")
print(f"[1(c)] #records: {n_txn}")
print(f"[1(c)] most popular item: {most_popular_item} (appears in {most_popular_in_txn} transactions)")

te = TransactionEncoder()
onehot = te.fit(transactions).transform(transactions)
df = pd.DataFrame(onehot, columns=te.columns_)

# 1(d) min_support=0.01 and min_confidence=0.08
freq = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.08).sort_values(
    ["confidence", "lift"], ascending=False
)
rules_path = os.path.join(OUT_DIR, "rules_s0.01_c0.08.csv")
rules.to_csv(rules_path, index=False)
print(f"[1(d)] saved: {rules_path} (rules={len(rules)})")

# 1(e)
supports = [0.001, 0.005, 0.01]
confidences = [0.05, 0.075, 0.1]
grid = pd.DataFrame(index=confidences, columns=supports, dtype=int)

for s in supports:
    f = apriori(df, min_support=s, use_colnames=True)
    for c in confidences:
        if len(f) == 0:
            grid.loc[c, s] = 0
        else:
            r = association_rules(f, metric="confidence", min_threshold=c)
            grid.loc[c, s] = len(r)

grid.index.name = "mct (confidence)"
grid.columns.name = "msv (support)"
print("[1(e)] rules count grid:\n", grid)

plt.figure(figsize=(6, 4))
grid = grid.astype(int)
sns.heatmap(grid, annot=True, fmt="d")
plt.title("Number of Rules by (support, confidence)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rules_heatmap.png"), dpi=200)
print(f"[1(e)] heatmap saved {os.path.join(OUT_DIR, 'rules_heatmap.png')}")
