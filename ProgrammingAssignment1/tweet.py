import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.json"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    txt = f.read().strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            records = obj
        else:
            raise ValueError
    except Exception:
        records = [json.loads(line) for line in txt.splitlines() if line.strip()]

EMOTIONS = ['anger','anticipation','disgust','fear','joy','love',
            'optimism','pessimism','sadness','surprise','trust']

texts, labels = [], []
for r in records:
    t = r.get("Tweet") or r.get("text") or ""
    labs_true = [emo for emo in EMOTIONS if bool(r.get(emo, False))]
    if t and labs_true:
        labels.append(labs_true[0])
        texts.append(t)
labels = np.array(labels)
print(f"Loaded training examples: {len(texts)}")

cv = CountVectorizer()
tfidf = TfidfVectorizer()

X_count = cv.fit_transform(texts)
X_tfidf = tfidf.fit_transform(texts)

print("[Dimensionality] CountVectorizer:", X_count.shape[1])
print("[Dimensionality] TfidfVectorizer:", X_tfidf.shape[1])

CHOSEN = ["joy", "anger", "love", "disgust"]
mask = np.isin(labels, CHOSEN)
texts4  = [t for t, m in zip(texts, mask) if m]
labels4 = labels[mask]

cls_to_idx = {c:i for i,c in enumerate(CHOSEN)}
y_idx = np.array([cls_to_idx[l] for l in labels4])

Xc4 = cv.transform(texts4)
Xt4 = tfidf.transform(texts4)

#PCA 2D (COUNT)
Zc = PCA(n_components=2, random_state=0).fit_transform(Xc4.toarray())
colors = ['tab:blue','tab:orange','tab:green','tab:red']

plt.figure(figsize=(6,5))
for i, cls in enumerate(CHOSEN):
    idx = (y_idx == i)
    plt.scatter(Zc[idx,0], Zc[idx,1], s=10, alpha=0.75, label=cls, c=colors[i])
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Tweets PCA (2D) CountVectorizer")
plt.legend(markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tweets_pca_count.png", dpi=200, bbox_inches="tight")
plt.show()

#PCA 2D (TF-IDF)
Zt = PCA(n_components=2, random_state=0).fit_transform(Xt4.toarray())

plt.figure(figsize=(6,5))
for i, cls in enumerate(CHOSEN):
    idx = (y_idx == i)
    plt.scatter(Zt[idx,0], Zt[idx,1], s=10, alpha=0.75, label=cls, c=colors[i])
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Tweets PCA (2D) TF-IDF")
plt.legend(markerscale=1.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tweets_pca_tfidf.png", dpi=200, bbox_inches="tight")
plt.show()
