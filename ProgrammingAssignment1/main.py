import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import color, filters, exposure, feature
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Flowers Data Set" / "flowers"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CLASSES = ["daisy", "dandelion", "rose", "tulip"]

#Edge histogram
one_images = {}
for c in CLASSES:
    cdir = DATA_DIR / c
    files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
    files.sort()
    one_images[c] = cdir / files[0]

hist_by_class = {}
plt.figure(figsize=(12, 7))

for i, (cls, path) in enumerate(one_images.items()):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    gray = color.rgb2gray(arr)

    dx = filters.sobel_h(gray)
    dy = filters.sobel_v(gray)
    ang = np.mod(np.arctan2(dy, dx), np.pi)

    hist, bin_centers = exposure.histogram(ang, nbins=36)
    hist = hist.astype(float)
    hist /= hist.sum()

    plt.subplot(2, 4, i+1)
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.title(cls)

    plt.subplot(2, 4, i+1+4)
    plt.bar(range(len(hist)), hist)
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")
    plt.title(f"Edge Histogram ({cls})")

    hist_by_class[cls] = hist

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "edge_histograms.png", dpi=200, bbox_inches="tight")
plt.show()

h1 = hist_by_class["daisy"].reshape(1, -1)
h2 = hist_by_class["rose"].reshape(1, -1)
print("daisy vs rose")
print("Euclidean:", pairwise_distances(h1, h2, metric="euclidean")[0, 0])
print("Manhattan:", pairwise_distances(h1, h2, metric="manhattan")[0, 0])
print("Cosine:", cosine_distances(h1, h2)[0, 0])

#HOG

cdir = DATA_DIR / "daisy"
sample = sorted([f for f in os.listdir(cdir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))])[0]
path = cdir / sample

img = Image.open(path).convert("RGB")
arr = np.asarray(img)
gray = color.rgb2gray(arr)

hog_vec, hog_img = feature.hog(
    gray, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), visualize=True, channel_axis=None
)
print("HOG vector length:", hog_vec.size)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(gray, cmap="gray"); axs[0].axis("off"); axs[0].set_title("Image")
axs[1].imshow(hog_img, cmap="gray"); axs[1].axis("off"); axs[1].set_title("HOG")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hog.png", dpi=200, bbox_inches="tight")
plt.show()

#PCA

paths, labels = [], []
for label, c in enumerate(CLASSES):
    cdir = DATA_DIR / c
    for fname in os.listdir(cdir):
        if fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
            paths.append(cdir / fname)
            labels.append(label)
labels = np.array(labels)

H = []
for p in paths:
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img)
    gray = color.rgb2gray(arr)
    dx = filters.sobel_h(gray)
    dy = filters.sobel_v(gray)
    ang = np.mod(np.arctan2(dy, dx), np.pi)
    hist, _ = exposure.histogram(ang, nbins=36)
    hist = hist.astype(float)
    hist /= hist.sum()
    H.append(hist)

H = np.array(H)

pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(H)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
plt.figure(figsize=(6, 5))
for i, cls in enumerate(CLASSES):
    idx = (labels == i)
    plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.75, label=cls, c=colors[i])
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of 36-bin Edge Histograms")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca.png", dpi=200, bbox_inches="tight")
plt.show()
