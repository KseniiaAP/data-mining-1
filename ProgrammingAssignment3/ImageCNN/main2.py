import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 2(a) Dataset loading, preprocessing, 80/20 split
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "flowers")
OUT_DIR = ROOT

IMG_SIZE = 180
BATCH = 32
EPOCHS = 20
BANNER_LAST_DIGIT = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full = datasets.ImageFolder(DATA_DIR, transform=tfm)
num_val = int(0.2 * len(full))
num_train = len(full) - num_val

train_ds, val_ds = random_split(
    full, [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

classes = full.classes
assert len(classes) == 4

# 2(b) Base CNN architecture
class Net(nn.Module):
    def __init__(self, conv2_filters=4, conv2_kernel=3, dense_units=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),  # Conv 1: 8 3x3
            nn.MaxPool2d(2),                                      # MaxPool 2x2
            nn.Conv2d(8, conv2_filters, kernel_size=conv2_kernel,
                      padding=conv2_kernel//2), nn.ReLU(),        # Conv 2
            nn.MaxPool2d(2),                                      # MaxPool 2x2
            nn.Flatten(),                                         # Flatten
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_dim = self.net[:-1](dummy).numel()
        self.fc1 = nn.Linear(feat_dim, dense_units)  # hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, 4)

    def forward(self, x):
        x = self.net(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2(c) Training loop and validation accuracy
def train_eval(model, epochs, tag):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    train_acc_hist, val_acc_hist = [], []

    for ep in range(1, epochs+1):
        model.train()
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        print(f"[{tag}] epoch {ep:02d} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    # 2(d / 2(e)) Plot learning curves for this model
    plt.figure()
    plt.plot(train_acc_hist, label="train acc")
    plt.plot(val_acc_hist, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{tag} - accuracy")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{tag}_accuracy.png")
    plt.savefig(out, dpi=180)
    print(f"Saved: {out}")

    return train_acc_hist, val_acc_hist

# 2(d) Base model: dense_units = 8
base = Net(conv2_filters=4, conv2_kernel=3, dense_units=8)
base_hist = train_eval(base, EPOCHS, "cnn_torch_base")

# 2(e) last Banner ID digit 9:
variants = []

for du in [4, 16]:
    tag = f"cnn_torch_dense_{du}"
    m = Net(conv2_filters=4, conv2_kernel=3, dense_units=du)
    hist = train_eval(m, EPOCHS, tag)
    variants.append((f"dense_{du}", hist))
