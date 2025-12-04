from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_dataloader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        root=str(data_dir),
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return dataloader, dataset


def build_resnet18_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.eval()
    model.to(device)
    return model, device


def extract_last_conv_features(model, device, dataloader):
    all_features = []
    all_labels = []

    extracted = []

    def hook(module, input, output):
        extracted.append(output.detach())

    handle = model.layer4[-1].register_forward_hook(hook)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            extracted.clear()
            _ = model(images)
            feat = extracted[0]

            feat = feat.cpu()
            n, c, h, w = feat.shape
            feat = feat.view(n, -1)

            all_features.append(feat)
            all_labels.append(labels.numpy())

    handle.remove()

    features = torch.cat(all_features, dim=0).numpy()
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def main():
    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parent

    data_dir = base_dir / "flowers"
    output_dir = base_dir / "features"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset")
    dataloader, dataset = get_dataloader(data_dir, batch_size=32)

    print("Loading ResNet18...")
    model, device = build_resnet18_device()

    print("Extracting features from the last convolutional layer...")
    features, labels = extract_last_conv_features(model, device, dataloader)

    features_path = output_dir / "resnet18_lastconv_features.npy"
    labels_path = output_dir / "labels.npy"
    classes_path = output_dir / "classes.txt"

    np.save(features_path, features)
    np.save(labels_path, labels)

    with open(classes_path, "w", encoding="utf-8") as f:
        for idx, class_name in enumerate(dataset.classes):
            f.write(f"{idx}\t{class_name}\n")

    print(f"Saved: {features_path}")
    print(f"Saved: {labels_path}")
    print(f"Saved: {classes_path}")


if __name__ == "__main__":
    main()