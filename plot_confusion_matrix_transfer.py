import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

import models.mambaout  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CIFAR-100 confusion matrix.")
    parser.add_argument("--model", default="mambaout_tiny_eca", type=str)
    parser.add_argument("--checkpoint-path", default="./output/cifar100_transfer_mambaout_tiny_eca_300ep/model_best.pth.tar", type=str)
    parser.add_argument("--data-root", default="./data", type=str)
    parser.add_argument("--output-dir", default="./figures", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--download", action="store_true", default=False)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    test_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    class_names = test_dataset.classes

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )

    load_checkpoint(model, args.checkpoint_path)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.int32)
    for true_label, pred_label in zip(all_labels, all_preds):
        conf_mat[true_label, pred_label] += 1

    conf_mat_norm = conf_mat.astype(np.float32)
    row_sums = conf_mat_norm.sum(axis=1, keepdims=True)
    conf_mat_norm = conf_mat_norm / np.maximum(row_sums, 1)

    correct = np.diag(conf_mat)
    total = conf_mat.sum(axis=1)
    errors = total - correct

    top_indices = np.argsort(errors)[-args.top_k:]
    sub_mat = conf_mat_norm[np.ix_(top_indices, top_indices)]
    sub_names = [class_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sub_mat, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(args.top_k))
    ax.set_yticks(np.arange(args.top_k))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=8)
    ax.set_yticklabels(sub_names, fontsize=8)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"CIFAR-100 Confusion Matrix Top-{args.top_k} Hardest Classes")

    for i in range(args.top_k):
        for j in range(args.top_k):
            value = sub_mat[i, j]
            if value > 0.01:
                text_color = "white" if value > 0.5 else "black"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color
                )

    plt.colorbar(im)
    plt.tight_layout()

    save_path = output_dir / f"cifar100_confusion_matrix_{args.model}_top{args.top_k}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


if __name__ == "__main__":
    main()
