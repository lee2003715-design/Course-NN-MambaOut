"""
CIFAR-100 transfer learning script for the 300-epoch MambaOut-Tiny-ECA checkpoint.

Default transfer setting:
ImageNet-100 MambaOut-Tiny-ECA 300-epoch checkpoint
    -> reset classifier head
    -> fine-tune on CIFAR-100

Use this file to replace train_cifar100.py in the project root.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Import local model registrations, including mambaout_tiny_eca if you replaced models/mambaout.py
import models.mambaout  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 transfer learning for MambaOut / MambaOut-ECA"
    )
    parser.add_argument("--data-root", default="./data", type=str,
                        help="Root folder that contains cifar-100-python. Default: ./data")
    parser.add_argument("--output-dir", default="./output/cifar100_transfer_mambaout_tiny_eca_300ep", type=str,
                        help="Directory to save checkpoints and summary.csv")
    parser.add_argument("--model", default="mambaout_tiny_eca", type=str,
                        help="Model name registered in timm. Use mambaout_tiny_eca for the improved architecture.")
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--checkpoint-path", default="./output/mambaout_tiny_eca_imagenet100_scratch_300ep/model_best.pth.tar", type=str,
                        help="ImageNet-100 checkpoint used for transfer initialization")
    parser.add_argument("--reset-head", action="store_true", default=True,
                        help="Do not load classifier head from source checkpoint. Default: True")
    parser.add_argument("--load-head", dest="reset_head", action="store_false",
                        help="Load classifier head too. Not recommended for ImageNet-100 -> CIFAR-100 transfer")
    parser.add_argument("--require-checkpoint", action="store_true",
                        help="Stop if checkpoint-path does not exist")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--early-stop-patience", default=0, type=int,
                        help="Stop if test Acc@1 does not improve for N epochs. 0 disables early stopping.")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use CUDA AMP mixed precision. Default: True")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--download", action="store_true", default=False,
                        help="Download CIFAR-100 if missing")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_state_dict(state_dict, model, reset_head=True):
    """Remove prefixes, optionally remove classifier head, and skip shape-mismatched tensors."""
    model_state = model.state_dict()
    cleaned = {}
    skipped = []

    for k, v in state_dict.items():
        # Common checkpoint prefixes from DataParallel / DDP / EMA wrappers
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]

        # ImageNet-100 and CIFAR-100 both have 100 classes, but class semantics differ.
        # Therefore the classifier should be re-initialized for transfer learning.
        if reset_head and k.startswith("head."):
            skipped.append(k)
            continue

        if k not in model_state:
            skipped.append(k)
            continue

        if model_state[k].shape != v.shape:
            skipped.append(k)
            continue

        cleaned[k] = v

    return cleaned, skipped


def load_transfer_checkpoint(model, checkpoint_path, reset_head=True, require_checkpoint=False):
    if not checkpoint_path:
        print("No checkpoint path provided. Training from random initialization.")
        return

    if not os.path.exists(checkpoint_path):
        msg = f"Checkpoint not found: {checkpoint_path}"
        if require_checkpoint:
            raise FileNotFoundError(msg)
        print(msg)
        print("Training from random initialization.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned, skipped = clean_state_dict(state_dict, model, reset_head=reset_head)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    print(f"Loaded transfer checkpoint: {checkpoint_path}")
    print(f"Loaded tensors: {len(cleaned)}")
    print(f"Skipped tensors: {len(skipped)}")
    if reset_head:
        print("Classifier head was reset for CIFAR-100 transfer learning.")
    if len(missing) > 0:
        print(f"Missing keys after loading: {len(missing)}")
    if len(unexpected) > 0:
        print(f"Unexpected keys after loading: {len(unexpected)}")


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== CIFAR-100 Transfer Config ==========")
    print(f"device          = {device}")
    print(f"model           = {args.model}")
    print(f"data_root       = {args.data_root}")
    print(f"checkpoint_path = {args.checkpoint_path}")
    print(f"reset_head      = {args.reset_head}")
    print(f"epochs          = {args.epochs}")
    print(f"batch_size      = {args.batch_size}")
    print(f"lr              = {args.lr}")
    print(f"output_dir      = {args.output_dir}")
    print(f"early_stop      = {args.early_stop_patience}")
    print("==============================================")

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda")
    )

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes
    )

    load_transfer_checkpoint(
        model,
        args.checkpoint_path,
        reset_head=args.reset_head,
        require_checkpoint=args.require_checkpoint
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    summary_path = output_dir / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr", "train_loss", "test_loss", "test_acc1", "test_acc5", "best_acc1"
        ])

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        avg_train_loss = train_loss_sum / train_count

        model.eval()
        test_loss_sum = 0.0
        test_count = 0
        acc1_sum = 0.0
        acc5_sum = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                bs = images.size(0)
                test_loss_sum += loss.item() * bs
                acc1_sum += acc1.item() * bs
                acc5_sum += acc5.item() * bs
                test_count += bs

        avg_test_loss = test_loss_sum / test_count
        acc1 = acc1_sum / test_count
        acc5 = acc5_sum / test_count
        current_lr = optimizer.param_groups[0]["lr"]

        improved = acc1 > best_acc1
        if improved:
            best_acc1 = acc1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "acc1": acc1,
                    "acc5": acc5,
                    "args": vars(args),
                },
                output_dir / "model_best.pth.tar"
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "acc1": acc1,
                "acc5": acc5,
                "args": vars(args),
            },
            output_dir / "last.pth.tar"
        )

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_lr,
                avg_train_loss,
                avg_test_loss,
                acc1,
                acc5,
                best_acc1,
            ])

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"LR: {current_lr:.2e} "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Test Loss: {avg_test_loss:.4f} "
            f"Acc@1: {acc1:.2f}% "
            f"Acc@5: {acc5:.2f}% "
            f"Best: {best_acc1:.2f}% (epoch {best_epoch})"
        )

        scheduler.step()

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch+1}: "
                f"no Acc@1 improvement for {args.early_stop_patience} epochs."
            )
            break

    print(f"Best CIFAR-100 Acc@1: {best_acc1:.2f}% at epoch {best_epoch}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
