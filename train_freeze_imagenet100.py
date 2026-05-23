import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import models.mambaout

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "./data/ImageNet100"
output_dir = "./outputs/test_run_freeze"
os.makedirs(output_dir, exist_ok=True)

batch_size = 64
epochs = 10
lr = 1e-3

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "val"),
    transform=transform_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

model = timm.create_model(
    "mambaout_tiny",
    pretrained=True,
    num_classes=100
)

# Freeze backbone，只訓練 head
for name, param in model.named_parameters():
    if not name.startswith("head"):
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=0.05
)

best_acc = 0.0
summary_path = os.path.join(output_dir, "summary.csv")

with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_acc"])

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Acc: {acc:.2f}%")

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_loss, acc])

    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": acc,
        },
        os.path.join(output_dir, "last.pth.tar")
    )

    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "acc": acc,
            },
            os.path.join(output_dir, "model_best.pth.tar")
        )

print(f"Best Freeze Backbone Accuracy: {best_acc:.2f}%")
print(f"Results saved to: {output_dir}")