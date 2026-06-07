import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Plot all experiment results for MambaOut reproduction project
# ============================================================

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 300,
})


def find_summary(folder_name: str) -> Path:
    """Try common locations for summary.csv."""
    candidates = [
        Path("output") / folder_name / "summary.csv",
        Path("outputs") / folder_name / "summary.csv",
        Path(folder_name) / "summary.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_summary(folder_name: str):
    path = find_summary(folder_name)
    if not path.exists():
        print(f"[WARN] summary.csv not found: {path}")
        return None
    df = pd.read_csv(path)
    if "epoch" in df.columns:
        df = df.sort_values("epoch").copy()
    print(f"[OK] Loaded {folder_name}: {path}")
    return df


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_single_curve(df, x_col, y_col, title, ylabel, save_name, ylim=None):
    plt.figure()
    plt.plot(df[x_col], df[y_col], linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / save_name, dpi=300)
    plt.close()


def plot_multi_curve(series, title, ylabel, save_name, ylim=None):
    plt.figure()
    for label, df, x_col, y_col in series:
        plt.plot(df[x_col], df[y_col], linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / save_name, dpi=300)
    plt.close()


def plot_bar(labels, values, title, ylabel, save_name, ylim=None):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / save_name, dpi=300)
    plt.close()


# ============================================================
# Experiment folders
# ============================================================

EXPERIMENTS = {
    "Original ImageNet-100 Scratch": "mambaout_tiny_imagenet100_scratch",
    "ECA ImageNet-100 50ep": "mambaout_tiny_eca_imagenet100_50ep",
    "ECA ImageNet-100 300ep": "mambaout_tiny_eca_imagenet100_scratch_300ep",
    "Original CIFAR-100 Transfer 300ep": "cifar100_transfer_mambaout_tiny_300ep",
    "ECA CIFAR-100 Transfer 300ep": "cifar100_transfer_mambaout_tiny_eca_300ep",
}

data = {name: load_summary(folder) for name, folder in EXPERIMENTS.items()}
if data.get("Original ImageNet-100 Scratch") is not None:
    data["Original ImageNet-100 Scratch"] = data["Original ImageNet-100 Scratch"][
        data["Original ImageNet-100 Scratch"]["epoch"] <= 300
    ].copy()

# ============================================================
# 1. Individual curves
# ============================================================

for name, df in data.items():
    if df is None:
        continue

    x_col = "epoch"
    loss_col = pick_col(df, ["train_loss", "loss"])
    acc_col = pick_col(df, ["eval_top1", "test_acc1", "test_acc", "top1", "val_acc"])

    safe_name = name.lower().replace(" ", "_").replace("-", "").replace("/", "_")

    if loss_col is not None:
        plot_single_curve(
            df, x_col, loss_col,
            title=f"{name} - Training Loss",
            ylabel="Training Loss",
            save_name=f"{safe_name}_train_loss.png",
        )

    if acc_col is not None:
        plot_single_curve(
            df, x_col, acc_col,
            title=f"{name} - Top-1 Accuracy",
            ylabel="Top-1 Accuracy (%)",
            save_name=f"{safe_name}_top1.png",
            ylim=(0, 100),
        )


# ============================================================
# 2. Original vs ECA ImageNet-100 50-epoch comparison
# ============================================================

orig = data.get("Original ImageNet-100 Scratch")
eca50 = data.get("ECA ImageNet-100 50ep")

if orig is not None and eca50 is not None:
    orig_50 = orig[orig["epoch"] <= 50].copy()
    eca_50 = eca50[eca50["epoch"] <= 50].copy()

    orig_acc = pick_col(orig_50, ["eval_top1", "test_acc1", "test_acc", "top1"])
    eca_acc = pick_col(eca_50, ["eval_top1", "test_acc1", "test_acc", "top1"])

    if orig_acc and eca_acc:
        plot_multi_curve(
            [
                ("MambaOut-Tiny", orig_50, "epoch", orig_acc),
                ("MambaOut-Tiny-ECA", eca_50, "epoch", eca_acc),
            ],
            title="ImageNet-100 Short-run Ablation: Original vs ECA",
            ylabel="Top-1 Accuracy (%)",
            save_name="imagenet100_original_vs_eca_50ep_top1.png",
            ylim=(0, 90),
        )


# ============================================================
# 3. CIFAR-100 transfer comparison
# ============================================================

cifar_series = []
for label in ["Original CIFAR-100 Transfer 300ep", "ECA CIFAR-100 Transfer 300ep"]:
    df = data.get(label)
    if df is None:
        continue
    acc_col = pick_col(df, ["test_acc1", "test_acc", "eval_top1", "top1"])
    if acc_col:
        cifar_series.append((label, df, "epoch", acc_col))

if len(cifar_series) >= 2:
    plot_multi_curve(
        cifar_series,
        title="CIFAR-100 Transfer Learning Comparison",
        ylabel="Top-1 Accuracy (%)",
        save_name="cifar100_transfer_comparison_top1.png",
        ylim=(60, 90),
    )


# ============================================================
# 4. Top-1 bar chart
# ============================================================

bar_labels = []
bar_values = []

# Manually include known results when needed.
known_results = {
    "Original ImageNet-100 Scratch": 88.18,
    "ECA ImageNet-100 50ep": 82.22,
    "ECA ImageNet-100 300ep": 87.84,
    "Original CIFAR-100 Transfer 300ep": 82.01,
    "ECA CIFAR-100 Transfer 300ep": 81.83,
}

for label, value in known_results.items():
    bar_labels.append(label)
    bar_values.append(value)

# If ECA 300ep ImageNet-100 exists, read its best value automatically.
eca300 = data.get("ECA ImageNet-100 300ep")
if eca300 is not None:
    acc_col = pick_col(eca300, ["eval_top1", "test_acc1", "test_acc", "top1"])
    if acc_col:
        bar_labels.append("ECA ImageNet-100 300ep")
        bar_values.append(float(eca300[acc_col].max()))

plot_bar(
    labels=bar_labels,
    values=bar_values,
    title="Top-1 Accuracy Summary",
    ylabel="Top-1 Accuracy (%)",
    save_name="top1_summary_bar.png",
    ylim=(70, 92),
)


# ============================================================
# 5. Export numeric summary table
# ============================================================

summary_rows = []
for name, df in data.items():
    if df is None:
        continue

    acc_col = pick_col(df, ["eval_top1", "test_acc1", "test_acc", "top1", "val_acc"])
    loss_col = pick_col(df, ["eval_loss", "test_loss", "loss"])
    train_loss_col = pick_col(df, ["train_loss"])

    row = {
        "experiment": name,
        "epochs": int(df["epoch"].max()) if "epoch" in df.columns else len(df),
    }

    if acc_col:
        best_idx = df[acc_col].idxmax()
        row["best_top1"] = float(df.loc[best_idx, acc_col])
        row["best_top1_epoch"] = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else None
        row["final_top1"] = float(df[acc_col].iloc[-1])

    if loss_col:
        best_loss_idx = df[loss_col].idxmin()
        row["best_eval_or_test_loss"] = float(df.loc[best_loss_idx, loss_col])
        row["best_loss_epoch"] = int(df.loc[best_loss_idx, "epoch"]) if "epoch" in df.columns else None

    if train_loss_col:
        row["final_train_loss"] = float(df[train_loss_col].iloc[-1])

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(FIG_DIR / "experiment_summary_table.csv", index=False, encoding="utf-8-sig")

print("All figures and experiment_summary_table.csv saved to ./figures/")
