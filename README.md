# MambaOut 論文復現與架構改良實驗

本專案為論文 **MambaOut: Do We Really Need Mamba for Vision?** 的課程復現與延伸實驗。  
本研究以官方 MambaOut Repository 為基礎，使用 `mambaout_tiny` 作為 baseline，並提出加入 **ECA lightweight channel attention** 的改良版本 `mambaout_tiny_eca`。

由於原論文主要在 ImageNet-1K 上進行大規模訓練，完整重現需要大量計算資源；本專題受限於單張 **NVIDIA RTX 3080 10GB**，因此採用 **ImageNet-100** 作為小資料集版本進行 scratch reproduction，並使用 **CIFAR-100** 進行跨資料集 transfer learning 測試。

> 本專案並非完整重現原論文 ImageNet-1K 訓練結果，而是在受限硬體環境下進行小資料集版本復現、架構改良與延伸分析。

---

## Paper Information

- Paper: **MambaOut: Do We Really Need Mamba for Vision?**
- Conference: CVPR 2025
- Official Repository: https://github.com/yuweihao/MambaOut
- Baseline Model: `mambaout_tiny`
- Modified Model: `mambaout_tiny_eca`

---

## Hardware and Environment

| Item | Setting |
|---|---|
| GPU | NVIDIA RTX 3080 10GB |
| OS | Windows |
| Framework | PyTorch |
| Model Library | timm |
| Dataset 1 | ImageNet-100 |
| Dataset 2 | CIFAR-100 |

---

## Dataset Structure

### ImageNet-100

本研究使用 ImageNet-100 作為 ImageNet-1K 的小規模替代資料集。資料夾結構如下：

```text
data/ImageNet100/
├── train/
└── validation/
```

每個 split 內部需為 ImageFolder 格式：

```text
train/
├── n01440764/
├── n01443537/
└── ...
```

### CIFAR-100

CIFAR-100 用於跨資料集 transfer learning。  
本專案使用 torchvision 的 CIFAR-100 格式，放置於：

```text
data/cifar-100-python/
```

或由 `torchvision.datasets.CIFAR100` 自動讀取：

```text
data/
├── cifar-100-python/
└── cifar-100-python.tar.gz
```

---

## Experiment Overview

| No. | Experiment | Model | Dataset / Transfer Setting | Epochs | Best Top-1 |
|---:|---|---|---|---:|---:|
| 1 | ImageNet-100 Scratch Reproduction | MambaOut-Tiny | ImageNet-100 | 300 | **88.18%** |
| 2 | Short-run Architecture Ablation | MambaOut-Tiny-ECA | ImageNet-100 | 50 | **82.22%** |
| 3 | Full Architecture Training | MambaOut-Tiny-ECA | ImageNet-100 | 300 | **87.84%** |
| 4 | Cross-Dataset Transfer Baseline | MambaOut-Tiny | ImageNet-100 50ep  checkpoint → CIFAR-100 | 50 | **86.28%** |
| 5 | Cross-Dataset Transfer with Improved Architecture | MambaOut-Tiny-ECA | ImageNet-100 300ep  checkpoint → CIFAR-100 | 50 | **81.83%** |

---

## Experiment 1: ImageNet-100 Scratch Reproduction

### Goal

驗證官方 `mambaout_tiny` 是否能在 ImageNet-100 上從 random initialization 正常收斂。

### Setting

| Item | Setting |
|---|---|
| Model | `mambaout_tiny` |
| Dataset | ImageNet-100 |
| Initialization | Random initialization |
| Epochs | 300 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Effective Batch Size | 256 |
| GPU | RTX 3080 10GB |

### Command

```bash
python train.py data/ImageNet100 --model mambaout_tiny --num-classes 100 --train-split train --val-split validation --epochs 300 --cooldown-epochs 0 --lr 5e-4 --warmup-epochs 20 --warmup-lr 1e-6 --min-lr 1e-5 --sched cosine --opt adamw --weight-decay 0.05 -b 64 --grad-accum-steps 4 --drop-path 0.1 --mixup 0.4 --cutmix 1.0 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel --native-amp --pin-mem -j 8 --output ./output --experiment mambaout_tiny_imagenet100_scratch
```

### Result

| Model | Dataset | Init | Epochs | Best Top-1 | Final Top-1 |
|---|---|---|---:|---:|---:|
| MambaOut-Tiny | ImageNet-100 | Random | 300 | **88.18%** | 87.94% |

---

## Experiment 2: MambaOut-Tiny-ECA Short-run Ablation

### Goal

在原始 MambaOut-Tiny 的 `GatedCNNBlock` 中加入 ECA lightweight channel attention，觀察是否能改善早期收斂表現。

### Architecture Modification

原始 MambaOut 移除 Mamba selective scan，主張 vision classification 不一定需要複雜的 sequence modeling。  
本研究延續此觀點，不重新加入 Mamba，而是在 `GatedCNNBlock` 中加入 **Efficient Channel Attention (ECA)**：

```text
Original:
GatedCNNBlock

Modified:
GatedCNNBlock + ECA Channel Attention
```

此改良目標是以低成本方式強化 channel-wise feature selection。

### Command

```bash
python train.py data/ImageNet100 --model mambaout_tiny_eca --num-classes 100 --train-split train --val-split validation --epochs 50 --cooldown-epochs 0 --lr 5e-4 --warmup-epochs 10 --warmup-lr 1e-6 --min-lr 1e-5 --sched cosine --opt adamw --weight-decay 0.05 -b 64 --grad-accum-steps 4 --drop-path 0.1 --mixup 0.4 --cutmix 1.0 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel --native-amp --pin-mem -j 8 --output ./output --experiment mambaout_tiny_eca_imagenet100_50ep
```

### Result

| Model | Dataset | Epochs | Best Top-1 |
|---|---|---:|---:|
| MambaOut-Tiny, original, same-period reference | ImageNet-100 | 50 | 約 80.86% |
| MambaOut-Tiny-ECA | ImageNet-100 | 50 | **82.22%** |

### Analysis

在 50-epoch short-run ablation 中，MambaOut-Tiny-ECA 達到 **82.22%** Top-1，高於原始 MambaOut-Tiny 同期約 **80.86%**，提升約 **1.36%**。  
此結果顯示 ECA 可能有助於 early-stage feature selection 與早期收斂。

---

## Experiment 3: MambaOut-Tiny-ECA Full ImageNet-100 Training

### Goal

完整訓練 MambaOut-Tiny-ECA 300 epochs，檢查 ECA 是否能提升最終 ImageNet-100 accuracy。

### Command

```bash
python train.py data/ImageNet100 --model mambaout_tiny_eca --num-classes 100 --train-split train --val-split validation --epochs 300 --cooldown-epochs 0 --lr 5e-4 --warmup-epochs 20 --warmup-lr 1e-6 --min-lr 1e-5 --sched cosine --opt adamw --weight-decay 0.05 -b 64 --grad-accum-steps 4 --drop-path 0.1 --mixup 0.4 --cutmix 1.0 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel --native-amp --pin-mem -j 8 --output ./output --experiment mambaout_tiny_eca_imagenet100_scratch_300ep
```

### Result

| Model | Dataset | Epochs | Best Top-1 | Best Epoch | Final Top-1 |
|---|---|---:|---:|---:|---:|
| MambaOut-Tiny | ImageNet-100 | 300 | **88.18%** | 306 | 87.94% |
| MambaOut-Tiny-ECA | ImageNet-100 | 300 | **87.84%** | 265 | 87.52% |

### Analysis

MambaOut-Tiny-ECA 在完整 300-epoch training 後達到 **87.84%** Top-1，略低於原始 MambaOut-Tiny 的 **88.18%**。  
因此，本研究觀察到：

- ECA 對早期訓練收斂有幫助。
- 但在完整 300 epochs 後，ECA 並未提升最終 ImageNet-100 Top-1 accuracy。
- 兩者差距僅約 0.34%，可能受到單次訓練隨機性、資料集規模與 regularization 設定影響。

---

## Experiment 4: Cross-Dataset Transfer Baseline

### Goal

建立原始 MambaOut-Tiny 的 CIFAR-100 transfer baseline，作為改良版模型的對照組。

### Setting

| Item | Setting |
|---|---|
| Source Dataset | ImageNet-100 |
| Target Dataset | CIFAR-100 |
| Source Model | MambaOut-Tiny |
| Target Model | MambaOut-Tiny |
| Transfer Strategy | Load ImageNet-100 checkpoint, reset classification head, fine-tune on CIFAR-100 |

### Result

| Model | Source Checkpoint | Target Dataset | Fine-tuning Epochs | Best Top-1 |
|---|---|---|---:|---:|
| MambaOut-Tiny | ImageNet-100 checkpoint | CIFAR-100 | 50 | **86.28%** |

---

## Experiment 5: Cross-Dataset Transfer with MambaOut-Tiny-ECA

### Goal

測試改良架構 MambaOut-Tiny-ECA 是否能從 ImageNet-100 transfer 到 CIFAR-100。

### Setting

| Item | Setting |
|---|---|
| Source Dataset | ImageNet-100 |
| Target Dataset | CIFAR-100 |
| Source Model | MambaOut-Tiny-ECA |
| Source Checkpoint | ImageNet-100 300-epoch checkpoint |
| Transfer Strategy | Load backbone, reset CIFAR-100 classification head |
| Fine-tuning Epochs | 50 |

### Command

```bash
python train_cifar100.py --model mambaout_tiny_eca --data-root ./data --checkpoint-path ./output/mambaout_tiny_eca_imagenet100_scratch_300ep/model_best.pth.tar --output-dir ./output/cifar100_transfer_mambaout_tiny_eca_300ep --epochs 50 --batch-size 64 --workers 4 --lr 1e-4 --require-checkpoint
```

### Result

| Model | Source Checkpoint | Target Dataset | Fine-tuning Epochs | Best Top-1 | Final Top-1 |
|---|---|---|---:|---:|---:|
| MambaOut-Tiny-ECA | ImageNet-100 300ep checkpoint | CIFAR-100 | 50 | **81.83%** | 81.75% |

### Comparison

| Transfer Experiment | Best Top-1 |
|---|---:|
| Original MambaOut-Tiny → CIFAR-100 | **86.28%** |
| MambaOut-Tiny-ECA 300ep → CIFAR-100 | **81.83%** |

### Analysis

MambaOut-Tiny-ECA 使用完整 300-epoch ImageNet-100 checkpoint 轉移到 CIFAR-100 後，達到 **81.83%** Top-1，表示改良後架構可成功 transfer 至 CIFAR-100。  
然而，此結果仍低於原始 MambaOut-Tiny transfer baseline 的 **86.28%**，因此目前不能證明 ECA 改良能提升跨資料集泛化能力。

可能原因包括：

1. ECA 強化 ImageNet-100 上的 channel-wise feature selection，但這些特徵未必更適合 CIFAR-100。
2. CIFAR-100 解析度較低，資料分布與 ImageNet-100 不同。
3. CIFAR-100 fine-tuning 過程中觀察到 train loss 快速下降，test loss 後期上升，顯示存在 overfitting。
4. ECA 的 early-stage improvement 不一定能直接轉化為更好的 cross-dataset transfer performance。

---

## Figure Generation

本專案提供圖表產生程式：

```bash
python plot_results_all_experiments.py
```

產生主要曲線與比較圖：

```text
figures/
├── original_imagenet100_scratch_top1.png
├── original_imagenet100_scratch_train_loss.png
├── eca_imagenet100_50ep_top1.png
├── eca_imagenet100_50ep_train_loss.png
├── eca_imagenet100_300ep_top1.png
├── eca_imagenet100_300ep_train_loss.png
├── original_cifar100_transfer_top1.png
├── original_cifar100_transfer_train_loss.png
├── eca_cifar100_transfer_300ep_top1.png
├── eca_cifar100_transfer_300ep_train_loss.png
├── imagenet100_original_vs_eca_50ep_top1.png
├── cifar100_transfer_comparison_top1.png
├── top1_summary_bar.png
└── experiment_summary_table.csv
```

若要產生 CIFAR-100 confusion matrix：

```bash
python plot_confusion_matrix_transfer.py --model mambaout_tiny_eca --checkpoint-path ./output/cifar100_transfer_mambaout_tiny_eca_300ep/model_best.pth.tar --data-root ./data --output-dir ./figures
```

輸出：

```text
figures/cifar100_confusion_matrix_mambaout_tiny_eca_top20.png
```

---

## Final Summary

| Experiment | Model | Dataset / Setting | Best Top-1 |
|---|---|---|---:|
| ImageNet-100 Scratch Reproduction | MambaOut-Tiny | ImageNet-100 | **88.18%** |
| Short-run Ablation | MambaOut-Tiny-ECA | ImageNet-100, 50 epochs | **82.22%** |
| Full Training | MambaOut-Tiny-ECA | ImageNet-100, 300 epochs | **87.84%** |
| Transfer Baseline | MambaOut-Tiny | ImageNet-100 → CIFAR-100 | **86.28%** |
| Improved Transfer | MambaOut-Tiny-ECA | ImageNet-100 300ep → CIFAR-100 | **81.83%** |

---

## Conclusion

本專題完成了 MambaOut-Tiny 在 ImageNet-100 上的小資料集版本 scratch reproduction，並進一步提出加入 ECA lightweight channel attention 的改良架構 MambaOut-Tiny-ECA。

實驗結果顯示：

1. 原始 MambaOut-Tiny 可在 ImageNet-100 上從 random initialization 正常收斂，達到 **88.18%** Top-1。
2. MambaOut-Tiny-ECA 在 50-epoch short-run ablation 中達到 **82.22%** Top-1，高於原始模型同期表現，顯示 ECA 有助於早期收斂。
3. 但在完整 300-epoch training 後，MambaOut-Tiny-ECA 達到 **87.84%** Top-1，略低於原始 MambaOut-Tiny 的 **88.18%**。
4. 在 CIFAR-100 transfer learning 中，MambaOut-Tiny-ECA 達到 **81.83%** Top-1，但仍低於原始 MambaOut-Tiny baseline 的 **86.28%**。

因此，本研究認為 ECA attention 對 ImageNet-100 的早期特徵學習有幫助，但目前未證明能提升最終 accuracy 或跨資料集泛化能力。此結果也說明，source dataset 上的 early-stage improvement 不一定能直接轉換為更好的 transfer learning performance。

---

## Repository Notes

本 repository 不包含：

- ImageNet-100 dataset
- CIFAR-100 dataset
- Training checkpoints (`*.pth`, `*.pth.tar`)
- `output/` training folders
- Python virtual environment

請依照 README 中的資料夾結構自行準備資料集，或使用 torchvision 下載 CIFAR-100。
