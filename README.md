# MambaOut 論文復現與延伸實驗

## 專案簡介

本專案為論文 **MambaOut: Do We Really Need Mamba for Vision?** 的復現與延伸實驗。

本研究使用官方 GitHub 提供之 PyTorch 實作與 pretrained weights，並以 `mambaout_tiny` 作為主要復現模型。由於原論文主要於 ImageNet-1K 上進行訓練與評估，但完整重現需要較高的計算資源與長時間訓練，因此本研究改以 ImageNet-100 作為簡化版資料集進行 fine-tuning。

除了完成 ImageNet-100 的模型復現外，本研究亦進一步設計：

1. Cross-Dataset Transfer Learning（跨資料集遷移實驗）
2. Freeze Backbone vs Full Fine-tuning 實驗

以分析 MambaOut pretrained feature 的泛化與遷移能力。

---

# 論文資訊

* Paper: MambaOut: Do We Really Need Mamba for Vision?
* Conference: CVPR 2025
* Framework: PyTorch
* Official Model: `mambaout_tiny`

---

# 實驗環境

| 項目        | 設定                   |
| --------- | -------------------- |
| GPU       | NVIDIA RTX 3080 10GB |
| Framework | PyTorch              |
| CUDA      | CUDA 12.1            |
| Python    | Python 3.9           |
| timm      | 0.6.13               |
| OS        | Windows              |

---

# 資料集

## 1. ImageNet-100

由於 ImageNet-1K 訓練成本較高，因此本研究使用 ImageNet-100 作為簡化版資料集進行復現。

資料夾結構：

```text
ImageNet100/
├── train/
└── val/
```

---

## 2. CIFAR-100

作為跨資料集實驗使用，用於分析模型在不同資料分布下的泛化能力。

本研究使用 torchvision 自動下載：

```python
from torchvision import datasets

datasets.CIFAR100(
    root="./data",
    train=True,
    download=True
)
```

---

# 復現流程

## Step 1. Clone 官方 Repo

```bash
git clone https://github.com/yuweihao/MambaOut.git
cd MambaOut
```

---

## Step 2. 建立虛擬環境

```bash
python -m venv my_venv
```

啟用環境：

```bash
my_venv\Scripts\activate
```

---

## Step 3. 安裝套件

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.6.13
```

---

## Step 4. 驗證 GPU 與模型

```bash
python test_import.py
python test_forward.py
```

成功輸出：

```text
CUDA: True
Model loaded!
```

---

# ImageNet-100 Fine-tuning

## 訓練指令

```bash
python train_imagenet.py data/ImageNet100 --model mambaout_tiny --pretrained --num-classes 100 -b 32 --epochs 10 --lr 1e-4 --img-size 224 --workers 4 --amp --output outputs --experiment test_run
```

---

## 修改內容

由於官方 pretrained weights 為 ImageNet-1K 的 1000 類分類模型，因此本研究於載入 checkpoint 時移除原始 classification head 權重，並使用：

```python
strict=False
```

---

# ImageNet-100 復現結果

| Model         | Dataset      | Training Strategy | Top-1 Accuracy |
| ------------- | ------------ | ----------------- | -------------: |
| MambaOut-Tiny | ImageNet-100 | Full Fine-tuning  |         89.72% |

---

# Cross-Dataset Transfer Learning

## 實驗目的

為了分析 MambaOut pretrained feature 的跨資料集泛化能力，本研究進一步將 ImageNet-100 fine-tuned 後的模型遷移至 CIFAR-100。

ImageNet-100 與 CIFAR-100 雖皆為 100 類分類任務，但兩者資料分布與影像解析度差異明顯：

| Dataset      | 特性              |
| ------------ | --------------- |
| ImageNet-100 | 高解析度自然影像        |
| CIFAR-100    | 32×32 小尺寸低解析度影像 |

---

## CIFAR-100 訓練結果

| Model         | Source Checkpoint             | Dataset   | Top-1 Accuracy |
| ------------- | ----------------------------- | --------- | -------------: |
| MambaOut-Tiny | ImageNet-100 fine-tuned model | CIFAR-100 |         86.28% |


---

# Freeze Backbone vs Full Fine-tuning

## 實驗目的

為了分析 MambaOut pretrained feature extractor 的遷移能力，本研究進一步設計 Freeze Backbone 實驗。

在此設定下，模型 backbone 全部凍結，只更新最後分類頭（classification head），藉此觀察在不更新 backbone 的情況下是否仍能維持良好表現。

---

## Freeze Backbone 設定

| 項目               | 設定                  |
| ---------------- | ------------------- |
| Total Params     | 24.47M              |
| Trainable Params | 1.56M               |
| Frozen Part      | Backbone            |
| Trainable Part   | Classification Head |

---

## Freeze Backbone 結果

| Method           | Trainable Params | Best Top-1 Acc |
| ---------------- | ---------------: | -------------: |
| Full Fine-tuning |           24.47M |         89.72% |
| Freeze Backbone  |            1.56M |         89.54% |



---

# 結果分析

Freeze Backbone 僅訓練分類頭，Top-1 Accuracy 仍可達到 89.54%，與 Full Fine-tuning 的 89.72% 僅相差 0.18%。

此結果表示：

> MambaOut 的 pretrained backbone 已具備良好的影像特徵萃取能力，即使不更新 backbone，只訓練最後分類頭，也能在 ImageNet-100 上取得接近完整微調的分類效能。

此外，在 Freeze Backbone 設定下，可訓練參數量由 24.47M 大幅下降至 1.56M，顯示 pretrained feature 已能有效遷移至 ImageNet-100，而不需要大量更新 backbone 參數。

---

# 結論

本研究成功完成 MambaOut-Tiny 於 ImageNet-100 上的 fine-tuning 復現，並進一步設計：

1. Cross-Dataset Transfer Learning
2. Freeze Backbone vs Full Fine-tuning

兩項延伸實驗。

實驗結果顯示：

* MambaOut pretrained backbone 具有良好的可遷移影像表徵能力
* 即使只訓練分類頭，也能維持接近完整微調的分類效能
* 模型在不同資料分布下仍具備一定泛化能力

顯示 MambaOut-Tiny 不僅可成功於 ImageNet-100 上復現，也能有效遷移至其他影像分類資料集。
