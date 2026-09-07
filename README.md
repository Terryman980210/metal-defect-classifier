# Metal Surface Defect Classifier

銅などの展伸材を含む**金属表面欠陥の分類**に特化した深層学習実装です。  
グレースケール256×256画像・約20クラスの分類を想定しています。

## 背景

2026年3月のApplied Sciences誌に掲載されたベンチマーク研究（DOI: [10.3390/app16063022](https://doi.org/10.3390/app16063022)）では、金属表面欠陥分類において以下の結果が得られています。

| モデル | 平均F1スコア | 推論速度 |
|---|---|---|
| **Swin Transformer** | **90.8%** | ~9.5ms |
| **CoAtNet** | **85.5%** | 中速 |
| ResNet50 | 高い | 中速 |
| EfficientNetV2 | 41.9% ⚠️ | 遅い |

> EfficientNetV2は汎用ベンチマーク（ImageNet）では強力ですが、金属欠陥分類タスクでは低性能でした。**ドメイン特化ベンチマークの重要性**を示す結果です。

## 対応モデル

| モデル名 | timm識別子 | 特徴 |
|---|---|---|
| `swin_base` | `swin_base_patch4_window8_256` | 精度最優先・Attention可視化対応 |
| `coatnet_2` | `coatnet_2_rw_224` | CNN+Transformerハイブリッド |
| `convnextv2_base` | `convnextv2_base` | GradCAM++との相性良好 |

## 特徴

- **グレースケール対応**: 1chグレースケール→3ch複製でImageNet転移学習を活用
- **256×256ネイティブ対応**: Swin Transformerはwindow_size=8で256専用設定
- **クラス不均衡対応**: WeightedRandomSamplerによるバランスサンプリング
- **XAI（説明可能AI）**:
  - Swin Transformer → Attention Map
  - ConvNeXt V2 / CoAtNet → GradCAM++
- **AMP（自動混合精度）**: RTX40系GPUで高速学習
- **Early Stopping**: 過学習防止

## ディレクトリ構成

```
metal-defect-classifier/
├── src/
│   ├── train.py          # 学習スクリプト
│   ├── evaluate.py       # 評価・混同行列
│   ├── visualize.py      # XAI可視化
│   ├── dataset.py        # DataLoader
│   ├── utils.py          # ユーティリティ
│   └── plot_history.py   # 学習曲線プロット
├── configs/
│   ├── swin_base.yaml
│   └── coatnet_2.yaml
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_visualize.sh
├── requirements.txt
└── README.md
```

## データセット構成

```
your_dataset/
├── train/
│   ├── class_01/   *.png
│   ├── class_02/
│   └── ...
├── val/
│   ├── class_01/
│   └── ...
└── test/
    ├── class_01/
    └── ...
```

## セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/Terryman980210/metal-defect-classifier.git
cd metal-defect-classifier

# 依存パッケージのインストール（CUDA環境推奨）
pip install -r requirements.txt
```

## 使い方

### 1. 学習

```bash
# Swin Transformer（推奨）
python src/train.py \
    --data_dir    /path/to/your/dataset \
    --model       swin_base \
    --num_classes 20 \
    --img_size    256 \
    --batch_size  32 \
    --epochs      100 \
    --lr          1e-4 \
    --output_dir  outputs

# シェルスクリプトで実行
bash scripts/run_train.sh /path/to/dataset swin_base
```

### 2. 評価

```bash
bash scripts/run_eval.sh /path/to/dataset swin_base
```

出力されるファイル：
- `confusion_matrix.png` — 正規化混同行列
- `report.txt` — クラスごとのPrecision / Recall / F1
- `history.png` — 学習曲線
- `results.json` — 数値サマリー

### 3. XAI可視化

```bash
bash scripts/run_visualize.sh swin_base /path/to/image.png
```

- `swin_base` → Attention Map
- `convnextv2_base` / `coatnet_2` → GradCAM++

## 学習ポイント

### グレースケール→3チャネル変換

```python
transforms.Grayscale(num_output_channels=3)
```

グレースケール画像をRGBに複製することで、ImageNet事前学習の特徴量（エッジ・テクスチャ表現）を転移学習に活用できます。

### Swin Transformer + Attention Map（可視化）

```
VGG（GradCAMが粗い）の課題:
  Conv → FC(4096) → FC(4096) → 出力
         ↑ここで空間情報が崩壊

Swin Transformer（Attention Mapが鮮明）:
  PatchEmbed → Stage1〜4（Attention保持） → GAP → 出力
```

### 推奨GPU・バッチサイズ

| GPU | 推奨バッチサイズ |
|---|---|
| RTX 4090 (24GB) | 64〜128 |
| RTX 4080 (16GB) | 32〜64 |
| RTX 3090 (24GB) | 32〜64 |

## 参考文献

1. Silva et al., "Cross-Dataset Benchmarking of Deep Learning Models for Surface Defect Classification in Metal Parts", *Applied Sciences*, 2026. [DOI: 10.3390/app16063022](https://doi.org/10.3390/app16063022)

2. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", *ICCV*, 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)

3. Dai et al., "CoAtNet: Marrying Convolution and Attention for All Data Sizes", *NeurIPS*, 2021. [arXiv:2106.04803](https://arxiv.org/abs/2106.04803)

4. Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", *CVPR*, 2023. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)

5. He et al., "Automated Detection of Defects on Metal Surfaces using Vision Transformers", *arXiv*, 2024. [arXiv:2410.04440](https://arxiv.org/abs/2410.04440)

## ライセンス

MIT License
