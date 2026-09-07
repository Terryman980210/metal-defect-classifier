# 【2026年最新】金属表面欠陥分類CNNの選び方と実装 〜VGGからSwin Transformerへ〜

## はじめに

銅の展伸材などの金属表面外観検査でVGG16/19を使っていましたが、以下の課題がありました。

- GradCAMがうまく機能せず「なぜその結果か」が分からない
- VGGは古く、最新アーキテクチャを検討したい

本記事では**2026年3月発表の最新ベンチマーク研究**をもとに、金属欠陥分類に最適なモデルを選定し、PyTorchによる実装を紹介します。

## 対象読者・環境

- 工業用外観検査でCNNを活用している方
- XAI（説明可能AI）の導入を検討している方
- GPU: RTX 4090 / グレースケール256×256 / 約20クラス / 各2000枚

## 最新ベンチマーク結果（金属欠陥専用）

2026年3月にApplied Sciences誌に掲載された論文[^1]が、**金属表面欠陥分類に特化した初の大規模ベンチマーク**を公開しました。

10種類のアーキテクチャを5つのデータセット（NEU-DET・X-SDD・KolektorSDD2・DAGM・MTDD）で統一条件評価しています。

| # | モデル | 平均F1スコア | 推論速度 | 外観検査適性 |
|---|---|---|---|---|
| 🥇 | **Swin Transformer** | **90.8%** | ~9.5ms | ◎ 精度最優先 |
| 🥈 | **CoAtNet** | **85.5%** | 中速 | ◎ バランス最良 |
| 🥉 | ResNet50 | 高い | 中速 | ○ |
| — | MobileNetV2 | 81~84% | ~4ms | △ エッジ向け |
| ⚠️ | **EfficientNetV2** | **41.9%** | 遅い | ✗ |

### ⚠️ 重要：EfficientNetV2は金属欠陥タスクに不向き

ImageNetで強力なEfficientNetV2が、金属欠陥分類では41.9%と大きく低迷しました。

原因は以下の通りです。

- 224×224への積極的なリサイズ → 微細欠陥情報が欠落
- ImageNet分布と産業用テクスチャの分布差が大きい

**汎用ベンチマークの結果をそのままドメイン特化タスクに転用できない**ことが実証されました。

## なぜVGGのGradCAMはうまくいかないのか

```
VGG19の構造上の問題:
  Conv → ... → Flatten → FC(4096) → FC(4096) → FC(num_classes)
                  ↑
          ここで空間情報が崩壊 → GradCAMが粗くなる

Swin Transformer（Attention Mapが鮮明）:
  PatchEmbed → Stage1〜4（Attention保持） → GAP → 出力
```

VGGはFC層が巨大で空間情報が失われやすく、GradCAMの可視化が粗くなります。
Swin TransformerはAttention Mapが自然に可視化できるため、「どこを見て判断したか」が直感的に分かります。

## 実装

コードは以下のGitHubリポジトリで公開しています。

👉 **[https://github.com/YOUR_USERNAME/metal-defect-classifier](https://github.com/YOUR_USERNAME/metal-defect-classifier)**

### セットアップ

```bash
git clone https://github.com/YOUR_USERNAME/metal-defect-classifier.git
cd metal-defect-classifier
pip install -r requirements.txt
```

### データセット構成

```
your_dataset/
├── train/
│   ├── class_01/   *.png (グレースケール)
│   ├── class_02/
│   └── ...
├── val/
└── test/
```

### グレースケール→転移学習の工夫

グレースケール画像をそのまま使うのではなく、**3チャネルに複製**することでImageNet事前学習の恩恵を受けます。

```python
# dataset.py より抜粋
transforms.Grayscale(num_output_channels=3),  # 1ch→3ch複製
transforms.Resize((256, 256)),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(15),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
```

### モデルの定義（timmを使用）

```python
import timm

# Swin Transformer Base（256×256専用設定）
model = timm.create_model(
    'swin_base_patch4_window8_256',
    pretrained=True,
    num_classes=20,   # あなたのクラス数
    in_chans=3,
)

# CoAtNet（CNN+Transformerハイブリッド）
model = timm.create_model(
    'coatnet_2_rw_224',
    pretrained=True,
    num_classes=20,
)

# ConvNeXt V2（GradCAM++との相性良好）
model = timm.create_model(
    'convnextv2_base',
    pretrained=True,
    num_classes=20,
)
```

### 学習の実行

```bash
# Swin Transformerで学習（推奨）
python src/train.py \
    --data_dir    /path/to/your/dataset \
    --model       swin_base \
    --num_classes 20 \
    --img_size    256 \
    --batch_size  32 \
    --epochs      100 \
    --lr          1e-4 \
    --amp         # RTX4090でAMP有効化
```

学習の主なポイントです。

- **AdamW + CosineAnnealingLR** でTransformerを安定学習
- **label_smoothing=0.1** で過学習を抑制
- **WeightedRandomSampler** でクラス不均衡に対応
- **Early Stopping（patience=15）** で最良モデルを自動保存
- **AMP（自動混合精度）** でRTX4090のTensorCoreを活用

### XAI可視化

```bash
# Swin Transformer → Attention Map
bash scripts/run_visualize.sh swin_base /path/to/image.png

# ConvNeXt V2 / CoAtNet → GradCAM++
bash scripts/run_visualize.sh convnextv2_base /path/to/image.png
```

**Swin TransformerのAttention Map**は、VGGのGradCAMと異なり、欠陥部位に明確に注意が集中する鮮明な可視化が得られます。

### 評価・混同行列

```bash
bash scripts/run_eval.sh /path/to/dataset swin_base
```

出力されるファイルです。

| ファイル | 内容 |
|---|---|
| `confusion_matrix.png` | 正規化混同行列 |
| `report.txt` | クラスごとPrecision/Recall/F1 |
| `history.png` | 学習曲線（Loss・Accuracy） |
| `results.json` | 数値サマリー |

## 256×256が有利な理由

ベンチマーク論文では全モデルを224×224に統一リサイズしており、これがEfficientNetV2低性能の一因とされています。

**256×256はその問題を回避でき、より微細な欠陥情報を保持できます。**

加えて、`swin_base_patch4_window8_256`はwindow_size=8で256に最適化されており、224→256の解像度差の影響を最小限に抑えられます。

## モデル比較まとめ

| 比較項目 | VGG16/19 | Swin-Base | CoAtNet-2 | ConvNeXt V2-Base |
|---|---|---|---|---|
| 金属欠陥F1 | — | **90.8%** | **85.5%** | 高い |
| パラメータ数 | 138M | 88M | 75M | 89M |
| GradCAM品質 | ✗ 粗い | — | ○ | ◎ |
| Attention可視化 | ✗ | ◎ | ○ | △ |
| グレースケール対応 | ○ | ○ | ○ | ○ |
| 256×256対応 | ○ | ◎ native | ○ | ○ |

## 移行ロードマップ

```
Step 1【今すぐ】Swin Transformer-Baseで再学習
        → VGGの精度を維持・向上 + Attention Mapで説明可能性を確保

Step 2【余裕があれば】CoAtNetと比較実験
        → 推論速度・精度のトレードオフを確認

Step 3【発展】ConvNeXt V2-Base + GradCAM++
        → 既存のGradCAMワークフローを継続したい場合
```

## まとめ

- VGGのGradCAMが粗い原因はFC層による空間情報の崩壊
- **金属欠陥専用ベンチマークでSwin TransformerとCoAtNetが最高性能**
- EfficientNetV2は汎用タスクで強力でも金属欠陥分類には不向き（41.9%）
- 256×256グレースケール画像は→3ch複製で転移学習を最大活用
- Swin TransformerのAttention Mapは説明可能性の問題を自然に解決

コードはGitHubで公開しています：
👉 **[https://github.com/YOUR_USERNAME/metal-defect-classifier](https://github.com/YOUR_USERNAME/metal-defect-classifier)**

---

## 参考文献

[^1]: Silva et al., "Cross-Dataset Benchmarking of Deep Learning Models for Surface Defect Classification in Metal Parts", *Applied Sciences*, 2026. https://doi.org/10.3390/app16063022

- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021. https://arxiv.org/abs/2103.14030
- Dai et al., "CoAtNet: Marrying Convolution and Attention for All Data Sizes", NeurIPS 2021. https://arxiv.org/abs/2106.04803
- Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", CVPR 2023. https://arxiv.org/abs/2301.00808
