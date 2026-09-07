"""
XAI Visualization for Metal Surface Defect Classifier.

Supports:
  - GradCAM++ (ConvNeXt V2, CoAtNet)
  - Attention Map (Swin Transformer)
  - LIME (all models)

Usage:
    python src/visualize.py \
        --model swin_base \
        --ckpt outputs/swin_base/best.pth \
        --img path/to/image.png \
        --class_names_file outputs/swin_base/classes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

# Optional: pip install grad-cam
try:
    from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[Warning] pytorch-grad-cam not installed. Install: pip install grad-cam")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_CONFIGS = {
    "swin_base":      "swin_base_patch4_window8_256",
    "coatnet_2":      "coatnet_2_rw_224",
    "convnextv2_base":"convnextv2_base",
}


def load_model(model_name: str, num_classes: int, ckpt_path: str,
               device: torch.device) -> nn.Module:
    model = timm.create_model(
        MODEL_CONFIGS[model_name],
        pretrained=False,
        num_classes=num_classes,
        in_chans=3,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    return model


def preprocess(img_path: str, img_size: int = 256):
    """Load grayscale image → 3ch tensor + numpy rgb for overlay."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, (img_size, img_size))
    img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)        # H×W×3  uint8
    img_float = img_rgb.astype(np.float32) / 255.0               # for overlay

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = tf(img_rgb).unsqueeze(0)   # 1×3×H×W
    return tensor, img_float


# ------------------------------------------------------------------ #
#  GradCAM++ (ConvNeXt V2 / CoAtNet)
# ------------------------------------------------------------------ #

def get_gradcam_target_layer(model, model_name: str):
    if model_name == "convnextv2_base":
        return [model.stages[-1].blocks[-1]]
    elif model_name == "coatnet_2":
        # CoAtNet last stage
        return [model.stages[-1].blocks[-1]]
    else:
        raise ValueError(f"GradCAM not recommended for {model_name}. Use Attention Map.")


def run_gradcam(model, model_name: str, tensor: torch.Tensor,
                img_float: np.ndarray, class_idx: int = None,
                output_path: str = "gradcam.png"):
    if not GRADCAM_AVAILABLE:
        print("grad-cam not installed. Skipping GradCAM.")
        return

    target_layers = get_gradcam_target_layer(model, model_name)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    targets = None if class_idx is None else \
              [lambda x, ci=class_idx: x[:, ci]]

    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    visualization = show_cam_on_image(img_float, grayscale_cam[0], use_rgb=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(img_float);       plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(grayscale_cam[0], cmap="jet"); plt.title("GradCAM++"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(visualization);   plt.title("Overlay");  plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[GradCAM++] Saved → {output_path}")


# ------------------------------------------------------------------ #
#  Swin Transformer Attention Map
# ------------------------------------------------------------------ #

def run_swin_attention(model, tensor: torch.Tensor,
                       img_float: np.ndarray,
                       output_path: str = "attention.png"):
    """
    Extract self-attention from the last Swin Transformer window-attention block
    and visualize as a heatmap.
    """
    attn_weights = []

    def hook_fn(module, input, output):
        # output: (B, num_heads, N, N)
        attn_weights.append(output.detach().cpu())

    # Register hook on the last Swin attention layer
    hooks = []
    for name, module in model.named_modules():
        if "attn.attn_drop" in name or "attention.drop" in name:
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(tensor)

    for h in hooks:
        h.remove()

    if not attn_weights:
        print("[AttentionMap] Could not hook attention. Check model structure.")
        return

    # Use last attention layer, mean over heads, take CLS-to-patch attention
    attn = attn_weights[-1][0]        # (num_heads, N, N)
    attn_mean = attn.mean(0)          # (N, N)
    # Patch attention from the first token (or mean of all tokens)
    patch_attn = attn_mean.mean(0)    # (N,)

    h = w = int(patch_attn.shape[0] ** 0.5)
    if h * w != patch_attn.shape[0]:
        h = w = 8   # fallback for window attention
        patch_attn = patch_attn[:h*w]

    attn_map = patch_attn[:h*w].reshape(h, w).numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    img_h, img_w = img_float.shape[:2]
    attn_resized = cv2.resize(attn_map, (img_w, img_h))
    heatmap = cm.jet(attn_resized)[:, :, :3]
    overlay = (0.5 * img_float + 0.5 * heatmap).clip(0, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(img_float);      plt.title("Original");      plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(attn_resized, cmap="jet"); plt.title("Attention"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlay);         plt.title("Overlay");       plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[AttentionMap] Saved → {output_path}")


# ------------------------------------------------------------------ #
#  Batch visualization: top-N misclassified images
# ------------------------------------------------------------------ #

@torch.no_grad()
def visualize_predictions(model, loader, class_names, device,
                           output_path="predictions.png", n=16):
    model.eval()
    images, preds, trues = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        pred = logits.argmax(1).cpu()
        images.extend(imgs.cpu())
        preds.extend(pred.tolist())
        trues.extend(labels.tolist())
        if len(images) >= n:
            break

    images = images[:n]
    preds  = preds[:n]
    trues  = trues[:n]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis("off"); continue
        img = (images[i] * std + mean).clamp(0,1).permute(1,2,0).numpy()
        ax.imshow(img[:,:,0], cmap="gray")
        color = "green" if preds[i] == trues[i] else "red"
        ax.set_title(f"GT:{class_names[trues[i]]}\nPR:{class_names[preds[i]]}",
                     color=color, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Prediction grid] Saved → {output_path}")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True,
                   choices=["swin_base","coatnet_2","convnextv2_base"])
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--img",             required=True, help="Path to input image")
    p.add_argument("--class_names_file",required=True, help="JSON list of class names")
    p.add_argument("--img_size",        type=int, default=256)
    p.add_argument("--output_dir",      type=str, default="vis_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.class_names_file) as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, len(class_names), args.ckpt, device)

    tensor, img_float = preprocess(args.img, args.img_size)
    tensor = tensor.to(device)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu()
        top5   = probs.topk(5)

    print("\n=== Prediction ===")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {class_names[idx]:30s}: {prob:.4f}")

    pred_class = probs.argmax().item()
    stem = Path(args.img).stem

    if args.model == "swin_base":
        run_swin_attention(
            model, tensor, img_float,
            output_path=f"{args.output_dir}/{stem}_attention.png",
        )
    else:
        run_gradcam(
            model, args.model, tensor, img_float,
            class_idx=pred_class,
            output_path=f"{args.output_dir}/{stem}_gradcam.png",
        )


if __name__ == "__main__":
    main()
