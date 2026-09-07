"""Plot training history from history.json."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="Path to history.json")
    p.add_argument("--output",  default="history.png")
    args = p.parse_args()

    with open(args.history) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="Train Loss")
    ax1.plot(epochs, h["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, h["train_acc"], label="Train Acc")
    ax2.plot(epochs, h["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
