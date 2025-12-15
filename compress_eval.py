import argparse
import torch
import torch.nn as nn
import itertools
import wandb

from models.mobilenet_cifar import mobilenet_v2_cifar10
from utils.dataset import get_cifar10_loaders
from utils.train_utils import evaluate
from compression.quantization import (
    quantize_model,
    model_size_bytes_fp32,
    model_size_bytes_quantized,
    calc_compression_ratio,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--weight_bits_list", nargs="+", type=int, default=[8, 6, 4, 2])
    parser.add_argument("--activation_bits_list", nargs="+", type=int, default=[8, 6, 4, 2])

    parser.add_argument("--wandb_project", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    _, test_loader = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size)

    # Load baseline model
    model = mobilenet_v2_cifar10()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Baseline evaluation
    base_loss, base_acc = evaluate(model, test_loader, criterion, device)
    baseline_bytes = model_size_bytes_fp32(model)
    baseline_mb = baseline_bytes / (1024 ** 2)

    print(f"Baseline size (fp32): {baseline_mb:.3f} MB, accuracy: {base_acc:.2f}")

    # Sweep over quantization configs
    for w_bits, a_bits in itertools.product(args.weight_bits_list, args.activation_bits_list):

        # ✅ NEW: Create a separate W&B run for each config
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=f"w{w_bits}_a{a_bits}",
                config={
                    "weight_quant_bits": w_bits,
                    "activation_quant_bits": a_bits,
                }
            )

        print(f"\nQuantizing: weight_bits={w_bits}, activation_bits={a_bits}")

        # Quantize model
        q_model = quantize_model(model, weight_bits=w_bits, activation_bits=a_bits)
        q_model = q_model.to(device)

        # Evaluate quantized model
        q_loss, q_acc = evaluate(q_model, test_loader, criterion, device)

        # Compute size + compression
        q_bytes = model_size_bytes_quantized(q_model)
        q_mb = q_bytes / (1024 ** 2)
        comp_ratio = calc_compression_ratio(baseline_bytes, q_bytes)

        print(f"Quantized size: {q_mb:.3f} MB, accuracy: {q_acc:.2f}, compression: {comp_ratio:.2f}x")

        # ✅ Log to W&B
        if args.wandb_project:
            wandb.log({
                "quantized_acc": q_acc,
                "quantized_loss": q_loss,
                "model_size_mb": q_mb,
                "compression_ratio": comp_ratio,
            })
            wandb.finish()


if __name__ == "__main__":
    main()
