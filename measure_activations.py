import argparse
import torch
import torch.nn as nn

from models.mobilenet_cifar import mobilenet_v2_cifar10
from utils.dataset import get_cifar10_loaders
from compression.quantization import quantize_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--activation_bits", type=int, default=8)
    return parser.parse_args()


def get_activation_size_bytes(tensor: torch.Tensor, bits: int):
    """
    Compute activation memory in bytes.
    """
    numel = tensor.numel()
    return (numel * bits) / 8.0


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, _ = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size)

    # Load baseline model
    model = mobilenet_v2_cifar10()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Get one batch of data
    images, _ = next(iter(train_loader))
    images = images.to(device)

    # Forward pass to capture activations
    activations_fp32 = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            activations_fp32.append(out.detach())

    # Register hooks on all Conv2d and Linear layers
    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    # Run forward pass
    with torch.no_grad():
        _ = model(images)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute fp32 activation size
    fp32_total_bytes = 0
    for act in activations_fp32:
        fp32_total_bytes += get_activation_size_bytes(act, bits=32)

    # Compute quantized activation size
    q_total_bytes = 0
    for act in activations_fp32:
        q_act, scale = quantize_tensor(act, args.activation_bits)
        q_total_bytes += get_activation_size_bytes(q_act, bits=args.activation_bits)

    # Compression ratio
    compression_ratio = fp32_total_bytes / q_total_bytes

    print("\n=== Activation Compression Measurement ===")
    print(f"FP32 activation size: {fp32_total_bytes / (1024**2):.3f} MB")
    print(f"Quantized activation size ({args.activation_bits} bits): {q_total_bytes / (1024**2):.3f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")


if __name__ == "__main__":
    main()
