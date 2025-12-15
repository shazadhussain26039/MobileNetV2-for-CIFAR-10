import argparse
import os
import torch
import torch.nn as nn

from models.mobilenet_cifar import mobilenet_v2_cifar10
from utils.dataset import get_cifar10_loaders
from utils.train_utils import get_optimizer_scheduler, train_one_epoch, evaluate

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--bn_eps", type=float, default=1e-5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args), name="baseline")

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = mobilenet_v2_cifar10(
        num_classes=10,
        width_mult=args.width_mult,
        dropout=args.dropout,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        epochs=args.epochs,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if args.wandb_project and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.save_path, "mobilenetv2_cifar_baseline.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "acc": best_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )

    print(f"Best Val Acc: {best_acc:.2f}")


if __name__ == "__main__":
    main()
