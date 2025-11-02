import numpy as np
import os
from typing import Any, Dict, List, Tuple
from logger import Logger
from dataio.gtsrb_dataset import GTSRBDataset
from train import train
from dataio.transforms import (
    ToCompose,
    ToResize,
    ToRotate,
    ToNoise,
    ToTensor,
    ToNormalize
)
from dataio.dataloader import DataLoader


def k_fold_splits(
    k: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    size = 51840
    indices = np.arange(size)
    rng.shuffle(indices)
    splits = []

    fold_sizes = np.full(k, size // k, dtype=int)
    remainder = size % k
    if remainder:
        fold_sizes[:remainder] += 1

    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[start + fold_size:]])
        splits.append((train_idx, val_idx))
        start = stop

    return splits


def cross_validate(
    model: Any,
    loss_fn: Any,
    optimizer: Any,
    num_epochs: int = 10,
    batch_size: int = 32,
    k: int = 5,
    seed: int = 42,
    log_dir: str = "logs",
    checkpoint_root: str = "checkpoints",
) -> List[Dict[str, float]]:
    splits = k_fold_splits(k=k, seed=seed)

    # Define data transforms
    train_transforms = ToCompose([
        ToResize(size=64),
        ToRotate(angle=15),
        ToNoise(mean=0, std=0.05),
        ToTensor(),
        ToNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    val_transforms = ToCompose([
        ToResize(size=64),
        ToTensor(),
        ToNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Directories
        fold_ckpt_dir = os.path.join(checkpoint_root, f"fold_{fold}")
        fold_log_dir = os.path.join(log_dir, f"fold_{fold}")

        # Logger for this fold
        logger = Logger(log_dir=fold_log_dir)
        logger.log_debug(f"Starting fold {fold + 1}/{k}.")

        # Prepare datasets
        train_dataset = GTSRBDataset(
            root="./data/gtsrb/",
            indices=train_idx.tolist(),
            split="train",
            transforms=train_transforms
        )

        val_dataset = GTSRBDataset(
            root="./data/gtsrb/",
            indices=val_idx.tolist(),
            split="val",
            transforms=val_transforms
        )

        print(len(train_dataset), len(val_dataset))

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train
        train_losses, val_losses, train_accs, val_accs = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimiser=optimizer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            checkpoint_dir=fold_ckpt_dir,
            log_dir=fold_log_dir,
        )

        logger.log_debug(f"Finished training fold {fold + 1}/{k}")

        # Metrics
        final_train_loss = train_losses[-1] if train_losses else None
        final_val_loss = val_losses[-1] if val_losses else None
        final_train_acc = train_accs[-1] if train_accs else None
        final_val_acc = val_accs[-1] if val_accs else None

        logger.log_debug(
            f"Fold {fold + 1} results - "
            f"Train Loss: {final_train_loss:.4f}, "
            f"Val Loss: {final_val_loss:.4f}, "
            f"Train Acc: {final_train_acc:.4f}, "
            f"Val Acc: {final_val_acc:.4f}"
        )

        # Append results
        results.append({
            "fold": fold + 1,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "train_acc": final_train_acc,
            "val_acc": final_val_acc,
        })

        # Compute running averages
        avg_train_acc = np.mean([r["train_acc"] for r in results])
        avg_val_acc = np.mean([r["val_acc"] for r in results])
        avg_train_loss = np.mean([r["train_loss"] for r in results])
        avg_val_loss = np.mean([r["val_loss"] for r in results])

        # Summary logger
        summary_logger = Logger(log_dir=log_dir)
        summary_logger.log_debug("Cross-validation complete.")
        summary_logger.log_debug(
            f"Average Train Loss: {avg_train_loss:.4f}, "
            f"Average Val Loss: {avg_val_loss:.4f}, "
            f"Average Train Acc: {avg_train_acc:.4f}, "
            f"Average Val Acc: {avg_val_acc:.4f}"
        )

    return results
