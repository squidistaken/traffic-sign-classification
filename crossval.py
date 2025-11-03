import numpy as np
import os
from typing import Any, Dict, List, Tuple, Callable
from logger import Logger
from dataio.gtsrb_dataset import GTSRBDataset
from train import train
from dataio.dataloader import DataLoader


def k_fold_splits(dataset: GTSRBDataset, k: int = 5, seed: int = 42
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold cross-validation splits.

    This function divides the dataset into k folds, each used once as the
    validation set while the remaining k-1 folds form the training set.

    Args:
        k (int, optional): The number of folds. Defaults to 5.
        dataset (GTSRBDataset): The dataset to split in k folds.
        seed (int, optional): The random seed for reproducibility. Defaults to
                              42.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of (train, validation)
                                             index splits.
    """
    rng = np.random.default_rng(seed)

    # Hardcoded size.
    size = len(dataset)
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
        train_idx = np.concatenate([indices[:start],
                                    indices[start + fold_size:]])

        splits.append((train_idx, val_idx))

        start = stop

    return splits


def cross_validate(
    model_fn: Callable,
    dataset: GTSRBDataset,
    loss_fn: Callable,
    optimizer_fn: Callable,
    train_transforms: Any,
    val_transforms: Any,
    num_epochs: int = 10,
    batch_size: int = 32,
    k: int = 5,
    seed: int = 42,
    log_dir: str = "logs",
    checkpoint_root: str = "checkpoints",
) -> List[Dict[str, float]]:
    """Perform k-fold cross-validation on a model.

    This function performs k-fold cross-validation on a given model, training
    and evaluating it on different splits of the dataset.

    Args:
        model_fn (Callable): Function to create the model.
        dataset (GTSRBDataset): The dataset to use for cross-validation.
        loss_fn (Callable): The loss function.
        optimizer_fn (Callable): Function to create the optimizer.
        num_epochs (int, optional): The number of epochs per fold. Defaults to
                                    10.
        batch_size (int, optional): The batch size. Defaults to 32.
        k (int, optional): The number of folds. Defaults to 5.
        seed (int, optional): The random seed for reproducibility. Defaults to
                              42.
        log_dir (str, optional): The directory for logs. Defaults to "logs".
        checkpoint_root (str, optional): The root directory for checkpoints.
                                         Defaults to "checkpoints".

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing metrics for
                                each fold.
    """
    splits = k_fold_splits(dataset, k, seed)

    results = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Directories.
        fold_ckpt_dir = os.path.join(checkpoint_root, f"fold_{fold}")
        fold_log_dir = os.path.join(log_dir, f"fold_{fold}")

        # Logger for this fold.
        logger = Logger(log_dir=fold_log_dir)
        logger.log_debug(f"Starting fold {fold + 1}/{k}.")

        # Prepare datasets.
        train_dataset = GTSRBDataset(
            indices=train_idx.tolist(),
            split="train",
            transforms=train_transforms,
            labels="filtered_labels_encoded.csv"
        )
        val_dataset = GTSRBDataset(
            indices=val_idx.tolist(),
            split="val",
            transforms=val_transforms,
            labels="filtered_labels_encoded.csv"
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)

        # Initialize model and optimizer per fold.
        fold_model = model_fn()

        param_list = []
        for layer in fold_model.layers:
            for name, param in layer.params().items():
                param_list.append((layer, name, param))

        optimizer = optimizer_fn(param_list)

        # Train.
        train_losses, val_losses, train_accs, val_accs = train(
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimiser=optimizer,
            num_epochs=num_epochs,
            checkpoint_dir=fold_ckpt_dir,
            log_dir=fold_log_dir,
        )
        logger.log_debug(f"Finished training fold {fold + 1}/{k}")

        # Metrics.
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

        # Append results.
        results.append(
            {
                "fold": fold + 1,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "train_acc": final_train_acc,
                "val_acc": final_val_acc,
            }
        )

        # Compute running averages.
        avg_train_acc = np.mean([r["train_acc"] for r in results])
        avg_val_acc = np.mean([r["val_acc"] for r in results])
        avg_train_loss = np.mean([r["train_loss"] for r in results])
        avg_val_loss = np.mean([r["val_loss"] for r in results])

    # Summary logger.
    summary_logger = Logger(log_dir=log_dir)

    summary_logger.log_debug("Cross-validation complete.")
    summary_logger.log_debug(
        f"Average Train Loss: {avg_train_loss:.4f}, "
        f"Average Val Loss: {avg_val_loss:.4f}, "
        f"Average Train Acc: {avg_train_acc:.4f}, "
        f"Average Val Acc: {avg_val_acc:.4f}"
    )

    return results
