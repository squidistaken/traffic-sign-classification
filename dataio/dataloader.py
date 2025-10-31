import numpy as np
from typing import Iterator, Callable, List, Optional, Tuple
from .gtsrb_dataset import GTSRBDataset


class DataLoader:
    """DataLoader for GTSRB dataset."""

    def __init__(
        self,
        dataset: "GTSRBDataset",
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 32,
        collate_fn: Optional[Callable] = None,
        prefetch_batches: int = 0,
    ) -> None:
        """Initialize the DataLoader.

        Args:
            dataset (GTSRBDataset): The Dataset Class instance.
            batch_size (int, optional): The batch size. Defaults to 128.
            shuffle (bool, optional): The flag to shuffle the dataset.
                                      Defaults to True.
            drop_last (bool, optional): The flag to drop the last incomplete
                                        batch. Defaults to False.
            seed (int, optional): The random seed for shuffling. Defaults to
                                  32.
            collate_fn (Optional[Callable], optional): The function to collate
                                                       samples into a batch.
                                                       Defaults to None.
            prefetch_batches (int, optional): The number of batches to
                                              prefetch. Defaults to 0.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        self.dataset = dataset

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self.rng = np.random.default_rng(seed=seed)

        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.prefetch_batches = prefetch_batches

    def default_collate(
        self, batch: List[Tuple[np.ndarray, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Defautly collate a batch of samples into a batch.

        Args:
            batch (List[Tuple[np.ndarray, int]]): The list of samples.

        Returns:
            tuple[np.ndarray, np.ndarray]: The batch of images and labels.
        """
        xs, ys = zip(*batch)
        # Example: (B, C, H, W).
        xs = np.stack(xs)
        # Example: (B,).
        ys = np.array(ys)

        return xs, ys

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over the dataset in batches, as an iterator. Each `batch` is
        a tuple (images, labels) where images are in NCHW format. Each batch
        has size `batch_size`, except possibly the last one (unless `drop_last`
        is True).

        Raises:
            ValueError: If the dataset is empty.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: The batches of images and
                                                     labels iterated.
        """
        n = len(self.dataset)

        if n == 0:
            raise ValueError("Dataset is empty")

        indices = np.arange(n)

        # Shuffle at the start of each epoch (not per batch).
        if self.shuffle:
            self.rng.shuffle(indices)

        # Iterate over the dataset in chunks of batch_size.
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size

            if end > n:
                if self.drop_last:
                    break
                else:
                    # Adjust the end to avoid index out of range.
                    end = n

            batch_indices = indices[start:end]
            batch = [self.dataset[i] for i in batch_indices]

            if self.collate_fn:
                batch = self.collate_fn(batch)
            else:
                batch = self.default_collate(batch)

            yield batch
