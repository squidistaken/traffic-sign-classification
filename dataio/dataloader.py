# mini-batch iterator, shuffling
# responsibilities: return NCHW batches with labels

import numpy as np
from typing import Iterator, Callable, List, Optional, Tuple
from .gtsrb_dataset import GTSRBDataset


class DataLoader:
    def __init__(
        self,
        dataset: GTSRBDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 32,
        collate_fn: Optional[Callable] = None,
        prefetch_batches: int = 0,  # 0 = no prefetch
    ) -> None:
        """
        DataLoader for GTSRB dataset. Yields batches of images and labels of size batch_size.
        Instances can be treated as iterators, like `for batch in dataloader`.
        """
        self.dataset = dataset
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
        xs, ys = zip(*batch)
        xs = np.stack(xs)  # (B, C, H, W)
        ys = np.array(ys)  # (B,)
        return xs, ys

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Allows calling the class instance as an iterator, like `for batch in dataloader`.
        In this case each `batch` is a tuple (images, labels) where images are in NCHW format.
        Each batch has size `batch_size`, except possibly the last one (unless `drop_last` is True).
        """
        n = len(self.dataset)
        indices = np.arange(n)

        # shuffle at the start of each epoch (not for each batch)
        if self.shuffle:
            self.rng.shuffle(indices)

        # iterate over dataset in chunks of batch_size
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_indices = indices[start:end]
            batch = [self.dataset[i] for i in batch_indices]
            if self.collate_fn:
                batch = self.collate_fn(batch)
            else:
                batch = self.default_collate(batch)
            yield batch
