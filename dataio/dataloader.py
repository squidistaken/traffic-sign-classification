import numpy as np
from typing import Iterator, Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from .gtsrb_dataset import GTSRBDataset


class DataLoader:
    """DataLoader for GTSRB dataset with prefetching and parallel data loading."""

    def __init__(
        self,
        dataset: GTSRBDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 32,
        collate_fn: Optional[Callable] = None,
        prefetch_batches: int = 2,
        num_workers: int = 4,  # Number of workers for parallel data loading
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
                                              prefetch. Defaults to 2.
            num_workers (int, optional): The number of workers for parallel
                                         data loading. Defaults to 4.
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
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_buffer = []

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

    def _load_batch(self, batch_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Load a batch of data."""
        batch = [self.dataset[i] for i in batch_indices]
        if self.collate_fn:
            batch = self.collate_fn(batch)
        else:
            batch = self.default_collate(batch)
        return batch

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
        # Initialize prefetch buffer
        self.prefetch_buffer = []
        # Prefetch initial batches
        for start in range(
            0, min(n, (self.prefetch_batches + 1) * self.batch_size), self.batch_size
        ):
            end = start + self.batch_size
            if end > n:
                if self.drop_last:
                    continue
                else:
                    end = n
            batch_indices = indices[start:end]
            future = self.executor.submit(self._load_batch, batch_indices)
            self.prefetch_buffer.append(future)
        # Iterate over the dataset
        for start in range(0, n, self.batch_size):
            if not self.prefetch_buffer:
                break
            # Wait for the first prefetched batch
            batch = self.prefetch_buffer.pop(0).result()
            yield batch
            # Prefetch the next batch if there are more
            end = start + (self.prefetch_batches + 1) * self.batch_size
            if end <= n:
                next_start = start + self.prefetch_batches * self.batch_size
                next_end = next_start + self.batch_size
                if next_end > n:
                    if self.drop_last:
                        continue
                    else:
                        next_end = n
                next_batch_indices = indices[next_start:next_end]
                future = self.executor.submit(self._load_batch, next_batch_indices)
                self.prefetch_buffer.append(future)

    def __del__(self):
        """Clean up the executor when the DataLoader is deleted."""
        self.executor.shutdown(wait=False)
