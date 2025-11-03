from typing import Optional, Callable, Literal, Tuple, List
import numpy as np
import os
import csv
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


class GTSRBDataset:
    """German Traffic Sign Recognition Benchmark Dataset Class.

    This class provides functionality to load and manage the GTSRB dataset,
    including handling images and labels, applying transformations, and
    supporting caching.
    """

    def __init__(
        self,
        root: str = "./data/gtsrb/",
        x_dir: str = "images/",
        labels: str = "labels.csv",
        indices: List[int] = [],
        split: Literal["train", "val", "test"] | None = None,
        transforms: Callable | None = None,
        for_torch: bool = False,
        cache: bool = False,
        num_workers: int = 4,
    ) -> None:
        """Initialise the GTSRB dataset.

        Args:
            root (str, optional): The root directory of the dataset.
                                  Defaults to "data/gtsrb/".
            x_dir (str, optional): The directory containing images.
                                   Defaults to "images/".
            labels (str, optional): The CSV file containing labels.
                                    Defaults to "labels.csv".
            indices (List[int], optional): The list of indices to include in
                                           the dataset. Defaults to [].
            split (Literal["train","val","test"], optional): The dataset split.
                                                             Defaults to
                                                             None.
            transforms (Optional[Callable], optional): The transformations to
                                                       apply to the images.
                                                       Defaults to None.
            for_torch (bool, optional): Whether the dataset is used for
                                        training with PyTorch. Defaults to
                                        False.
            cache (bool, optional): Whether to cache loaded images. Defaults to
                                    False.
            num_workers (int, optional): Number of workers for parallel
                                         loading. Defaults to 4.
        """
        self.root = root
        self.x_dir = x_dir
        self.labels = labels
        self.indices = indices
        self.split = split
        self.transforms = transforms
        self.cache = cache
        self.num_workers = num_workers
        self.for_torch = for_torch
        self.labels_data = self._load_labels()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.image_cache = {} if cache else None

    def _load_labels(self) -> List[Tuple[str, int]]:
        """Load the labels from the labels file.

        Returns:
            List[Tuple[str, int]]: A list of tuples containing filename and
                                   label.
        """
        with open(os.path.join(self.root, self.labels), "r") as f:
            reader = csv.reader(f, delimiter=";")

            # Skip header
            next(reader)

            return [(row[0], int(row[7])) for row in reader]

    def __len__(self) -> int:
        """Set the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self.indices is not None and len(self.indices) > 0:
            return len(self.indices)

        return len(self.labels_data)

    def load_ppm_image(self, filepath: str) -> np.ndarray:
        """Load a PPM P6 image from the given filepath.

        Args:
            filepath (str): The path to the PPM image file. It is expected to
                            be in PPM P6 format and 8-bit per channel.

        Raises:
            ValueError: If the file is not a valid PPM P6 file.
            ValueError: If the PPM file is not 8-bit.

        Returns:
            np.ndarray: The loaded image as a NumPy array.
        """
        with open(filepath, "rb") as f:
            header = f.readline()

            if header.strip() != b"P6":
                raise ValueError("Not a valid PPM P6 file")

            while True:
                line = f.readline()

                if line.startswith(b"#"):
                    continue
                else:
                    break

            dimensions = line.strip().split()
            width, height = int(dimensions[0]), int(dimensions[1])
            maxval = int(f.readline().strip())

            if maxval > 255:
                raise ValueError("Only 8-bit PPM files are supported")

            img_data = f.read()
            image = np.frombuffer(img_data, dtype=np.uint8)
            image = image.reshape((height, width, 3))

            return image

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Raises:
            IndexError: If the index is out of range.

        Returns:
            Tuple[np.ndarray, int]: The image and its corresponding label, as a
                                    tuple.
        """
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        actual_index = self.indices[index]
        filename, label = self.labels_data[actual_index]
        filepath = os.path.join(self.root, self.x_dir, filename)

        # Check the cache first.
        if self.cache and filepath in self.image_cache:
            image = self.image_cache[filepath]
        else:
            image = self.load_ppm_image(filepath)
            if self.cache:
                self.image_cache[filepath] = image

        # Convert to PIL Image if we are using PyTorch.
        if self.for_torch:
            image = Image.fromarray(image, mode="RGB")
        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __del__(self):
        """Clean up the executor when the dataset is deleted."""
        self.executor.shutdown(wait=False)


def tests() -> None:
    """Run tests for GTSRBDataset class."""

    # Test with a small subset of indices
    dataset = GTSRBDataset(
        indices=list(range(10)),
        transforms=lambda x: x / 255.0,
    )

    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        img, label = dataset[i]
        print(f"Image shape: {img.shape}, Label: {label}")

    # Test with a single index
    dataset_single = GTSRBDataset(
        indices=[0],
        transforms=lambda x: x / 255.0,
    )

    print(f"Single index dataset length: {len(dataset_single)}")

    img, label = dataset_single[0]

    print(f"Single image shape: {img.shape}, Label: {label}")

    # Test without transforms
    dataset_no_transform = GTSRBDataset(
        indices=list(range(5)),
        transforms=None,
    )

    print(f"Dataset without transforms length: {len(dataset_no_transform)}")

    for i in range(len(dataset_no_transform)):
        img, label = dataset_no_transform[i]

        print(f"Image shape (no transform): {img.shape}, Label: {label}")


if __name__ == "__main__":
    tests()
