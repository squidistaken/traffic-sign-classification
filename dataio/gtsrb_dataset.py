from typing import Optional, Callable, Literal, Tuple, List
import numpy as np
import os


class GTSRBDataset:
    def __init__(
        self,
        root: str = "data/gtsrb/",
        x_dir: str = "images/",
        labels: str = "labels.csv",
        indices: List[int] = [],
        split: Literal["train", "val", "test"] = "train",
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.x_dir = x_dir
        self.labels = labels
        self.indices = indices
        self.split = split
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.indices)

    def load_ppm_image(self, filepath: str) -> np.ndarray:
        """
        Copilot-generated PPM P6 image loader.
        """
        with open(filepath, "rb") as f:
            header = f.readline()
            if header.strip() != b"P6":
                raise ValueError("Not a valid PPM P6 file")
            # Skip comments
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
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        actual_index = self.indices[index]
        with open(os.path.join(self.root, self.labels), "r") as f:
            lines = f.readlines()[1:]  # Skip header
            line = lines[actual_index].strip().split(";")
            filename, label = line[0], int(line[7])
            image = self.load_ppm_image(os.path.join(self.root, self.x_dir, filename))
            if self.transforms:
                image = self.transforms(image)
            return image, label


def tests() -> None:
    dataset = GTSRBDataset(
        indices=list(range(10)),
        transforms=lambda x: x / 255.0,
    )
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(f"Image shape: {img.shape}, Label: {label}")


if __name__ == "__main__":
    tests()
