from typing import Tuple
import numpy as np
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        img = Image.fromarray(x)
        img = img.resize((self.size, self.size))
        return np.array(img)


class ToGrayscale:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        img = Image.fromarray(x)
        img = img.convert("L")  # convert to grayscale
        img = img.convert("RGB")  # convert back to RGB
        return np.array(img)


class ToTensor:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.transpose((2, 0, 1)).astype(np.float32) / 255.0  # HWC to CHW


class Normalize:
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std  # assuming x is in [0, 1] range


# todo: resize, center-crop/pad-to-square, normalize, augment
