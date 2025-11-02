from typing import Tuple
import numpy as np
from PIL import Image


class ToCompose:
    """The ToCompose Transform Class to combine multiple transforms."""

    def __init__(self, transforms: Tuple) -> None:
        """Initialize the Compose transform.

        Args:
            transforms (Tuple): The tuple of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the composed transforms on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The transformed image array.
        """
        for t in self.transforms:
            x = t(x)

        return x


class ToResize:
    """The ToResize Transform Class to resize images."""

    def __init__(self, size: int) -> None:
        """Initialize the Resize transform.

        Args:
            size (int): The target size to resize the image to.
        """
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the resize transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The resized image array.
        """
        img = Image.fromarray(x)
        img = img.resize((self.size, self.size))

        return np.array(img)


class ToCenterCrop:
    """The ToCenterCrop Transform Class to crop images to a square."""

    def __init__(self, size: int) -> None:
        """Initialize the CenterCrop transform.

        Args:
            size (int): The target size to crop the image to.
        """
        self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the CenterCrop transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The cropped image array.
        """
        height, width = x.shape[:2]
        # Calculate the starting points for cropping.
        start_x = (width - self.size) // 2
        start_y = (height - self.size) // 2
        # Crop the image.
        cropped_img = x[start_y : start_y + self.size, start_x : start_x + self.size]

        return cropped_img


class ToGrayscale:
    """The ToGrayscale Transform Class to convert images to grayscale."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the ToGrayscale transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The grayscale image array.
        """
        img = Image.fromarray(x)
        # Convert to grayscale.
        img = img.convert("L")
        # Convert back to RGB, as some models expect 3 channels.
        # It remains grayscale visually.
        img = img.convert("RGB")

        return np.array(img)


class ToTensor:
    """The ToTensor Transform Class to convert images to tensors."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the ToTensor transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The image array as a tensor.
        """
        # HWC to CHW
        return x.transpose((2, 0, 1)).astype(np.float32) / 255.0


class ToNormalize:
    """The ToNormalize Transform Class to normalize images."""

    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        """Initialize the Normalize transform.

        Args:
            mean (Tuple[float, float, float]): The mean for each channel.
            std (Tuple[float, float, float]): The standard deviation
                                              for each channel.
        """
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the Normalize transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The normalized image array.
        """
        # We assume x is in [0, 1] range
        return (x - self.mean) / self.std


class ToRotate:
    """The ToRotate Transform Class to rotate images randomly up to a certain angle."""

    def __init__(self, angle: float) -> None:
        """Initialize the Rotate transform.

        Args:
            angle (float): The maximum angle to rotate the image by.
        """
        self.angle = angle

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the Rotate transform on the input with a random angle.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The rotated image array.
        """
        angle = np.random.uniform(-self.angle, self.angle)
        img = Image.fromarray(x)
        img = img.rotate(angle)

        return np.array(img)


class ToNoise:
    """The ToNoise Transform Class to add Gaussian noise to images."""

    def __init__(self, mean=0, std=0.1):
        """
        Initialize the Noise transform.

        Args:
            mean (float): The mean of the Gaussian noise.
            std (float): The standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Call the Noise transform on the input.

        Args:
            x (np.ndarray): The input image array.

        Returns:
            np.ndarray: The noisy image array.
        """
        # Generate Gaussian noise with the same shape as the input image
        noise = np.random.normal(self.mean, self.std, x.shape)
        # Add the noise to the image
        noisy_img = x + noise
        # Clip the values to ensure they are within the valid range
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img
