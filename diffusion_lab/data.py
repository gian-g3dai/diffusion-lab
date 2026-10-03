"""Dataset loaders. Images are scaled to [-1, 1], which is what the diffusion
process assumes (noise is N(0, 1), so data should live on a similar scale)."""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# name -> (constructor, channels, image_size)
_DATASETS = {
    "mnist": (datasets.MNIST, 1, 28),
    "fashion_mnist": (datasets.FashionMNIST, 1, 28),
    "cifar10": (datasets.CIFAR10, 3, 32),
}


def dataset_info(name: str):
    """Returns (channels, image_size) for a dataset name."""
    _, c, s = _DATASETS[name]
    return c, s


def get_dataloader(name: str, root: str, batch_size: int, num_workers: int = 2,
                   image_size: int | None = None) -> DataLoader:
    ctor, channels, native_size = _DATASETS[name]
    image_size = image_size or native_size
    tf = [transforms.Resize(image_size)] if image_size != native_size else []
    if channels == 3:
        tf.append(transforms.RandomHorizontalFlip())
    tf += [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
    ds = ctor(root=root, train=True, download=True, transform=transforms.Compose(tf))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
