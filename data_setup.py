import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 16

def create_dataloaders(
    train_dir: str, 
    val_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS):

    """Creates training and validation DataLoaders.

  Takes in a training directory and validation directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to validation directory.
    transform: torchvision transforms to perform on training and validation data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, val_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, val_dataloader, class_names = create_dataloaders(
                                                                        train_dir=path/to/train_dir,
                                                                        val_dir=path/to/val_dir,
                                                                        transform=some_transform,
                                                                        batch_size=32,
                                                                        num_workers=4)
  """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, class_names
