import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#####################
#   DATA UTILS      #
#####################
class MNIST_partial(Dataset):
    def __init__(self, data="./data", transform=None, split="train"):
        """
        Load the reduced MNIST dataset from CSV.
        :param data: Path to folder containing train.csv and val.csv.
        :param transform: Optional transform function.
        :param split: "train" or "val"
        """
        self.data_dir = data
        self.transform = transform
        if split == "train":
            filename = os.path.join(self.data_dir, "train.csv")
        elif split == "val":
            filename = os.path.join(self.data_dir, "val.csv")
        else:
            raise AttributeError("split must be 'train' or 'val'")
        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df["image"])

    def __getitem__(self, idx):
        img_str = self.df["image"].iloc[idx]
        label = int(self.df["label"].iloc[idx])
        img_list = re.split(r",", img_str)
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        img_float = [float(el) for el in img_list]
        # Reshape to (1, 28, 28)
        img_tensor = torch.tensor(img_float).reshape(1, 28, 28)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label

def get_dataloader(dataset, batch_size, shuffle):
    """Return a DataLoader for the given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#####################
# TRAINING UTILS    #
#####################
def plot_training_metrics(train_acc, val_acc, train_loss, val_loss):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs = list(range(1, len(train_acc) + 1))
    axes[0].plot(epochs, train_acc, label="Training Accuracy")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend()
    axes[1].plot(epochs, train_loss, label="Training Loss")
    axes[1].plot(epochs, val_loss, label="Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))





import torch.fft

def fourier_encode(image_tensor, n_features=100):
    """
    Computes Fourier features from an image tensor.
    
    Parameters:
      image_tensor: A tensor of shape [1, 28, 28] (or [28,28]).
      n_features  : Number of Fourier features to extract.
    
    Returns:
      A 1D tensor of Fourier features (absolute values of the FFT).
    """
    # Ensure the image is 2D.
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    fft_result = torch.fft.fft2(image_tensor)
    fft_abs = torch.abs(fft_result).flatten()
    # Select the first n_features; you might also choose to sort or use a different selection method.
    return fft_abs[:n_features]
