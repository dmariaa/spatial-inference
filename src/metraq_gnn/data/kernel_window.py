"""Adapt existing masked graph windows without reading target labels."""
import torch
from torch.utils.data import Dataset


def adapt_kernel_window(sample):
    result = sample.clone()
    channels = sample.context_mask.shape[-1]
    result.kernel_values = sample.x[:, -1, :channels].clone()
    result.kernel_available = sample.context_mask[:, -1, :].clone()
    if torch.any(result.kernel_available & sample.target_mask):
        raise ValueError("target AQ cannot be visible")
    if torch.any(result.kernel_available & sample.forbidden_context_mask[:, None]):
        raise ValueError("forbidden AQ cannot be visible")
    if not result.kernel_available.any(dim=0).all():
        raise ValueError("no current visible AQ sources for a sample/channel")
    return result


class KernelWindowDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return adapt_kernel_window(self.dataset[index])

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
