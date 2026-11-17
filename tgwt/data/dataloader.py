"""
Dataloader helpers for TGWT training and evaluation.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from .wikitext import WikiTextDataset, prepare_wikitext103


def get_dataloaders(
    seq_len: int = 1024,
    batch_size: int = 4,
    cache_dir: str = "./data_cache",
    max_train_samples: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation DataLoaders.
    """
    train_ds, val_ds = prepare_wikitext103(
        cache_dir=cache_dir,
        seq_len=seq_len,
        max_train_samples=max_train_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
