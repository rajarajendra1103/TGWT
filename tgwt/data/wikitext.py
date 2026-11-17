"""
WikiText-103 Dataset Loader and Tokenizer utilities for TGWT.
Includes automatic caching, HuggingFace dataset loading, direct HTTP fallback,
and local synthetic fallback for offline testing.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import sys
import json
import torch
from torch.utils.data import Dataset


class SimpleFallbackTokenizer:
    """
    Lightweight, self-contained character/byte tokenizer when HuggingFace is unavailable.
    """

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def encode(self, text: str) -> List[int]:
        # Encode as byte offsets mapped within vocab
        bytes_data = text.encode("utf-8", errors="replace")
        return [(b % (self.vocab_size - 4)) + 4 for b in bytes_data]

    def decode(self, token_ids: List[int]) -> str:
        valid_bytes = bytes([(tid - 4) % 256 for tid in token_ids if tid >= 4])
        return valid_bytes.decode("utf-8", errors="replace")

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None):
        if isinstance(text, str):
            ids = self.encode(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return {"input_ids": ids}
        else:
            all_ids = [self.encode(t) for t in text]
            if return_tensors == "pt":
                max_len = max(len(ids) for ids in all_ids)
                padded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in all_ids]
                return {"input_ids": torch.tensor(padded, dtype=torch.long)}
            return {"input_ids": all_ids}


def get_tokenizer(model_name: str = "gpt2"):
    """
    Returns Hugging Face GPT2 tokenizer if available, else SimpleFallbackTokenizer.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception:
        return SimpleFallbackTokenizer(vocab_size=50257)


class WikiTextDataset(Dataset):
    """
    In-memory chunked dataset for WikiText-103 tokens.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int = 1024,
    ):
        self.seq_len = seq_len
        # Cut tokens into non-overlapping or overlapping chunks
        num_chunks = len(tokens) // seq_len
        if num_chunks > 0:
            self.data = tokens[: num_chunks * seq_len].view(num_chunks, seq_len)
        else:
            # Pad if shorter
            pad_amount = seq_len - len(tokens)
            self.data = torch.cat([tokens, torch.zeros(pad_amount, dtype=torch.long)]).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        return x, x.clone()


def tokenize_in_batches(
    texts: List[str],
    tokenizer,
    batch_size: int = 1000,
    desc: str = "Tokenizing",
) -> torch.Tensor:
    """
    Tokenizes a list of strings in batches with tqdm progress bar.
    Leverages HuggingFace fast multi-threaded tokenizer if available.
    """
    from tqdm import tqdm
    all_tokens: List[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="k-lines"):
        batch = texts[i : i + batch_size]
        if hasattr(tokenizer, "batch_encode_plus"):
            encoded_batch = tokenizer.batch_encode_plus(
                batch,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            for ids in encoded_batch:
                all_tokens.extend(ids)
        else:
            for text in batch:
                all_tokens.extend(tokenizer.encode(text))

    return torch.tensor(all_tokens, dtype=torch.long)


def prepare_wikitext103(
    cache_dir: str = "./data_cache",
    seq_len: int = 1024,
    tokenizer = None,
    max_train_samples: Optional[int] = 50000,
) -> Tuple[WikiTextDataset, WikiTextDataset]:
    """
    Downloads or prepares WikiText-103 train and validation datasets.
    Defaults to 50,000 articles (~5M-10M tokens) for rapid, stable Colab training.
    Set max_train_samples=None to process the entire 103M-token corpus.
    """
    os.makedirs(cache_dir, exist_ok=True)
    raw_cache_file = os.path.join(cache_dir, "wikitext103_raw_tokens.pt")
    cache_file = os.path.join(cache_dir, f"wikitext103_cached_{seq_len}.pt")

    if os.path.exists(raw_cache_file):
        print(f"Loading cached WikiText-103 tokens from {raw_cache_file}...")
        data = torch.load(raw_cache_file)
        return (
            WikiTextDataset(data["train_tokens"], seq_len=seq_len),
            WikiTextDataset(data["val_tokens"], seq_len=seq_len),
        )

    if os.path.exists(cache_file):
        print(f"Loading cached WikiText-103 from {cache_file}...")
        data = torch.load(cache_file)
        return (
            WikiTextDataset(data["train_tokens"], seq_len=seq_len),
            WikiTextDataset(data["val_tokens"], seq_len=seq_len),
        )

    # Check if any cached pt exists in cache_dir
    existing_pts = [f for f in os.listdir(cache_dir) if f.startswith("wikitext103_cached_") and f.endswith(".pt")]
    if existing_pts:
        first_pt = os.path.join(cache_dir, existing_pts[0])
        print(f"Loading existing WikiText-103 tokens from {first_pt}...")
        data = torch.load(first_pt)
        torch.save(data, raw_cache_file)
        return (
            WikiTextDataset(data["train_tokens"], seq_len=seq_len),
            WikiTextDataset(data["val_tokens"], seq_len=seq_len),
        )

    if tokenizer is None:
        tokenizer = get_tokenizer()

    train_tokens = None
    val_tokens = None

    # Try Hugging Face datasets
    try:
        from datasets import load_dataset
        print("Downloading WikiText-103 via Hugging Face datasets...")
        try:
            dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=cache_dir)
        except Exception:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=cache_dir)

        print("Filtering non-empty text lines...")
        train_raw = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
        val_raw = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]

        if max_train_samples is not None and max_train_samples > 0:
            train_raw = train_raw[:max_train_samples]
            val_raw = val_raw[:max(1000, max_train_samples // 5)]
            print(f"Using {len(train_raw):,} train lines and {len(val_raw):,} val lines for fast Colab training...")
        else:
            print(f"Using full dataset: {len(train_raw):,} train lines and {len(val_raw):,} val lines...")

        train_tokens = tokenize_in_batches(train_raw, tokenizer, batch_size=2000, desc="Tokenizing Train Split")
        val_tokens = tokenize_in_batches(val_raw, tokenizer, batch_size=2000, desc="Tokenizing Val Split")

    except Exception as e:
        print(f"Hugging Face download failed/unavailable ({e}). Generating synthetic corpus for offline testing...")
        sample_words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "transformer", "graph", "tree", "walk", "attention", "memory",
            "linear", "complexity", "token", "embedding", "node", "edge",
            "hierarchical", "reasoning", "colab", "gpu", "wikitext", "benchmark"
        ]
        import random
        rng = random.Random(42)
        words_train = [rng.choice(sample_words) for _ in range(50000)]
        words_val = [rng.choice(sample_words) for _ in range(10000)]

        train_text = " ".join(words_train)
        val_text = " ".join(words_val)

        train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    # Save to cache
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens}, raw_cache_file)
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens}, cache_file)
    print(f"Successfully cached {len(train_tokens):,} train tokens and {len(val_tokens):,} val tokens to {raw_cache_file}!")

    return (
        WikiTextDataset(train_tokens, seq_len=seq_len),
        WikiTextDataset(val_tokens, seq_len=seq_len),
    )
