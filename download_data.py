"""
Download and prepare WikiText-103 dataset.
Usage:
    python download_data.py --cache_dir ./data_cache --seq_len 1024
"""

import argparse
from tgwt.data.wikitext import prepare_wikitext103, get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Download and prepare WikiText-103")
    parser.add_argument("--cache_dir", type=str, default="./data_cache", help="Cache directory")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for chunking")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer model name")
    parser.add_argument("--max_train_samples", type=int, default=50000, help="Number of text lines to process (default 50,000 for fast Colab training)")
    parser.add_argument("--full", action="store_true", help="Download and tokenize the entire 103M-token corpus (takes ~15-20 mins)")
    args = parser.parse_args()

    max_samples = None if args.full else args.max_train_samples
    print(f"Preparing WikiText-103 (seq_len={args.seq_len}, max_lines={max_samples or 'ALL'})...")
    tokenizer = get_tokenizer(args.tokenizer)
    train_ds, val_ds = prepare_wikitext103(
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        max_train_samples=max_samples,
    )
    print(f"Dataset preparation complete! Train chunks: {len(train_ds):,}, Val chunks: {len(val_ds):,}")


if __name__ == "__main__":
    main()
