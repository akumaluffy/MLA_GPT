import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from .config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS

def create_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    return tokenizer, trainer

def train_and_save_tokenizer():
    tokenizer, trainer = create_tokenizer()
    print(f"Data Path: {DATA_PATH}")

    # Always re-download the dataset
    print("Downloading WikiText-103 dataset...")
    try:
        from datasets import load_dataset
        ds_train = load_dataset("wikitext", "wikitext-103-v1", split="train")
    except ImportError:
        raise ImportError("Please install the datasets library: pip install datasets")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Write the dataset to file
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for example in ds_train:
            f.write(example["text"].rstrip() + "\n")
    print(f"Wrote {len(ds_train)} documents to {DATA_PATH}")

    # Train and save the tokenizer
    tokenizer.train([DATA_PATH], trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer trained and saved at: {TOKENIZER_PATH}")

def load_tokenizer():
    """Loads the tokenizer from file if it exists; otherwise, trains and saves a new one."""
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}. Training a new one...")
        train_and_save_tokenizer()

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Tokenizer loaded from {TOKENIZER_PATH}")
    return tokenizer  