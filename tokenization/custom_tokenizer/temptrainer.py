import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from .config import DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS
from .data_processing import load_dataset, clean_and_save_dataset


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

    # Ensure cleaned data is available; if not, run the cleaning pipeline
    if not os.path.exists(DATA_PATH):
        print(f"Cleaned data file not found at {DATA_PATH}. Running cleaning step...")
        dataset = load_dataset()
        clean_and_save_dataset(dataset)
    else:
        print(f"Found existing cleaned data at {DATA_PATH}")

    # Train and save the tokenizer
    print(f"Training tokenizer on cleaned data at {DATA_PATH}")
    tokenizer.train([DATA_PATH], trainer)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer trained and saved at: {TOKENIZER_PATH}")


def load_tokenizer():
    """Loads the tokenizer from file if it exists; otherwise, trains (and cleans) and saves a new one."""
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}. Training a new one...")
        try:
            train_and_save_tokenizer()
        except Exception as e:
            raise RuntimeError(
                f"Failed to train and save tokenizer at {TOKENIZER_PATH}: {e}"
            )
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file still not found at {TOKENIZER_PATH} after training."
        )

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Tokenizer loaded from {TOKENIZER_PATH}")
    return tokenizer