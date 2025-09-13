import os
from tokenizers import Tokenizer

# Define paths
base_folder = os.path.abspath("..")
DATA_PATH = f"{base_folder}/tokenization/wikitext-103-train.txt"
TOKENIZER_PATH = f"{base_folder}/tokenization/custom_tokenizer.json"

# Load the tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Load the dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Tokenize the dataset and count tokens
total_tokens = 0
for line in lines:
    encoded = tokenizer.encode(line.strip())
    total_tokens += len(encoded.tokens)

print(f"Total number of tokens in the dataset: {total_tokens}")