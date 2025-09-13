#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast

EVAL_BATCH_SIZE = 128
NUM_WORKERS     = 4

# Hard-coded project root
PROJECT_ROOT = "/workspace/GPT"
print(f"Project root: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)

from tokenization.custom_tokenizer.trainer import load_tokenizer
from models.transformer_setup import ModelConfig, TransformerModel

def load_model(checkpoint_path, device, tokenizer):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print("Checkpoint loaded. Model configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    model = TransformerModel(
        vocab_size              = tokenizer.get_vocab_size(),
        embed_dim               = cfg["n_embd"],
        num_heads               = cfg["n_head"],
        num_layers              = cfg["n_layer"],
        max_seq_len             = cfg["block_size"],
        dropout_prob            = cfg["dropout"],
        use_gradient_checkpoint = cfg.get("gradient_checkpointing", False),
        use_flash_attn          = True   # force flash-attn on every head
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model.eval()

def calculate_perplexity(model, dataloader, device):
    total_tokens      = 0
    total_loss_tensor = torch.tensor(0.0, device=device)
    # inference_mode + fp16 autocast
    with torch.inference_mode(), autocast("cuda", torch.float16):
        for x, y in tqdm(dataloader, desc="Calculating perplexity"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y)
            bsz, seq = x.shape
            total_loss_tensor += loss * (bsz * seq)
            total_tokens      += bsz * seq

    avg_loss   = (total_loss_tensor / total_tokens).item()
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss

def main():
    tokenizer = load_tokenizer()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # always use best_model.pt
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "best_model.pt")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, device, tokenizer)

    config = ModelConfig()

    # load & tokenize test data
    from data import get_wikitext_data
    texts = get_wikitext_data()["test"]["text"]
    tokens = []
    for t in tqdm(texts, desc="Tokenizing test data"):
        tokens.extend(tokenizer.encode(t).ids)

    # build dataloader
    from torch.utils.data import Dataset, DataLoader
    class TokenizedDataset(Dataset):
        def __init__(self, toks, block_size):
            self.toks       = toks
            self.block_size = block_size
        def __len__(self):
            return len(self.toks) - self.block_size
        def __getitem__(self, i):
            x = torch.tensor(self.toks[i : i + self.block_size], dtype=torch.long)
            y = torch.tensor(self.toks[i + 1 : i + 1 + self.block_size], dtype=torch.long)
            return x, y

    ds = TokenizedDataset(tokens, config.block_size)
    loader = DataLoader(
        ds,
        batch_size   = EVAL_BATCH_SIZE,
        shuffle      = False,
        num_workers  = NUM_WORKERS,
        pin_memory   = True
    )

    perp, avg = calculate_perplexity(model, loader, device)
    print("\nEvaluation Results:")
    print(f"  Average Loss: {avg:.4f}")
    print(f"  Perplexity:   {perp:.2f}")

if __name__ == "__main__":
    main()