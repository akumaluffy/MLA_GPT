import os
import sys
import torch
import math

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    print("Flash Attention is available!")
except ImportError:
    HAS_FLASH_ATTN = False
    print("Flash Attention is not available, falling back to standard attention")

base_folder = os.path.abspath("..")
print(f"Your base folder is: {base_folder}")
sys.path.append(base_folder)

from tokenization.custom_tokenizer.config import TOKENIZER_PATH
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

vocab_size = tokenizer.get_vocab_size()

from transformer_setup import ModelConfig, TransformerModel
config = ModelConfig()

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']
    print("Checkpoint loaded. Model configuration:")
    for key, val in config_dict.items():
        print(f"  {key}: {val}")

    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=config_dict['n_embd'],
        num_heads=config_dict['n_head'],
        num_layers=config_dict['n_layer'],
        max_seq_len=config_dict['block_size'],
        dropout_prob=config_dict['dropout'],
        use_gradient_checkpoint=config_dict.get('gradient_checkpointing', False),
        use_flash_attn=config_dict.get('use_flash_attn', False),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    return model, config_dict['block_size']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_choice = input("Which model would you like to inference (best_model): ")

    # Default to best_model
    if not model_choice:
        model_choice = "best_model"

    checkpoint_path = os.path.join("checkpoints", f"{model_choice}.pt")
    model, block_size = load_model(checkpoint_path, device)

    print("\nEnter your prompt below. Type 'exit' to quit.")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower().strip() == "exit":
            break
        
        prompt_ids = tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_tensor = model.generate(
                input_tensor, 
                max_new_tokens=300, 
                max_seq_len=block_size,
                temperature=1.0
            )

        generated_text = tokenizer.decode(generated_tensor[0].tolist())
        print("\nGenerated text:")
        print(generated_text)

if __name__ == "__main__":
    main()