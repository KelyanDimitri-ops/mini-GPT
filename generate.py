import os
import sys
import torch

# Ajout de la racine au sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gpt.model import GPT, GPTConfig
from tokenizer.tokenizer import Tokenizer


def load_model_and_tokenizer(ckpt_path: str = "checkpoints/ckpt_latest.pt",
                             tokenizer_path: str = "tokenizer/vocab_1000.json"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint introuvable à '{ckpt_path}'. Attend la fin de l'itération 200 de train.py.")
        sys.exit(1)

    # Chargement du tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)

    # Chargement du checkpoint PyTorch
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(
        f"✅ Modèle chargé depuis '{ckpt_path}' (Itération {checkpoint.get('iter', 'N/A')}, Val BPB: {checkpoint.get('val_bpb', 0.0):.4f})")
    return model, tokenizer, device


def generate_prompt(prompt: str, max_tokens: int = 150, temperature: float = 0.8, top_k: int = 40):
    model, tokenizer, device = load_model_and_tokenizer()

    encoded = tokenizer.encode(prompt)
    context = torch.tensor([encoded], dtype=torch.long, device=device)

    print(f"\n🎭 **Prompt initial** : {prompt}")
    print("-" * 50)

    with torch.no_grad():
        generated_ids = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        text = tokenizer.decode(generated_ids[0].tolist())
        print(text)
    print("-" * 50)


if __name__ == "__main__":
    prompt_text = "HARPAGON :"
    generate_prompt(prompt_text, max_tokens=200, temperature=0.7, top_k=35)