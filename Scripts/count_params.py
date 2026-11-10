import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt.model import GPT, GPTConfig
from tokenizer.tokenizer import Tokenizer


def count_parameters():
    tokenizer_path = "tokenizer/vocab_1000.json"
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
        vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)
    else:
        vocab_size = 1001

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=360
    )
    model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for p in model.transformer.wte.parameters()) + \
                       sum(p.numel() for p in model.transformer.wpe.parameters())
    transformer_core_params = total_params - embedding_params

    print("\n📊 --- DÉCOMPTE DÉTAILLÉ DES PARAMÈTRES (10M Budget) ---")
    print(f"🔹 Vocabulaire BPE        : {vocab_size} tokens")
    print(f"🔹 Embeddings (wte + wpe) : {embedding_params:,} paramètres")
    print(f"🔹 Bloc Transformer (6L)  : {transformer_core_params:,} paramètres")
    print(f"🔹 Total des paramètres   : {total_params:,} (~{total_params/1e6:.2f}M)")

    if total_params <= 10_500_000:
        print("✅ Le modèle respecte la contrainte des 10 Millions de paramètres !")
    else:
        print("⚠️ Le modèle dépasse le budget autorisé de 10M de paramètres.")


if __name__ == "__main__":
    count_parameters()