import sys
import os
import torch

# Importation des modules gpt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpt.model import GPT, GPTConfig


def analyze_model_parameters(vocab_size=1000, block_size=256, n_layer=6, n_head=6, n_embd=384):
    """
    Calcule et affiche le détail du nombre de paramètres d'une configuration GPT.
    """
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    model = GPT(config)

    print("=" * 70)
    print(f" ANALYSE DU MODÈLE MINI-GPT (~10M Paramètres)")
    print(f"   Vocab Size: {vocab_size} | Context: {block_size} | Layers: {n_layer} | Heads: {n_head} | Emb Dim: {n_embd}")
    print("=" * 70)

    # 1. Détail par grand composant
    print(f"{'Composant / Couche':<35} | {'Nombre de Paramètres':<20} | {'% du Total':<10}")
    print("-" * 70)

    # Embeddings
    wte_params = sum(p.numel() for p in model.transformer.wte.parameters())
    wpe_params = sum(p.numel() for p in model.transformer.wpe.parameters())
    print(f"{'wte (Token Embeddings)':<35} | {wte_params:>20,}")
    print(f"{'wpe (Position Embeddings)':<35} | {wpe_params:>20,}")

    # Blocs Transformer (Toutes les couches)
    total_blocks_params = 0
    for idx, block in enumerate(model.transformer.h):
        block_params = sum(p.numel() for p in block.parameters())
        total_blocks_params += block_params
        if idx == 0:
            print(f"{'  ↳ Block 0 (Attention + MLP)':<35} | {block_params:>20,}")

    print(f"{f'h (Total {n_layer} Blocs Transformer)':<35} | {total_blocks_params:>20,}")

    # LayerNorm Finale
    ln_f_params = sum(p.numel() for p in model.transformer.ln_f.parameters())
    print(f"{'ln_f (LayerNorm finale)':<35} | {ln_f_params:>20,}")

    # Tête de sortie (LM Head)
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"{'lm_head (LM Head - Tied wte)':<35} | {lm_head_params:>20,} (Partagés)")

    print("-" * 70)

    # 2. Comptage total réels (Gestion du Weight Tying)
    # On utilise les identifiants mémoire (data_ptr) pour exclure les doublons
    unique_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            unique_params[param.data_ptr()] = param.numel()

    total_unique_params = sum(unique_params.values())

    print(f"{'TOTAL PARAMÈTRES UNIQUES':<35} | {total_unique_params:>20,}")
    print("=" * 70)

    print(f"\n Note pour la soutenance :")
    print(f"   • Les {n_layer} blocs Transformer représentent {(total_blocks_params/total_unique_params)*100:.1f}% de la taille du modèle.")
    print(f"   • Avec un dictionnaire BPE de {vocab_size} tokens, le modèle fait exactement {total_unique_params / 1e6:.2f}M de paramètres.")


if __name__ == "__main__":
    # Test avec la taille de vocabulaire BPE typique (ex: 1000 tokens)
    analyze_model_parameters(vocab_size=1000)