import os
import sys

# Import direct depuis le même dossier (tokenizer/)
from train_gpt2_bpe import train_gpt2_bpe
from tokenizer import Tokenizer


def build_vocabulary():
    # Détection automatique du fichier source (data/ ou data_pipeline/)
    candidates = [
        "data/moliere_train.txt",
        "data_pipeline/moliere_train.txt",
        "moliere_train.txt"
    ]
    train_path = next((p for p in candidates if os.path.exists(p)), "")
    output_path = "tokenizer/vocab_1000.json"
    vocab_size = 1000

    if not train_path:
        print("⚠️ Fichier moliere_train.txt introuvable dans data/ ou data_pipeline/.")
        sys.exit(1)

    print(f"📖 Chargement de '{train_path}'...")
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"🔤 Apprentissage BPE en cours (vocab_size={vocab_size})...")
    merges, vocab = train_gpt2_bpe(text, vocab_size=vocab_size)

    special_tokens = {"<|endoftext|>": vocab_size}
    tokenizer = Tokenizer(merges, vocab, special_tokens=special_tokens)
    tokenizer.save(output_path)
    print(f"✅ Fichier '{output_path}' généré avec succès !")


if __name__ == "__main__":
    build_vocabulary()