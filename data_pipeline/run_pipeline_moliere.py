import os
import sys

# Ingestion dynamique des modules du dossier data_pipeline/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from clean_text import clean_text
from pii_masking import mask_pii
from minhash_dedup import deduplicate_documents


def run_full_pipeline(input_path: str, output_path: str):
    """Applique le pipeline complet : Nettoyage -> Masquage PII -> MinHash Deduplication."""
    if not os.path.exists(input_path):
        print(f"⚠️ Fichier introuvable : {input_path}")
        return

    print(f"🚀 Lancement du pipeline complet sur '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    # 1. Nettoyage Unicode et caractères de contrôle
    print("1️⃣ Nettoyage Unicode et espaces...")
    cleaned = clean_text(data, normalize_typography=True)

    # 2. Anonymisation PII
    print("2️⃣ Masquage des données sensibles (PII)...")
    masked = mask_pii(cleaned)

    # 3. Déduplication MinHash / LSH par paragraphe
    print("3️⃣ Déduplication MinHash...")
    paragraphs = [p.strip() for p in masked.split("\n\n") if p.strip()]
    deduped = deduplicate_documents(paragraphs, threshold=0.8)
    final_text = "\n\n".join(deduped)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Pipeline à 3 étapes achevé ➔ '{output_path}'")


if __name__ == "__main__":
    # Détection automatique du fichier d'origine
    candidates = [
        "data_pipeline/moliere_train.txt",
        "data/moliere_train.txt",
        "moliere_train.txt"
    ]
    input_file = next((p for p in candidates if os.path.exists(p)), "")

    if input_file:
        run_full_pipeline(input_file, "data_pipeline/moliere_train_pipeline.txt")
    else:
        print("⚠️ Fichier source moliere_train.txt introuvable.")