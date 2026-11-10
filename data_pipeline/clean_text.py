import unicodedata
import re
import os


def clean_text(text: str) -> str:
    """
    Nettoie et normalise une chaîne de texte brut.
    - Normalisation Unicode NFKC (harmonise ligatures et diacritiques)
    - Suppression des caractères de contrôle non imprimables (hors \\n et \\t)
    - Normalisation des guillemets et apostrophes
    - Suppression des espaces horizontaux superflus
    """
    # 1. Normalisation Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Suppression des caractères de contrôle invisibles (conservant \n et \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # 3. Standardisation des guillemets et apostrophes
    text = re.sub(r'[«»“”]', '"', text)
    text = re.sub(r"[’`‘]", "'", text)

    # 4. Réduction des espaces horizontaux multiples sans impacter les sauts de ligne
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Limitation des sauts de ligne consécutifs à 2 maximum
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text.strip()


def clean_file(input_path: str, output_path: str):
    """Lit un fichier, applique le nettoyage et sauvegarde le résultat."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    cleaned_data = clean_text(raw_data)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_data)

    print(f"✅ Fichier nettoyé : '{input_path}' ({len(raw_data)} car.) ➔ '{output_path}' ({len(cleaned_data)} car.)")


if __name__ == "__main__":
    # Test unitaire rapide
    sample = "L'élève  a dit : « Bonjour ! »\x00   Voici mon   email \t et un saut.\n\n\nFin."
    print("--- Avant ---")
    print(repr(sample))
    print("--- Après ---")
    print(repr(clean_text(sample)))