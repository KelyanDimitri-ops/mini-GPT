import unicodedata
import re
import os


def clean_text(text: str, normalize_typography: bool = False) -> str:
    """
    Par défaut : NFC seulement ; « » ’ œ … et les espaces insécables sont conservés.
    normalize_typography=True : ancien comportement (NFKC + guillemets/apostrophes simplifiés).
    """
    if normalize_typography:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFC", text)

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    if normalize_typography:
        text = re.sub(r'[«»“”]', '"', text)
        text = re.sub(r"[’`‘]", "'", text)

    text = re.sub(r'[ \t]+', ' ', text)
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