import re
import os

# Motifs Regex adaptés aux formats de données français et internationaux
PII_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    # Numéros de téléphone français (formats: 0612345678, +33 6 12 34 56 78, 01.23.45.67.89)
    "PHONE": r'(?:(?:\+|00)33[\s.-]?(?:\(0\)[\s.-]?)?|0)[1-9](?:[\s.-]?\d{2}){4}\b',
    # Adresses IP IPv4
    "IP_ADDRESS": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    # Cartes bancaires (16 chiffres séparés ou non)
    "CREDIT_CARD": r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
    # Numéro de Sécurité Sociale Français (NIR : 15 chiffres)
    "SOCIAL_SECURITY": r'\b[12]\s?\d{2}\s?(?:0[1-9]|1[0-2])\s?(?:2[AB]|\d{2})\s?\d{3}\s?\d{3}(?:\s?\d{2})?\b',
}


def mask_pii(text: str) -> str:
    """
    Détecte et remplace les informations personnelles sensibles par des tokens anonymisés.
    """
    for tag, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"<{tag}>", text)
    return text


def process_pii_file(input_path: str, output_path: str):
    """Traite un fichier et génère la version anonymisée sur disque."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    masked_content = mask_pii(content)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(masked_content)

    print(f"🛡️ Masquage PII appliqué : '{input_path}' ➔ '{output_path}'")


if __name__ == "__main__":
    # Test unitaire de validation PII
    sample_text = (
        "Contactez-moi à kelyan.test@ece.fr ou au 06 12 34 56 78.\n"
        "Mon IP est 192.168.1.50 et mon NIR est 1 05 06 75 123 456 78."
    )
    print("--- Texte Brut ---")
    print(sample_text)
    print("\n--- Texte Anonymisé ---")
    print(mask_pii(sample_text))