import re
import os
import random
from typing import List, Set, Dict


class MinHashLSH:
    """
    Implémentation de MinHash et LSH (Locality-Sensitive Hashing)
    pour la déduplication efficace de textes quasi-identiques.
    """

    def __init__(self, num_perm: int = 64, threshold: float = 0.8, k_shingle: int = 3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.k_shingle = k_shingle

        # Grand nombre premier pour les fonctions de hachage h(x) = (a*x + b) % PRIME
        self._prime = 2147483647

        # Génération déterministe des constantes de hachage (a, b)
        gen = random.Random(42)
        self._a = [gen.randint(1, self._prime - 1) for _ in range(num_perm)]
        self._b = [gen.randint(0, self._prime - 1) for _ in range(num_perm)]

    def get_shingles(self, text: str) -> Set[int]:
        """Découpe un texte en k-shingles (n-grammes de mots) et calcule leurs hachages."""
        words = re.findall(r'\w+', text.lower())
        if len(words) < self.k_shingle:
            return {hash(" ".join(words)) & 0xFFFFFFFF}

        shingles = set()
        for i in range(len(words) - self.k_shingle + 1):
            shingle = " ".join(words[i: i + self.k_shingle])
            # Masquage 32-bit sans signe
            shingles.add(hash(shingle) & 0xFFFFFFFF)
        return shingles

    def compute_signature(self, shingles: Set[int]) -> List[int]:
        """Calcule la signature MinHash (vecteur de taille num_perm)."""
        if not shingles:
            return [0] * self.num_perm

        sig = []
        for a, b in zip(self._a, self._b):
            min_hash = float('inf')
            for s in shingles:
                h = (a * s + b) % self._prime
                if h < min_hash:
                    min_hash = h
            sig.append(min_hash)
        return sig

    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estime l'indice de similitude de Jaccard entre deux signatures MinHash."""
        equal_count = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return equal_count / len(sig1)


def deduplicate_documents(documents: List[str], threshold: float = 0.8, num_perm: int = 64) -> List[str]:
    """
    Filtre une liste de paragraphes/documents en retirant les doublons quasi-identiques.
    """
    lsh = MinHashLSH(num_perm=num_perm, threshold=threshold)
    signatures = []

    print(f"🔍 Calcul des signatures MinHash pour {len(documents)} paragraphes...")
    for doc in documents:
        shingles = lsh.get_shingles(doc)
        sig = lsh.compute_signature(shingles)
        signatures.append(sig)

    kept_documents = []
    kept_signatures = []

    print("⚡ Filtrage des doublons quasi-identiques...")
    for i, doc in enumerate(documents):
        is_duplicate = False
        sig = signatures[i]

        # Comparaison avec les documents déjà conservés
        for kept_sig in kept_signatures:
            sim = lsh.jaccard_similarity(sig, kept_sig)
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_documents.append(doc)
            kept_signatures.append(sig)

    removed_count = len(documents) - len(kept_documents)
    print(f"✅ Déduplication terminée : {removed_count} doublons supprimés sur {len(documents)} blocs.")
    return kept_documents


def deduplicate_file(input_path: str, output_path: str, threshold: float = 0.8):
    """Lit un fichier texte, supprime les paragraphes en double et sauvegarde le résultat."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Découpage par paragraphes (blocs séparés par deux sauts de ligne)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    deduped_paragraphs = deduplicate_documents(paragraphs, threshold=threshold)
    cleaned_content = "\n\n".join(deduped_paragraphs)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    print(f"💾 Fichier dédupliqué sauvegardé ➔ '{output_path}'")


if __name__ == "__main__":
    # Test unitaire rapide
    doc1 = "Le comique de Molière réside dans la satire des mœurs et l'exagération des traits de caractère."
    doc2 = "Le comique de Moliere reside dans la satire des moeurs et l'exagération des traits de caractere."  # Quasi-doublon
    doc3 = "L'avare est une comédie en cinq actes rédigée en prose par Molière."

    sample_docs = [doc1, doc2, doc3]
    result = deduplicate_documents(sample_docs, threshold=0.75)

    print("\n--- RÉSULTAT DU TEST ---")
    print(f"Nombre de documents originaux : {len(sample_docs)}")
    print(f"Nombre de documents conservés : {len(result)}")
    assert len(result) == 2, "Erreur: Le quasi-doublon aurait dû être filtré !"