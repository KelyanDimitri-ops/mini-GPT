import re

WORD_RE = re.compile(r"\w+", re.UNICODE)

FR_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "et", "en", "un", "une", "que", "qui", "dans",
    "pour", "est", "au", "aux", "sur", "par", "pas", "ne", "ce", "il", "elle", "se", "je",
    "vous", "nous", "à", "a", "ou", "mais", "avec", "plus", "l", "d", "qu", "s", "n", "j", "c",
}


def passes_quality(text: str, min_words: int = 50, max_chars: int = 200_000):
    """Renvoie (True, 'ok') ou (False, raison). Filtres heuristiques de type Gopher/C4."""
    words = WORD_RE.findall(text.lower())
    n_words = len(words)
    if n_words < min_words:
        return False, "trop_court"
    if len(text) > max_chars:
        return False, "trop_long"

    mean_len = sum(len(w) for w in words) / n_words
    if not 3.0 <= mean_len <= 10.0:
        return False, "longueur_moyenne_des_mots"

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False, "vide"
    letters = sum(c.isalpha() for c in non_space)
    if letters / len(non_space) < 0.75:
        return False, "peu_de_lettres"

    symbols = sum(c in "{}[]<>|\\^~#" for c in non_space)
    if symbols / len(non_space) > 0.03:
        return False, "trop_de_symboles"

    stop_ratio = sum(w in FR_STOPWORDS for w in words) / n_words
    if stop_ratio < 0.12:
        return False, "pas_du_francais"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.7:
        return False, "lignes_repetees"

    return True, "ok"