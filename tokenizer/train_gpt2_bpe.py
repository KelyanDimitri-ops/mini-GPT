import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids_chunks: list[list[int]]) -> dict[tuple[int, int], int]:
    """Compte la fréquence des paires uniquement À L'INTÉRIEUR de chaque morceau."""
    counts = {}
    for chunk in ids_chunks:
        for pair in zip(chunk, chunk[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_chunk(chunk: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Applique la fusion sur un seul morceau d'octets."""
    new_chunk = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i + 1] == pair[1]:
            new_chunk.append(idx)
            i += 2
        else:
            new_chunk.append(chunk[i])
            i += 1
    return new_chunk


def train_gpt2_bpe(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """
    Entraîne BPE avec la pré-tokenisation Regex de GPT-2.
    """
    assert vocab_size >= 256, "Vocabulaire minimal de 256 pour la base UTF-8"
    num_merges = vocab_size - 256

    # 1. Découpage du texte en morceaux via la Regex GPT-2
    compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)
    raw_chunks = re.findall(compiled_pattern, text)

    # 2. Conversion de chaque morceau en sa propre liste d'octets UTF-8
    ids_chunks = [list(chunk.encode("utf-8")) for chunk in raw_chunks]

    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    print(f"Text découpé en {len(ids_chunks)} morceaux distincts.")

    for i in range(num_merges):
        # Compter les paires sans croiser les morceaux
        stats = get_stats(ids_chunks)
        if not stats:
            break

        top_pair = max(stats, key=stats.get)
        count = stats[top_pair]

        if count < 2:
            print(f"Arrêt précoce : plus de paire fréquente (étape {i}).")
            break

        idx = 256 + i

        # Fusionner la paire dans chaque morceau séparément
        ids_chunks = [merge_chunk(chunk, top_pair, idx) for chunk in ids_chunks]

        merges[top_pair] = idx
        vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

    return merges, vocab


if __name__ == "__main__":
    corpus = "L'apprentissage du langage naturel avec un mini-GPT ! Le chat, le chien."

    merges, vocab = train_gpt2_bpe(corpus, vocab_size=275)

    print(f"Nombre de fusions apprises : {len(merges)}")