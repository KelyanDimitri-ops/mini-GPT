import regex as re
import json
import os

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    Tokenizer BPE au niveau des octets avec pré-tokenisation Regex (style GPT-2/GPT-4).
    """

    def __init__(self, merges: dict[tuple[int, int], int], vocab: dict[int, bytes],
                 special_tokens: dict[str, int] = None):
        self.merges = merges  # (id1, id2) -> new_id
        self.vocab = vocab  # id -> bytes
        self.special_tokens = special_tokens or {}  # "<|endoftext|>" -> 10000
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        self.compiled_pattern = re.compile(GPT2_SPLIT_PATTERN)

    def _encode_chunk(self, chunk_bytes: list[int]) -> list[int]:
        """
        Applique les fusions BPE de manière itérative sur un morceau d'octets.
        Priorité absolue : fusionner la paire qui a le plus petit nouvel ID (apprise en premier).
        """
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # 1. Lister toutes les paires adjacentes présentes dans le morceau
            stats = {pair: i for i, pair in enumerate(zip(ids, ids[1:]))}

            # 2. Trouver la paire présente qui a le rang de fusion le plus prioritaire
            pair_to_merge = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))

            # Si aucune paire du morceau n'est dans les merges appris, on arrête
            if pair_to_merge not in self.merges:
                break

            idx = self.merges[pair_to_merge]

            # 3. Appliquer la fusion
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair_to_merge[0] and ids[i + 1] == pair_to_merge[1]:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids

    def encode(self, text: str, allowed_special: set = "all") -> list[int]:
        """
        Convertit un texte en liste d'identifiants (tokens).
        """
        if allowed_special == "all":
            special_flags = set(self.special_tokens.keys())
        else:
            special_flags = allowed_special

        if special_flags:
            special_pattern = "(" + "|".join(re.escape(tok) for tok in special_flags) + ")"
            special_chunks = re.split(special_pattern, text)
        else:
            special_chunks = [text]

        tokens = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                tokens.append(self.special_tokens[chunk])
            else:
                raw_chunks = re.findall(self.compiled_pattern, chunk)
                for raw_chunk in raw_chunks:
                    chunk_bytes = list(raw_chunk.encode("utf-8"))
                    merged_ids = self._encode_chunk(chunk_bytes)
                    tokens.extend(merged_ids)

        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Reconstruit le texte d'origine à partir d'une liste d'identifiants de tokens.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.special_tokens.values():
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"ID invalide lors du décodage: {idx}")

        full_bytes = b"".join(part_bytes)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """
        Enregistre la configuration minimale du Tokenizer (merges et tokens spéciaux)
        dans un fichier JSON sur le disque.
        """
        serialized_merges = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}

        payload = {
            "special_tokens": self.special_tokens,
            "merges": serialized_merges
        }

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"💾 Tokenizer sauvegardé avec succès dans '{file_path}'")

    @classmethod
    def load(cls, file_path: str):
        """
        Reconstruit une instance complète de Tokenizer depuis un fichier JSON
        en régénérant déterministement le vocabulaire d'octets.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier de tokenizer introuvable : {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        merges = {}
        for pair_str, idx in payload["merges"].items():
            p0, p1 = map(int, pair_str.split(","))
            merges[(p0, p1)] = idx

        special_tokens = payload.get("special_tokens", {})

        # Reconstitution déterministe du vocabulaire d'octets
        vocab = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(merges.items(), key=lambda item: item[1])
        for (p0, p1), idx in sorted_merges:
            vocab[idx] = vocab[p0] + vocab[p1]

        print(f"📖 Tokenizer chargé depuis '{file_path}' ({len(vocab)} tokens dans le vocabulaire)")
        return cls(merges=merges, vocab=vocab, special_tokens=special_tokens)


if __name__ == "__main__":
    from train_gpt2_bpe import train_gpt2_bpe

    text_train = "L'apprentissage avec mini-GPT est passionnant ! Le chat mange la souris."
    merges, vocab = train_gpt2_bpe(text_train, vocab_size=280)

    special_toks = {"<|endoftext|>": 280}
    tokenizer = Tokenizer(merges, vocab, special_tokens=special_toks)

    sample = "Le chat mange la souris. <|endoftext|>"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("\n--- TEST DU TOKENIZER ---")
    print(f"Texte original  : '{sample}'")
    print(f"Tokens (IDs)    : {encoded}")
    print(f"Texte réassemblé: '{decoded}'")
    assert sample == decoded, "Erreur: Le texte décodé ne correspond pas au texte d'origine !"
    print("✅ Succès: Encodage et décodage parfaitement réversibles !")

    # Test de sauvegarde et rechargement via Tokenizer.save / Tokenizer.load
    test_path = "tokenizer/test_vocab.json"
    tokenizer.save(test_path)
    loaded_tokenizer = Tokenizer.load(test_path)

    assert loaded_tokenizer.encode(sample) == encoded, "Erreur: Le tokenizer rechargé ne produit pas les mêmes tokens !"
    print("✅ Succès: Sauvegarde et rechargement JSON validés !")

    if os.path.exists(test_path):
        os.remove(test_path)