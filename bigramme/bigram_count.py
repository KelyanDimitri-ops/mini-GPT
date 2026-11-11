import torch


class BigramCountModel:
    """
    Modèle de langage Bigramme basé sur les fréquences de co-occurrence des caractères.
    """

    def __init__(self, text: str, smoothing: float = 1.0):
        # 1. Construction du vocabulaire (niveau caractère)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # 2. Matrice de comptage N de taille (V, V)
        N = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.int32)
        for ch1, ch2 in zip(text, text[1:]):
            ix1 = self.stoi[ch1]
            ix2 = self.stoi[ch2]
            N[ix1, ix2] += 1

        # 3. Lissage laplacien (Add-1 smoothing par défaut) et normalisation
        # P[i, j] = P(caractère j sachant le caractère i)
        P = (N + smoothing).float()
        self.P = P / P.sum(dim=1, keepdim=True)

    def evaluate_loss(self, text: str) -> tuple[float, float]:
        """
        Calcule la Negative Log-Likelihood (NLL) moyenne et le BPB (Bits Per Byte).
        """
        log_likelihood = 0.0
        n = 0

        for ch1, ch2 in zip(text, text[1:]):
            if ch1 in self.stoi and ch2 in self.stoi:
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                prob = self.P[ix1, ix2]
                log_likelihood += torch.log(prob).item()
                n += 1

        nll = -log_likelihood / n  # Loss en nats
        bpb = nll / math.log(2)  # Conversion en bits par octet (BPB)
        return nll, bpb

    def generate(self, start_char: str, max_new_tokens: int = 200, seed: int = 42) -> str:
        """
        Génère du texte par échantillonnage dans la distribution de probabilité Bigramme.
        """
        g = torch.Generator().manual_seed(seed)
        out = [start_char]
        curr_ix = self.stoi.get(start_char, 0)

        for _ in range(max_new_tokens):
            p = self.P[curr_ix]
            # Échantillonnage selon la distribution de probabilité p
            curr_ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(self.itos[curr_ix])

        return "".join(out)


if __name__ == "__main__":
    import math

    # Exemple sur un texte de démonstration
    dataset = "et être ou ne pas être, telle est la question. être ou ne pas être."

    model = BigramCountModel(dataset, smoothing=1.0)
    loss, bpb = model.evaluate_loss(dataset)

    print(f"Vocabulaire ({model.vocab_size} caractères) : {model.chars}")
    print(f"Perte NLL (Loss) : {loss:.4f} nats")
    print(f"Perte BPB        : {bpb:.4f} bits/byte")
    print("\nTexte généré :")
    print(model.generate(start_char="a", max_new_tokens=10))