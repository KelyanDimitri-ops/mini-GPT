import random
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.value import Value


class BigramNNModel:
    """
    Modèle Bigramme implémenté sous forme de réseau de neurones à une couche
    entraîné via notre moteur autograd scalaire (Value).
    """

    def __init__(self, text: str):
        # Vocabulaire au niveau caractère
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Matrice de poids W de taille (V, V) initialisée aléatoirement
        # W[i][j] représente le score brut (logit) de transition du caractère i vers j
        self.W = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(self.vocab_size)]
            for _ in range(self.vocab_size)
        ]

    def parameters(self) -> list[Value]:
        """Retourne la liste à plat de tous les paramètres du modèle (V * V poids)."""
        return [w for row in self.W for w in row]

    def zero_grad(self):
        """Réinitialise les gradients de tous les poids à zéro."""
        for p in self.parameters():
            p.grad = 0.0

    def forward_pair(self, ix_in: int, ix_target: int) -> Value:
        """
        Passe avant pour un couple (caractère_entrée, caractère_cible) :
        1. Extraction des logits W[ix_in]
        2. Softmax via exp() et division
        3. Perte NLL via -log(probabilité_cible)
        """
        logits = self.W[ix_in]

        # 1. Softmax : exp(logit_k) / sum(exp(logit_m))
        counts = [logit.exp() for logit in logits]

        # Somme de tous les counts pour le dénominateur
        total_count = counts[0]
        for c in counts[1:]:
            total_count = total_count + c

        # Probabilité du vrai caractère suivant
        prob_target = counts[ix_target] / total_count

        # 2. Perte NLL pour ce caractère : -ln(P(c_t | c_{t-1}))
        loss = -prob_target.log()
        return loss

    def train(self, text: str, epochs: int = 50, lr: float = 10.0):
        """Boucle d'entraînement par descente de gradient."""
        # Préparation des couples (entrée, cible)
        pairs = [
            (self.stoi[ch1], self.stoi[ch2])
            for ch1, ch2 in zip(text, text[1:])
            if ch1 in self.stoi and ch2 in self.stoi
        ]

        print(f"Entraînement du modèle Bigramme NN sur {len(pairs)} exemples...")

        for epoch in range(epochs):
            # 1. Remise à zéro des gradients
            self.zero_grad()

            # 2. Accumulation de la perte moyenne sur le dataset
            total_loss = Value(0.0)
            for ix_in, ix_target in pairs:
                total_loss = total_loss + self.forward_pair(ix_in, ix_target)

            mean_loss = total_loss / len(pairs)

            # 3. Rétropropagation
            mean_loss.backward()

            # 4. Descente de gradient (mise à jour des poids W)
            for p in self.parameters():
                p.data -= lr * p.grad

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Époque {epoch + 1:02d}/{epochs} | Loss (NLL): {mean_loss.data:.4f} nats")


if __name__ == "__main__":
    dataset = "et être ou ne pas être, telle est la question. être ou ne pas être."

    model = BigramNNModel(dataset)
    model.train(dataset, epochs=50, lr=15.0)