import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Mécanisme d'attention multi-têtes causale (Multi-Head Causal Self-Attention).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Projections Q, K, V combinées en une seule matrice linéaire
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Projection de sortie
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Masque causal : matrice triangulaire inférieure
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                              config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time-steps (séquence), Channels (n_embd)

        # Calcul de Q, K, V en une seule multiplication matricielle
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Redimensionnement pour le multi-têtes : (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Matrice de scores : (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # Application du masque causal (-inf sur le futur)
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Combinaison pondérée des valeurs V
        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Concaténation des têtes

        # Projection finale
        return self.c_proj(y)


class MLP(nn.Module):
    """
    Réseau Feed-Forward position-wise avec expansion x4 et activation GELU.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Bloc Transformer Décodeur standard avec architecture Pre-LayerNorm (style GPT-2).
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN et connexions résiduelles (+ x)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    """Configuration des hyperparamètres du modèle Mini-GPT (~10M de paramètres)."""

    def __init__(self, vocab_size=50257, block_size=256, n_layer=6, n_head=6, n_embd=384):
        self.vocab_size = vocab_size
        self.block_size = block_size  # Longueur max du contexte (context window)
        self.n_layer = n_layer  # Nombre de blocs Transformer
        self.n_head = n_head  # Nombre de têtes d'attention
        self.n_embd = n_embd  # Dimension des embeddings


class GPT(nn.Module):
    """
    Modèle GPT complet : Embeddings -> Layers Transformer -> Final LN -> Head LM.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Tête de sortie du modèle de langage (LM Head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Partage des poids entre WTE et LM Head (Weight Tying)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Séquence de taille {t} dépasse la limite de {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)

        # Forward des embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = tok_emb + pos_emb

        # Passage dans tous les blocs Transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calcul de la perte Cross-Entropy si les cibles sont fournies
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Mode inférence : on ne garde que le logit du dernier token de la séquence
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génération autoregressive de texte à partir d'un prompt (idx).
        """
        for _ in range(max_new_tokens):
            # Recadrage du contexte si la séquence dépasse block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx