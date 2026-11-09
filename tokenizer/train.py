import os
import sys
import time
import math
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gpt.model import GPT, GPTConfig
from tokenizer.tokenizer import tokenizer

# -----------------------------------------------------------------------------
# 1. Hyperparamètres d'entraînement
# -----------------------------------------------------------------------------
# Matériel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Données & Modèle
batch_size = 16  # Nombre de séquences par batch
block_size = 256  # Longueur du contexte (T)
n_layer = 6  # Nombre de blocs Transformer
n_head = 6  # Nombre de têtes d'attention
n_embd = 384  # Dimension des embeddings (~10.8M de paramètres)

# Optimisation & Schedule
learning_rate = 6e-4  # LR maximal
min_lr = 6e-5  # LR minimal en fin de Cosine Decay
warmup_iters = 100  # Nombre d'itérations de chauffe
max_iters = 2000  # Nombre total d'itérations d'entraînement
weight_decay = 0.1  # Régularisation L2 (AdamW)
beta1, beta2 = 0.9, 0.95

# Évaluation et Sauvegarde
eval_interval = 200  # Calcul du BPB et échantillons toutes les X itérations
eval_iters = 20  # Nombre de batchs pour estimer la loss
out_dir = 'checkpoints'
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Chargement du Tokenizer et des Données
# -----------------------------------------------------------------------------
tokenizer_path = "tokenizer/vocab_500.json"  # Fichier généré lors de la S1
if not os.path.exists(tokenizer_path):
    print(f"Tokenizer introuvable à '{tokenizer_path}'. Entraîne-le d'abord !")
    sys.exit(1)

tokenizer = tokenizer.load(tokenizer_path)
vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)

# Chargement du corpus (ex: moliere.txt)
data_path = "data/moliere.txt"
if not os.path.exists(data_path):
    print(f"⚠️ Fichier de données introuvable à '{data_path}'.")
    sys.exit(1)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Encodage de l'intégralité du texte en tokens
tokens = tokenizer.encode(text)
total_bytes = len(text.encode("utf-8"))
total_tokens = len(tokens)
bytes_per_token = total_bytes / total_tokens

print(f" Corpus chargé : {len(text)} caractères | {total_bytes} octets")
print(f" Tokens générés : {total_tokens} | Taux de compression : {bytes_per_token:.2f} octets/token")

# Split Train / Validation (90% / 10%)
data_tensor = torch.tensor(tokens, dtype=torch.long)
n_train = int(0.9 * len(data_tensor))
train_data = data_tensor[:n_train]
val_data = data_tensor[n_train:]


def get_batch(split):
    """Génère un mini-batch d'entrées (x) et de cibles décalées d'un token (y)."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# 3. Initialisation du Modèle et de l'Optimiseur
# -----------------------------------------------------------------------------
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd
)
model = GPT(config).to(device)


def configure_optimizers(model, weight_decay, learning_rate, betas):
    """
    Sépare les paramètres 2D (poids linéaires/embeddings -> weight decay)
    des paramètres 1D (biais, LayerNorm -> pas de weight decay).
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2))


# -----------------------------------------------------------------------------
# 4. Planificateur de Taux d'Apprentissage (Cosine Decay avec Warmup)
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# 5. Boucle de Validation & Calcul du BPB
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss_and_bpb():
    """
    Calcule la perte moyenne (en nats) et la convertit en Bits Per Byte (BPB).
    BPB = (Loss_nats / ln(2)) * (Total_Tokens / Total_Bytes)
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()

        # Formule exacte de conversion NLL (nats) vers BPB
        bpb = (mean_loss / math.log(2)) / bytes_per_token
        out[split] = (mean_loss, bpb)

    model.train()
    return out


# -----------------------------------------------------------------------------
# 6. Boucle d'Entraînement Principale
# -----------------------------------------------------------------------------
print(f"\n🚀 Lancement de l'entraînement sur {device.upper()} ({max_iters} itérations)...")
start_time = time.time()

for iter_num in range(1, max_iters + 1):

    # Ajustement du Learning Rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Évaluation périodique
    if iter_num % eval_interval == 0 or iter_num == 1:
        res = estimate_loss_and_bpb()
        train_loss, train_bpb = res['train']
        val_loss, val_bpb = res['val']

        print(f"\n--- [Iter {iter_num:04d}/{max_iters}] LR: {lr:.6f} ---")
        print(f"Train Loss: {train_loss:.4f} nats | Train BPB: {train_bpb:.4f}")
        print(f"Val   Loss: {val_loss:.4f} nats | Val   BPB: {val_bpb:.4f}")

        # Génération d'un échantillon de texte de démonstration
        prompt = "LE COMIQUE :"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n Échantillon généré :\n{generated_text}\n" + "-" * 50)

        # Sauvegarde du meilleur checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter': iter_num,
            'val_bpb': val_bpb,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))

    # Forward / Backward Pass
    X, Y = get_batch('train')
    logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient Clipping pour éviter l'explosion des gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

exec_time = time.time() - start_time
print(f"\nEntraînement terminé en {exec_time / 60:.2f} minutes !")