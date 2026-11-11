import os
import sys
import time
import math
import io
import torch
import torch.nn as nn

# Force l'encodage UTF-8 sous Windows pour éviter les plantages d'émojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ajout du dossier racine au sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gpt.model import GPT, GPTConfig
from tokenizer.tokenizer import Tokenizer


def get_data_path(filename: str) -> str:
    """Cherche automatiquement le fichier dans data_pipeline/, data/ ou à la racine."""
    candidates = [
        os.path.join("data_pipeline", filename),
        os.path.join("data", filename),
        filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""

# -----------------------------------------------------------------------------
# 1. Hyperparamètres d'entraînement
# -----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Données & Modèle
batch_size = 16   # Nombre de séquences par batch
block_size = 256  # Longueur du contexte (T)
n_layer = 6       # Nombre de blocs Transformer
n_head = 6        # Nombre de têtes d'attention
n_embd = 360      # Dimension des embeddings (~9.81M de paramètres)

# Optimisation & Schedule
learning_rate = 6e-4  # LR maximal
min_lr = 6e-5         # LR minimal en fin de Cosine Decay
warmup_iters = 100    # Nombre d'itérations de chauffe
max_iters = 2000      # Nombre total d'itérations d'entraînement
weight_decay = 0.1    # Régularisation L2 (AdamW)
beta1, beta2 = 0.9, 0.95

# Évaluation et Sauvegarde
eval_interval = 200  # Calcul du BPB et échantillons toutes les X itérations
eval_iters = 20      # Nombre de batchs pour estimer la loss
out_dir = 'checkpoints'
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Chargement du Tokenizer et des Données
# -----------------------------------------------------------------------------
tokenizer_path = "tokenizer/vocab_1000.json"
if not os.path.exists(tokenizer_path):
    print(f"⚠️ Tokenizer introuvable à '{tokenizer_path}'. Exécute d'abord python tokenizer/build_vocab.py !")
    sys.exit(1)

tokenizer = Tokenizer.load(tokenizer_path)
vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)

# Chargement prioritaire du dataset dédupliqué par MinHash et filtré PII
train_path = get_data_path("moliere_train_pipeline.txt")
if not train_path:
    train_path = get_data_path("moliere_train.txt")

val_path = get_data_path("moliere_val.txt")

if not train_path or not val_path:
    print("⚠️ Fichiers d'entraînement et/ou de validation introuvables.")
    sys.exit(1)

with open(train_path, "r", encoding="utf-8") as f:
    train_text = f.read()

with open(val_path, "r", encoding="utf-8") as f:
    val_text = f.read()

# Encodage du texte en tokens
train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)

# Calcul du taux de compression sur la validation pour le BPB
val_bytes = len(val_text.encode("utf-8"))
val_token_count = len(val_tokens)
bytes_per_token = val_bytes / val_token_count

print(f"📖 Train : {len(train_text)} car | {len(train_tokens)} tokens ({train_path})")
print(f"📖 Val   : {len(val_text)} car | {val_token_count} tokens ({val_path})")
print(f"🔤 Taux de compression Validation : {bytes_per_token:.2f} octets/token")

# Transformation en Tenseurs PyTorch
train_data = torch.tensor(train_tokens, dtype=torch.long)
val_data = torch.tensor(val_tokens, dtype=torch.long)


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
    Sépare les paramètres 2D (matrices de poids) des 1D (biais, LayerNorm).
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = []
    nodecay_params = []

    for pn, p in param_dict.items():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


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
    """Calcule la perte moyenne (nats) et la convertit en Bits Per Byte (BPB)."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()

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

    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 or iter_num == 1:
        res = estimate_loss_and_bpb()
        train_loss, train_bpb = res['train']
        val_loss, val_bpb = res['val']

        print(f"\n--- [Iter {iter_num:04d}/{max_iters}] LR: {lr:.6f} ---")
        print(f"Train Loss: {train_loss:.4f} nats | Train BPB: {train_bpb:.4f}")
        print(f"Val   Loss: {val_loss:.4f} nats | Val   BPB: {val_bpb:.4f}")

        # Échantillon de génération
        prompt = "LE COMIQUE :"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n📝 Échantillon généré :\n{generated_text}\n" + "-" * 50)

        # Sauvegarde du checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter': iter_num,
            'val_bpb': val_bpb,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))

    X, Y = get_batch('train')
    logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

exec_time = time.time() - start_time
print(f"\n✅ Entraînement terminé en {exec_time / 60:.2f} minutes !")