import os
import re
import matplotlib.pyplot as plt


def parse_log_file(log_path: str):
    """
    Parse les logs d'entraînement pour extraire les métriques réelles.
    """
    iters, train_losses, val_losses = [], [], []
    train_bpbs, val_bpbs = [], []

    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Regex tolérante prenant en compte la mention LR: 0.000xxx ---
    pattern = (
        r'---\s*\[Iter\s+(\d+)/\d+\].*?---\s*'
        r'Train Loss:\s+([\d.]+)\s+nats\s*\|\s*Train BPB:\s+([\d.]+)\s*'
        r'Val\s+Loss:\s+([\d.]+)\s+nats\s*\|\s*Val\s+BPB:\s+([\d.]+)'
    )

    matches = re.findall(pattern, content, re.DOTALL)
    for it, tr_l, tr_b, val_l, val_b in matches:
        iters.append(int(it))
        train_losses.append(float(tr_l))
        train_bpbs.append(float(tr_b))
        val_losses.append(float(val_l))
        val_bpbs.append(float(val_b))

    return iters, train_losses, val_losses, train_bpbs, val_bpbs


def generate_loss_curve(log_path: str = "train.log", output_path: str = "docs/loss_curve.png"):
    parsed = parse_log_file(log_path)

    if parsed and len(parsed[0]) > 0:
        iters, train_losses, val_losses, train_bpbs, val_bpbs = parsed
        print(f"✅ {len(iters)} points de mesure réels extraits depuis '{log_path}' !")
    else:
        print(f"⚠️ Aucun motif valide trouvé dans '{log_path}'. Vérifie le contenu du fichier.")
        return

    # Style visuel
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Graphique 1 : Loss
    ax1.plot(iters, train_losses, label="Train Loss", color="#1f77b4", linewidth=2, marker='o', markersize=4)
    ax1.plot(iters, val_losses, label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=2, marker='s', markersize=4)
    ax1.set_title("Évolution de la Perte (Cross-Entropy Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Itérations")
    ax1.set_ylabel("Loss (nats)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Graphique 2 : BPB
    ax2.plot(iters, train_bpbs, label="Train BPB", color="#2ca02c", linewidth=2, marker='o', markersize=4)
    ax2.plot(iters, val_bpbs, label="Val BPB", color="#d62728", linestyle="--", linewidth=2, marker='s', markersize=4)
    ax2.set_title("Évolution du score BPB (Bits Per Byte)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Itérations")
    ax2.set_ylabel("BPB")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Performance du Modèle mini-GPT (Données Réelles)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Graphique réel sauvegardé sous '{output_path}'")


if __name__ == "__main__":
    generate_loss_curve()