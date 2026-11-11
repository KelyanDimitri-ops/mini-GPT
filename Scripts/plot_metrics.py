import os
import re
import matplotlib.pyplot as plt


def read_log_robustly(log_path: str) -> str:
    """Lit le fichier de log en essayant plusieurs encodages (UTF-16 de PowerShell, UTF-8, Latin-1)."""
    encodings = ["utf-16", "utf-16-le", "utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(log_path, "r", encoding=enc) as f:
                content = f.read()
                if "Iter" in content:
                    return content
        except Exception:
            continue
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_log_file(log_path: str):
    if not os.path.exists(log_path):
        return None

    content = read_log_robustly(log_path)

    iters = [int(x) for x in re.findall(r'\[Iter\s+(\d+)/\d+\]', content)]
    train_losses = [float(x) for x in re.findall(r'Train Loss:\s+([\d.]+)', content)]
    train_bpbs = [float(x) for x in re.findall(r'Train BPB:\s+([\d.]+)', content)]
    val_losses = [float(x) for x in re.findall(r'Val\s+Loss:\s+([\d.]+)', content)]
    val_bpbs = [float(x) for x in re.findall(r'Val\s+BPB:\s+([\d.]+)', content)]

    n = min(len(iters), len(train_losses), len(train_bpbs), len(val_losses), len(val_bpbs))
    if n == 0:
        return None

    return iters[:n], train_losses[:n], val_losses[:n], train_bpbs[:n], val_bpbs[:n]


def generate_loss_curve(log_path: str = "train.log", output_path: str = "docs/loss_curve.png"):
    parsed = parse_log_file(log_path)

    if parsed:
        iters, train_losses, val_losses, train_bpbs, val_bpbs = parsed
        print(f"✅ {len(iters)} points de mesure réels extraits avec succès depuis '{log_path}' !")
    else:
        print(f"⚠️ Aucun motif valide trouvé dans '{log_path}'.")
        return

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

    plt.suptitle("Performance du Modèle mini-GPT (Corpus Dédupliqué - 9.81M)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Graphique réel sauvegardé sous '{output_path}'")


if __name__ == "__main__":
    generate_loss_curve()