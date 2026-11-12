import argparse, json, os
from datasets import load_dataset

def collect(name, config, out_path, max_mb, min_chars=200, seed=42):
    ds = load_dataset(name, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_bytes, n_docs, limit = 0, 0, max_mb * 1_000_000
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex["text"]
            if len(text) < min_chars:
                continue
            f.write(json.dumps({"text": text, "source": name}, ensure_ascii=False) + "\n")
            n_bytes += len(text.encode("utf-8"))
            n_docs += 1
            if n_docs % 10_000 == 0:
                print(f"{n_docs} docs, {n_bytes / 1e6:.0f} Mo")
            if n_bytes >= limit:
                break
    print(f"Terminé : {n_docs} docs, {n_bytes / 1e6:.0f} Mo -> {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-mb", type=int, required=True)
    a = p.parse_args()
    collect(a.name, a.config, a.out, a.max_mb)