"""
Pipeline jsonl -> jsonl :
nettoyage -> filtres qualité/langue -> masquage PII -> dédup exacte -> dédup MinHash (LSH) -> mélange.
"""
import argparse, hashlib, json, os, random, re, sys, time
from collections import Counter

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from clean_text import clean_text
from pii_masking import mask_pii
from quality_filter import passes_quality
from datasketch import MinHash, MinHashLSH

WORD_RE = re.compile(r"\w+", re.UNICODE)


def minhash_of(text: str, num_perm: int, k: int = 5) -> MinHash:
    """Signature MinHash d'un document à partir de ses k-grammes de mots (hachage déterministe)."""
    words = WORD_RE.findall(text.lower())
    m = MinHash(num_perm=num_perm, seed=1)
    grams = [" ".join(words[i:i + k]).encode("utf-8") for i in range(max(1, len(words) - k + 1))]
    m.update_batch(grams)
    return m


def run(inputs, out_path, limit=None, normalize_typography=False,
        threshold=0.8, num_perm=64, seed=1337):
    t0 = time.time()
    stats = Counter()
    kept = []
    seen_exact = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for path in inputs:
        source = os.path.splitext(os.path.basename(path))[0]
        print(f"--- Lecture de {path}")
        with open(path, "r", encoding="utf-8") as f:
            for n, line in enumerate(f):
                if limit and n >= limit:
                    break
                raw = json.loads(line)["text"]
                stats[f"{source}/docs_entree"] += 1
                stats[f"{source}/octets_entree"] += len(raw.encode("utf-8"))

                text = clean_text(raw, normalize_typography=normalize_typography)

                ok, reason = passes_quality(text)
                if not ok:
                    stats[f"{source}/rejet_{reason}"] += 1
                    continue

                text = mask_pii(text)

                h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if h in seen_exact:
                    stats[f"{source}/rejet_doublon_exact"] += 1
                    continue
                seen_exact.add(h)

                m = minhash_of(text, num_perm)
                if lsh.query(m):
                    stats[f"{source}/rejet_quasi_doublon"] += 1
                    continue
                lsh.insert(str(len(kept)), m)

                kept.append({"text": text, "source": source})
                stats[f"{source}/docs_gardes"] += 1
                stats[f"{source}/octets_gardes"] += len(text.encode("utf-8"))

                if stats[f"{source}/docs_entree"] % 20000 == 0:
                    print(f"{source}: {stats[f'{source}/docs_entree']} lus, "
                          f"{stats[f'{source}/docs_gardes']} gardés ({time.time() - t0:.0f} s)")

    random.Random(seed).shuffle(kept)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(out_path + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(stats.items())), f, indent=2, ensure_ascii=False)

    print(f"\nTerminé en {time.time() - t0:.0f} s : {len(kept)} documents -> {out_path}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="nb max de documents par fichier (test)")
    p.add_argument("--normalize-typography", action="store_true")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    run(a.inputs, a.out, a.limit, a.normalize_typography, a.threshold, seed=a.seed)