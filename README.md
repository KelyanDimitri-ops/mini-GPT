# mini-GPT

mini-GPT codé from scratch (autograd, BPE, transformer) + pipeline de données françaises. Piscine Lab, ECE Paris. (FR)

A GPT built from scratch: autograd, BPE tokenizer, decoder-only transformer, plus a French web-data pipeline. A lab pool project, ECE Paris.(EN)

# Avancement
- [x] Mini-autograd 
- [x] Bigramme par comptage
- [x] Tokenizer BPE
- [x] Transformer décodeur
- [x] Pipeline de données
- [ ] Ablations-datasheet-soutenance

# Structure du dépôt
 mini-GPT-main/
├── data_pipeline/         # Pipeline à 3 étapes (Clean, PII, MinHash)
├── gpt/                   # Code source du modèle Transformer
├── tokenizer/             # Tokenizer BPE personnalisé
├── Scripts/               # Scripts d'analyse (count_params, plot_metrics)
├── docs/                  # Visuels et graphique loss_curve.png
├── checkpoints/           # Checkpoints du modèle 
├── train.py               # Entraînement avec Cosine Decay & BPB
└── generate.py            # Script d'inférence interactif

# Lancer le projet

"""
pip install -r requirements.txt
python data_pipeline/collect.py --name HuggingFaceFW/fineweb-2 --config fra_Latn --out work/raw/fineweb_fr.jsonl --max-mb 700
python data_pipeline/collect.py --name wikimedia/wikipedia --config 20231101.fr --out work/raw/wiki_fr.jsonl --max-mb 150
python data_pipeline/collect.py --name manu/project_gutenberg --split fr --out work/raw/gutenberg_fr.jsonl --max-mb 150 --buffer-size 50
python data_pipeline/collect.py --name Nicolas-BZRD/DEBATS_opendata --out work/raw/debats.jsonl --max-mb 150 --buffer-size 200
python data_pipeline/run_pipeline.py --inputs work/raw/fineweb_fr.jsonl work/raw/wiki_fr.jsonl work/raw/gutenberg_fr.jsonl work/raw/debats.jsonl --max-mb-per-source 100 --out work/clean/dataset_v2.jsonl
python data_pipeline/decontaminate.py --data work/clean/dataset_v2.jsonl --eval data/validation.jsonl --out work/clean/dataset_v2b.jsonl
python -m ftb.contamination --inputs work/clean/dataset_v2b.jsonl

"""

# Journal
https://github.com/KelyanDimitri-ops/mini-GPT/blob/main/docs/journal.md
