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
 1. Vérifier le budget de paramètres 
python scripts/count_params.py

 2. Exécuter le pipeline de données
python data_pipeline/run_pipeline.py

 3. Lancer l'entraînement
python train.py

 4. Générer du texte
python generate.py

# Journal
https://github.com/KelyanDimitri-ops/mini-GPT/blob/main/docs/journal.md
