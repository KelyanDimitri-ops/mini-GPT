# 📓 Journal de Bord - Mini-GPT

day1: 17/10/26 Choix du sujet : "Feed the beast", nourris ton propre GPT. Lecture du sujet et prise d'informations à ce propos.

day2: 18/10/26 Création du dépôt GitHub, mise en place de la structure des fichiers et connexion avec la plateforme Intelligence Lab.

day3: 29/10/26 Reprise du travail ; après une période assez lourde en examens, je reprends enfin le développement sur mon mini-GPT. Le programme était bien chargé, mais je remets le pied sur l'étrier. J'avais commencé à visionner les vidéos d'Andrej Karpathy (un peu dur à assimiler au début), mais je commence enfin à comprendre explicitement la structure du code et du moteur d'autograd.

day4: 30/10/26 Implémentation du fichier value.py qui joue le rôle de moteur scalaire, similaire à engine.py de Karpathy. Une étape assez technique mais surmontée ! Prochaine étape : basculer sur PyTorch pour valider les calculs de tenseurs et vérifier que je suis sur la bonne voie.

day5: 10/11/26 J'ai effectué le gros du travail ces derniers jours. Ça m'a pris énormément de temps (d'autant plus avec un souci matériel sur mon ordi), mais le modèle a enfin réalisé son premier entraînement complet ! J'ai pu tracer les courbes de Loss et le score BPB, et j'ai commencé à coder les scripts de nettoyage pour le dataset. On garde le rythme !

day6: 11/11/26 Finalisation du pipeline et calibration stricte sous les 10M de paramètres. Gros bond en avant aujourd'hui : le pipeline à 3 étapes (Nettoyage Unicode, anonymisation PII et déduplication MinHash) est totalement fonctionnel. Résultat top : près de 46 % de doublons éliminés sur le corpus de Molière ! J'ai également recalibré l'architecture (`n_embd = 360`, 6 couches, 6 têtes) pour verrouiller le modèle à **9,81M de paramètres**, respectant parfaitement la contrainte du projet. J'ai finalisé les scripts d'inférence (`generate.py`), de comptage (`count_params.py`) et de visualisations (`plot_metrics.py`), puis lancé l'entraînement final sur le dataset dédupliqué.