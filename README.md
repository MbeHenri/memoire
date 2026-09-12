# Composition des méthodes HRMS et RMN par apprentissage ensembliste pour l'identification des réseaux moléculaires

> *English summary — This repository contains a Master's thesis (Université de Yaoundé I) proposing
> to combine mass spectrometry (HRMS/GNPS) and NMR (MADByTE) molecular networks through multi-view
> clustering (MCLES, MVGL) and ensemble learning (Bagging, Boosting), evaluated on the MSRC-V1
> benchmark.*

## Présentation

Pour identifier des composés à partir de réseaux moléculaires, les chercheurs s'appuient
généralement sur la spectrométrie de masse haute résolution (HRMS, via l'outil **GNPS**) ou
sur la résonance magnétique nucléaire (RMN, via l'outil **MADByTE**), mais rarement sur les
deux à la fois. Ce mémoire propose une méthode qui exploite la complémentarité de ces deux
sources en combinant les réseaux issus de GNPS et de MADByTE, puis en appliquant :

- une extraction de clusters/voisinages par **DBSCAN** sur les réseaux moléculaires,
- des algorithmes de **clustering multi-vues** (MCLES, MVGL) réimplémentés from scratch,
- des techniques d'**apprentissage ensembliste** (Bagging, Boosting) pour combiner les vues.

La meilleure configuration testée (Bagging + MCLES comme apprenant faible, sur le jeu de
données benchmark MSRC-V1) atteint environ **80,1 % d'accuracy, 74,3 % de NMI et 81,1 % de
pureté**.

**Auteur :** MBE MBE MINDJANA LOIC HENRI (Matricule 18T2603)
**Encadreurs :** Pr. WAFFO TEGUO Pierre, Pr. MELATAGIA YONTA Paulin
**Établissement :** Université de Yaoundé I, Faculté des Sciences, Département d'Informatique — Master Recherche, option Science des données (2022/2023)

Le mémoire complet (PDF) est disponible dans [`docs/memoire_modifie-1.pdf`](docs/memoire_modifie-1.pdf).

## Structure du dépôt

```txt
memoire/
├── docs/                    # Le mémoire lui-même
│   ├── memoire_modifie-1.pdf    # Version PDF compilée
│   └── latex/                   # Sources LaTeX (chapitres, images, bibliographie, articles résumés)
│
├── datasets/                # Jeux de données utilisés
│   ├── msrc-v1/                 # Benchmark MSRC-V1 (clustering multi-vues)
│   ├── test/rmn/                 # Données RMN de test (spectres HSQC/TOCSY d'antibiotiques)
│   └── load.py, utils.py, exploration.ipynb
│
├── utils/                   # Code des algorithmes
│   ├── mcles/scratch/            # Implémentation from-scratch de MCLES
│   ├── mvgl/scratch/             # Implémentation from-scratch de MVGL
│   ├── ClusterEnsembles/          # Bibliothèque de combinaison d'ensembles de clusters
│   ├── madbyte/                   # Copie vendorisée de l'outil MADByTE (traitement RMN)
│   ├── gnps/                      # Traitement des données GNPS (HRMS)
│   ├── clusters_network.py        # Extraction de clusters à partir des réseaux moléculaires
│   └── vote.py                    # Vote majoritaire pour le consensus de clustering
│
├── results/                 # Résultats des expérimentations (CSV, graphiques ACC/NMI/pureté)
│
├── bagging.py / bagging.ipynb      # Méthode ensembliste par Bagging
├── bootsing.py                     # Méthode ensembliste par Boosting
├── metriques.py                    # Métriques d'évaluation (ACC, NMI, pureté, ...)
├── process_madbyte_gnps.py         # Pipeline de traitement GNPS + MADByTE
├── analyse.ipynb                   # Notebook d'analyse
├── experiments_complet.ipynb       # Notebook des expérimentations complètes
├── experiments_analyse.ipynb       # Notebook d'analyse des expérimentations
└── requirements.txt                # Dépendances Python
```
