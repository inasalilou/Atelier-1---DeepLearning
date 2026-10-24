<<<<<<< HEAD
# Laboratoire de régression sur les cours de bourse

Ce projet explore et compare des réseaux de neurones pour estimer le prix de clôture (`close`) d'Apple (ticker `AAPL`) à partir de données boursières historiques. Il comprend une exploration des données, une recherche d'hyperparamètres, l'entraînement d'un modèle de référence et une comparaison avec un modèle régularisé.

## Contenu du dépôt

| Fichier ou dossier | Rôle |
| --- | --- |
| `data/prices-split-adjusted.csv` | Données historiques contenant notamment la date, le ticker, les prix OHLC et le volume. |
| `eda.py` | Produit des statistiques descriptives et des graphiques exploratoires pour AAPL. |
| `baseline_gridsearch.py` | Charge et prépare les données, définit le MLP de référence et recherche ses hyperparamètres avec validation chronologique. |
| `best_params.json` | Meilleure configuration enregistrée par la recherche d'hyperparamètres. |
| `train_and_evaluate.py` | Entraîne le MLP de référence, enregistre les métriques par époque et trace les courbes d'apprentissage. |
| `regularized_model.py` | Définit le MLP régularisé et son optimiseur. |
| `compare_models.py` | Entraîne et compare les modèles de référence et régularisé. |
| `plots/` | Graphiques d'analyse exploratoire. |
| `training_history.csv` | Métriques d'entraînement et de test du dernier entraînement de référence enregistré. |
| `training_curves.png` | Courbes de perte MSE et de score R² du dernier entraînement de référence. |
| `model_comparison.png` | Comparaison des courbes des deux modèles. |
| `requirements.txt` | Dépendances Python du projet. |

## Données et préparation

Le CSV fourni contient 851 264 lignes et les colonnes `date`, `symbol`, `open`, `close`, `low`, `high` et `volume`. Parmi elles, 1 762 concernent AAPL. Le pipeline de régression :

1. filtre les lignes AAPL et trie les observations chronologiquement quand une colonne de date est disponible ;
2. convertit les variables en valeurs numériques et élimine les lignes incomplètes ;
3. utilise `open`, `high`, `low` et `volume` comme variables d'entrée, et `close` comme cible ;
4. sépare les données dans l'ordre chronologique : 80 % pour l'entraînement et 20 % pour le test ;
5. standardise les variables d'entrée en ajustant le scaler uniquement sur les données d'entraînement.

L'exploration (`eda.py`) affiche un aperçu, les statistiques descriptives, les types de colonnes et le nombre de valeurs manquantes. Elle génère aussi une courbe du prix de clôture et une matrice de corrélation des variables de prix et de volume.

## Modèles et entraînement

Le modèle de référence est un perceptron multicouche (MLP) avec quatre variables d'entrée, deux couches cachées suivies de ReLU, et une sortie scalaire. Il est entraîné avec la perte MSE. Adam et SGD sont pris en charge par le code d'entraînement.

`baseline_gridsearch.py` effectue une recherche avec `GridSearchCV`, un `TimeSeriesSplit` à trois partitions et une pipeline comprenant le `StandardScaler` et le réseau skorch. La grille actuelle teste :

- taux d'apprentissage : `0.01` et `0.001` ;
- dimension cachée : `32` et `64` ;
- deux couches cachées ;
- batch de 32 observations ;
- optimiseur Adam ;
- jusqu'à 15 époques par configuration.

La sélection d'hyperparamètres est réalisée sur la portion d'entraînement, sans mélanger l'ordre temporel. La meilleure configuration est écrite dans `best_params.json`.

L'entraînement final dure 100 époques. Il enregistre à chaque époque la perte MSE et le score R² sur les ensembles d'entraînement et de test dans `training_history.csv`, puis crée `training_curves.png`.

Le modèle régularisé ajoute une normalisation par lot (`BatchNorm1d`), un dropout de 20 % dans les couches cachées et un weight decay de `1e-4`. `compare_models.py` réutilise les meilleurs hyperparamètres trouvés, entraîne les deux modèles avec la même séparation temporelle et trace leurs pertes et scores R² dans `model_comparison.png`.

## Installation

Depuis la racine du projet, dans PowerShell :
=======
# Atelier 1 : Deep Learning avec PyTorch — Régression et Classification Multi-Classes

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DNN%20%2F%20MLP-ee4c2c)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GridSearchCV-orange)

- **Auteur :** Inass ALILOU
- **Formation :** Master Sécurité IT & Big Data (MBD) — Faculté des Sciences et Techniques de Tanger (FSTT)
- **Module :** Deep Learning
- **Supervisé par :** Pr. Lotfi ELAACHAK

---

## 1. Vue d'ensemble du projet

Cet atelier met en pratique la conception, l'entraînement et l'évaluation de réseaux de neurones avec **PyTorch**, sur deux problèmes supervisés complémentaires : la régression d'une série financière et la classification multi-classes de défaillances industrielles.

Les objectifs pédagogiques sont les suivants :

- approfondir l'utilisation de PyTorch pour les tâches de régression et de classification ;
- concevoir des perceptrons multi-couches (MLP) et contrôler leur boucle d'entraînement ;
- rechercher des hyperparamètres avec `GridSearchCV`, `skorch` et des stratégies de validation adaptées au problème ;
- analyser la convergence et la généralisation à l'aide de courbes de perte et de métriques propres à chaque tâche ;
- comparer des modèles de base avec des modèles régularisés, notamment par Dropout, Batch Normalization et pénalisation L2 (`weight_decay`).

Les scripts créent leurs graphiques et historiques dans le dossier de la partie correspondante. Les données brutes ne sont pas incluses dans les scripts : chaque CSV doit être placé dans le sous-dossier `data/` indiqué ci-dessous.

## 2. Structure de l'arborescence du dépôt

```text
Atelier-1---DeepLearning/
├── README.md
├── part1_regression/
│   ├── data/
│   │   └── prices-split-adjusted.csv          (donnée locale)
│   ├── plots/
│   │   ├── close_price_aapl.png
│   │   └── price_correlation_aapl.png
│   ├── baseline_gridsearch.py
│   ├── best_params.json                       (résultat de GridSearchCV)
│   ├── compare_models.py
│   ├── eda.py
│   ├── model_comparison.png                   (généré par comparaison)
│   ├── regularized_model.py
│   ├── requirements.txt
│   ├── train_and_evaluate.py
│   ├── training_curves.png                    (généré par entraînement)
│   └── training_history.csv                   (généré par entraînement)
└── part2_classification/
    ├── data/
    │   └── predictive_maintenance.csv          (donnée locale)
    ├── plots/
    │   ├── class_distribution.png
    │   ├── classification_learning_curves.png  (généré par entraînement)
    │   ├── confusion_matrix_baseline.png       (généré par entraînement)
    │   ├── models_comparison.png                (généré par comparaison)
    │   └── sensors_correlation.png
    ├── balance_and_preprocess.py
    ├── baseline_gridsearch.py
    ├── best_params_classification.json         (résultat de GridSearchCV)
    ├── classification_training_history.csv     (généré par entraînement)
    ├── compare_models.py
    ├── eda.py
    ├── regularized_model.py
    ├── requirements.txt
    └── train_and_evaluate.py
```

Les fichiers CSV sont des données d'entrée locales ; ils ne sont pas des scripts du projet et doivent être récupérés séparément. Les images, historiques CSV et fichiers JSON de paramètres sont des artefacts d'analyse ou d'exécution.

## 3. Partie 1 : Régression sur les séries financières (dataset NYSE)

### Problématique et données

La première partie prédit le cours de clôture (`close`) d'Apple (`AAPL`) à partir des indicateurs journaliers `open`, `high`, `low` et `volume` du dataset Kaggle NYSE. Le fichier attendu est `part1_regression/data/prices-split-adjusted.csv`. Le chargement filtre le symbole `AAPL`, valide les colonnes utiles, ordonne les observations par date et élimine les valeurs non valides.

### Analyse exploratoire

Le script [`part1_regression/eda.py`](part1_regression/eda.py) affiche un aperçu, des statistiques descriptives, les types de colonnes et les valeurs manquantes. Il produit :

- `plots/close_price_aapl.png`, l'évolution temporelle du cours de clôture ;
- `plots/price_correlation_aapl.png`, la matrice de corrélation des variables `open`, `high`, `low`, `volume` et `close`.

Les prix OHLC décrivent des mesures de marché étroitement liées. La matrice permet d'observer la dépendance linéaire entre les valeurs d'ouverture, hautes, basses et de clôture ; elle ne prouve toutefois pas à elle seule la qualité prédictive du modèle.

### Modélisation et optimisation

Le modèle de base est un MLP configurable en largeur et en nombre de couches cachées. Il prend quatre indicateurs en entrée, utilise des activations ReLU et prédit une valeur continue. La fonction de coût est `nn.MSELoss`.

La recherche d'hyperparamètres emploie `skorch.NeuralNetRegressor` et `GridSearchCV`, avec `TimeSeriesSplit` afin de préserver l'ordre temporel des observations pendant la validation croisée. Un `StandardScaler` est intégré au pipeline et ajusté séparément dans les partitions d'entraînement de chaque pli. Les paramètres retenus sont enregistrés dans `best_params.json`.

L'entraînement final sépare les données chronologiquement (sans mélange), ajuste le scaler sur le train uniquement et évalue le modèle à chaque époque. Le fichier `training_history.csv` consigne la MSE et le score $R^2$ d'entraînement et de test ; `training_curves.png` les visualise. Sur une exécution enregistrée dans le dépôt, le score $R^2$ de test atteint environ **0,966 à la troisième époque**. La courbe est nécessaire pour interpréter la convergence : le score de test peut varier d'une époque à l'autre, il ne faut donc pas assimiler un score ponctuel à une généralisation stable.

### Régularisation et comparaison

Le modèle régularisé ajoute `BatchNorm1d` et Dropout (probabilité 0,2 dans l'implémentation) aux couches cachées. L'optimiseur peut appliquer une pénalisation L2 via `weight_decay=1e-4`. Le script de comparaison entraîne les deux modèles sur le même découpage temporel et affiche les MSE, les scores $R^2$ ainsi que l'écart de perte de généralisation, calculé comme `test_loss - train_loss`. L'effet de la régularisation doit être interprété à partir de ces mesures et des courbes générées ; une réduction de l'écart n'est pas garantie par la seule présence de ces techniques.

## 4. Partie 2 : Classification multi-classes (maintenance prédictive)

### Problématique et données

Cette partie apprend à distinguer six modalités de `Failure Type` :

1. `No Failure`
2. `Heat Dissipation Failure`
3. `Power Failure`
4. `Overstrain Failure`
5. `Tool Wear Failure`
6. `Random Failures`

Les entrées comprennent la catégorie de produit `Type` et cinq mesures physiques : température de l'air, température du procédé, vitesse de rotation, couple et usure de l'outil. Le fichier attendu est `part2_classification/data/predictive_maintenance.csv`. Les identifiants `UDI`, `Product ID` et la cible binaire `Target` sont écartés : l'objectif est la prédiction multi-classes de `Failure Type`.

### Déséquilibre des classes et prétraitement

La classe `No Failure` représente la grande majorité des observations (environ 96,5 % dans la distribution habituelle de ce dataset). Une Accuracy élevée peut donc masquer l'incapacité à reconnaître les défaillances rares. Il est important de compléter cette mesure par le rappel (sensibilité), la précision, le F1-score par classe, le Macro F1 et une matrice de confusion.

La fonction `get_data()` de [`part2_classification/balance_and_preprocess.py`](part2_classification/balance_and_preprocess.py) :

- mappe `Type` selon `L=0`, `M=1`, `H=2` et encode `Failure Type` avec `LabelEncoder` ;
- crée une partition train/test stratifiée de 80 % / 20 % avec la graine 42 ;
- ajuste `StandardScaler` uniquement sur le train et transforme le test avec ce scaler ;
- applique `SMOTE(random_state=42, k_neighbors=3)` au train seulement. Le jeu de test garde ainsi sa distribution naturelle et n'est pas enrichi avec des exemples synthétiques.

Le sur-échantillonnage augmente la représentation des classes minoritaires pendant l'apprentissage ; il ne crée pas de nouvelles observations réelles. Le GridSearchCV utilise ensuite une validation `StratifiedKFold` sur les données d'entraînement équilibrées.

Le script [`part2_classification/eda.py`](part2_classification/eda.py) affiche les statistiques descriptives, les types, les valeurs nulles et les effectifs par classe. Il enregistre `plots/class_distribution.png` et `plots/sensors_correlation.png`.

### Modélisation et évaluation

Le MLP de base utilise six entrées et six logits de sortie, avec `nn.CrossEntropyLoss` et Adam. La sélection d'hyperparamètres s'appuie sur `skorch.NeuralNetClassifier`, `GridSearchCV`, une validation `StratifiedKFold` à trois plis et le score `f1_macro`. Les meilleurs paramètres sont sauvegardés dans `best_params_classification.json`.

L'entraînement natif PyTorch dure 35 époques. À chaque époque, le script conserve la perte Cross-Entropy et l'Accuracy sur les données d'entraînement et de test. Il affiche ensuite l'Accuracy globale et le rapport de classification complet (précision, rappel/sensibilité et F1-score par classe), puis enregistre `classification_learning_curves.png` et `confusion_matrix_baseline.png`.

Une exécution enregistrée dans `classification_training_history.csv` donne une Accuracy de test d'environ **97,15 % à la dernière époque**. Ce résultat ne suffit pas pour juger la détection des pannes : le rapport de classification et la matrice de confusion générés par le script sont indispensables, en particulier pour les classes minoritaires. Les scores varient selon les données et les paramètres effectivement utilisés.

### Impact de la régularisation

Le modèle régularisé ajoute `BatchNorm1d` après les couches linéaires cachées, `Dropout(0.3)` après les activations ReLU et un `weight_decay=1e-4` sur Adam. La normalisation des activations peut stabiliser l'optimisation ; le Dropout et la pénalisation L2 visent à limiter le surapprentissage. Le script `compare_models.py` entraîne les modèles de base et régularisé pendant le même nombre d'époques, puis compare :

| Mesure | Interprétation |
|---|---|
| Train Accuracy | Capacité à classer les observations vues à l'entraînement |
| Test Accuracy | Proportion de prédictions correctes sur le test |
| Test Macro F1-Score | Moyenne non pondérée du F1-score des six classes |
| Perte de généralisation | Différence entre la Cross-Entropy de test et celle d'entraînement |

Les matrices de confusion côte à côte sont enregistrées dans `plots/models_comparison.png`. Les résultats comparatifs réels sont imprimés par le script ; BatchNorm, Dropout et Weight Decay sont des mécanismes à évaluer, et non une garantie automatique d'une meilleure performance.

## 5. Synthèse et bilan des apprentissages

Cet atelier met en œuvre les principales étapes d'un projet de Deep Learning supervisé : préparation des données, définition d'un réseau, entraînement, suivi des métriques et comparaison de modèles. Il permet notamment de pratiquer :

- les boucles PyTorch (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`) et les modes `model.train()` / `model.eval()` ;
- l'interopérabilité entre PyTorch, Scikit-learn et `skorch` pour la recherche d'hyperparamètres ;
- les découpages et validations adaptés à la structure des données : chronologie pour la série financière, stratification pour les classes ;
- l'importance du traitement du déséquilibre, de SMOTE appliqué au train et de métriques comme le Macro F1 et le rappel par classe ;
- l'analyse du compromis biais-variance et la comparaison des effets de Batch Normalization, Dropout et Weight Decay sur la généralisation.

Les résultats doivent toujours être interprétés à la lumière du protocole expérimental, de la distribution des classes et des artefacts réellement produits lors de l'exécution.

## 6. Guide d'exécution

### 6.1 Récupérer le projet et préparer les données

Depuis la page GitHub du dépôt, ouvrez **Code → HTTPS** et exécutez la commande `git clone` fournie par GitHub pour ce dépôt. Placez-vous ensuite à la racine du projet :

```console
cd Atelier-1---DeepLearning
```

Récupérez séparément les deux fichiers de données et placez-les aux emplacements suivants :

- `part1_regression/data/prices-split-adjusted.csv`
- `part2_classification/data/predictive_maintenance.csv`

### 6.2 Créer et activer un environnement virtuel

Dans PowerShell sous Windows :
>>>>>>> 6ee10cfb (feat: complete deep learning lab 1 (regression & classification))

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
<<<<<<< HEAD
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le projet utilise notamment PyTorch, scikit-learn, skorch, pandas, NumPy, Matplotlib et Seaborn. PyTorch sélectionne automatiquement CUDA si disponible, sinon le processeur.

## Exécution

Exécuter les commandes depuis la racine du projet :

```powershell
# Exploration des données et génération des graphiques dans plots/
python eda.py

# Recherche d'hyperparamètres et mise à jour de best_params.json
python baseline_gridsearch.py

# Entraînement du modèle de référence et génération des courbes et métriques
python train_and_evaluate.py

# Entraînement et comparaison des modèles de référence et régularisé
python compare_models.py
```

Les scripts d'entraînement lisent `best_params.json`. Si ce fichier est absent, `train_and_evaluate.py` et `compare_models.py` lancent automatiquement la recherche d'hyperparamètres. Pour relancer cette recherche explicitement, exécuter `baseline_gridsearch.py`.

## Résultats déjà enregistrés

La configuration sélectionnée actuellement est :

```json
{
  "lr": 0.01,
  "optimizer": "Adam",
  "hidden_dim": 64,
  "num_layers": 2,
  "batch_size": 32
}
```

`training_history.csv` contient 100 époques. À la dernière époque enregistrée, les métriques sont :

| Mesure | Entraînement | Test |
| --- | ---: | ---: |
| MSE | 1.4206 | 2.6996 |
| R² | 0.9981 | 0.9576 |

La plus faible MSE de test enregistrée dans ce fichier est d'environ `0.4085` (ligne d'époque `97`, soit la 98e époque si l'on compte à partir de 1). Les performances évoluent au fil des époques : il ne faut donc pas confondre le meilleur score observé pendant l'entraînement avec la métrique de la dernière époque.

## Interprétation et limites

- Le modèle estime le prix de clôture à partir de variables de la même observation (`open`, `high`, `low`, `volume`) ; il ne s'agit pas d'une prévision du cours futur.
- Le jeu de test est évalué à chaque époque pour tracer les courbes. Consulter ces résultats à répétition pour choisir un modèle ou une époque peut influencer les décisions prises sur le test ; une évaluation finale réellement indépendante devrait utiliser une période de test réservée et non consultée pendant le développement.
- Les métriques dépendent du jeu de données et de la période ; elles ne constituent ni une garantie de performance future ni un conseil financier.
- Les scripts enregistrent les paramètres et les métriques, mais ne sauvegardent pas les poids des modèles entraînés.
=======
```

Sous macOS ou Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.3 Installer les dépendances

Chaque partie possède son propre fichier de dépendances. Depuis la racine, installez les deux ensembles :

```console
python -m pip install --upgrade pip
python -m pip install -r part1_regression/requirements.txt
python -m pip install -r part2_classification/requirements.txt
```

### 6.4 Exécuter la Partie 1 — Régression

Depuis la racine du projet :

```console
cd part1_regression
python eda.py
python baseline_gridsearch.py
python train_and_evaluate.py
python compare_models.py
```

La recherche écrit `best_params.json`. L'entraînement produit l'historique CSV et les courbes de perte et de $R^2$ ; la comparaison produit `model_comparison.png`.

### 6.5 Exécuter la Partie 2 — Classification

Après l'exécution de la Partie 1, le terminal se trouve dans `part1_regression/`. Revenez à la racine, puis entrez dans le dossier de classification :

```console
cd ..
cd part2_classification
python eda.py
python balance_and_preprocess.py
python baseline_gridsearch.py
python train_and_evaluate.py
python compare_models.py
```

`balance_and_preprocess.py` peut être lancé seul pour vérifier les dimensions et les effectifs après préparation. La recherche crée `best_params_classification.json`. L'entraînement et la comparaison enregistrent leurs historiques et graphiques dans le dossier de la partie, notamment sous `plots/`.

Pour lancer uniquement la Partie 2 depuis la racine du projet, exécutez d'abord `cd part2_classification`.

L'ordre ci-dessus exécute d'abord l'analyse exploratoire, puis la recherche d'hyperparamètres, l'entraînement et enfin la comparaison. La recherche d'hyperparamètres est la plus coûteuse en calcul ; elle peut être omise si le fichier JSON des meilleurs paramètres correspondant à la partie est déjà disponible.
>>>>>>> 6ee10cfb (feat: complete deep learning lab 1 (regression & classification))
