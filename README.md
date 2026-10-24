# Atelier 1 : Deep Learning avec PyTorch — Régression et Classification Multi-Classes

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DNN%20%2F%20MLP-ee4c2c)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GridSearchCV-orange)

**Auteur :** Inass ALILOU  
**Formation :** Master Sécurité IT & Big Data — Faculté des Sciences et Techniques de Tanger (FSTT)  
**Module :** Deep Learning  
**Encadrant :** Pr. Lotfi ELAACHAK

---

## 1. Vue d'ensemble du projet

Cet atelier met en œuvre des réseaux de neurones profonds pour deux problèmes d'apprentissage supervisé : la régression sur des cours boursiers historiques et la classification multi-classes de défaillances de machines. Il a pour objectifs pédagogiques de :

- approfondir l'utilisation de **PyTorch** pour définir, entraîner et évaluer des MLP ;
- implémenter une boucle d'entraînement avec `zero_grad()`, `backward()` et `step()`, ainsi que les modes `train()` et `eval()` ;
- rechercher des hyperparamètres avec Scikit-learn `GridSearchCV` et son intégration PyTorch, `skorch` ;
- choisir une validation adaptée à la structure des données : chronologique pour une série financière, stratifiée pour une classification ;
- suivre les courbes d'apprentissage et interpréter les mesures de performance et de généralisation ;
- étudier Batch Normalization, Dropout et la pénalisation L2 (`weight_decay`) comme techniques de régularisation.

Chaque partie est autonome et possède son propre fichier `requirements.txt`, ses données d'entrée et ses artefacts. Les chemins utilisés par les scripts sont relatifs au dossier du script et non au répertoire courant.

## 2. Arborescence du dépôt

```text
Atelier-1---DeepLearning/
├── README.md
├── part1_regression/
│   ├── data/
│   │   └── prices-split-adjusted.csv
│   ├── plots/
│   │   ├── close_price_aapl.png
│   │   └── price_correlation_aapl.png
│   ├── baseline_gridsearch.py
│   ├── best_params.json
│   ├── compare_models.py
│   ├── eda.py
│   ├── model_comparison.png
│   ├── regularized_model.py
│   ├── requirements.txt
│   ├── train_and_evaluate.py
│   ├── training_curves.png
│   └── training_history.csv
└── part2_classification/
    ├── data/
    │   └── predictive_maintenance.csv
    ├── plots/
    │   ├── class_distribution.png
    │   ├── classification_learning_curves.png
    │   ├── confusion_matrix_baseline.png
    │   ├── models_comparison.png
    │   └── sensors_correlation.png
    ├── balance_and_preprocess.py
    ├── baseline_gridsearch.py
    ├── best_params_classification.json
    ├── classification_training_history.csv
    ├── compare_models.py
    ├── eda.py
    ├── regularized_model.py
    ├── requirements.txt
    └── train_and_evaluate.py
```

Les CSV sous `data/` sont les données d'entrée nécessaires aux expériences. Les figures, historiques CSV et fichiers JSON sont des résultats d'analyse ou d'exécution ; certains sont générés ou remplacés lors du lancement des scripts.

## 3. Partie 1 : Régression sur les séries boursières — NYSE / AAPL

### Problématique et données

Le premier exercice estime le prix de clôture `close` d'Apple (`AAPL`) à partir de variables de marché de la même observation : `open`, `high`, `low` et `volume`. Le dataset NYSE est lu depuis `part1_regression/data/prices-split-adjusted.csv`. Le pipeline conserve les lignes AAPL, trie les observations par date si cette colonne est disponible, convertit les valeurs utiles en données numériques et écarte les observations invalides.

Cette tâche estime le cours de clôture associé aux indicateurs du même jour ; elle ne constitue pas, à elle seule, une prévision du cours futur.

### Analyse exploratoire

Le script `part1_regression/eda.py` présente un aperçu du dataset, les statistiques descriptives, les types des colonnes et les valeurs manquantes. Il génère :

- `part1_regression/plots/close_price_aapl.png`, l'évolution du cours de clôture ;
- `part1_regression/plots/price_correlation_aapl.png`, la corrélation entre prix OHLC et volume.

Les variables `open`, `high`, `low` et `close` sont des mesures liées du prix d'une même séance et présentent généralement une forte corrélation. Cette analyse descriptive renseigne sur les relations entre variables, mais ne remplace pas une évaluation sur des observations tenues à l'écart.

### MLP, optimisation et entraînement

Le modèle de référence est un MLP à quatre entrées, constitué de couches cachées linéaires suivies d'activations ReLU et d'une sortie scalaire. Il minimise la perte quadratique moyenne (`nn.MSELoss`). Les entrées sont standardisées : le scaler est ajusté sur le train et appliqué au test.

`part1_regression/baseline_gridsearch.py` utilise `skorch.NeuralNetRegressor` avec Scikit-learn `GridSearchCV`. La validation croisée `TimeSeriesSplit` respecte l'ordre chronologique au lieu de mélanger passé et futur. Le scaler est intégré au pipeline de recherche afin que chaque pli de validation ajuste sa transformation sur ses seules données d'entraînement. La meilleure configuration est sauvegardée dans `part1_regression/best_params.json`.

L'entraînement final utilise une séparation chronologique train/test et dure 100 époques. `training_history.csv` conserve la MSE et le score de détermination $R^2$ sur les deux partitions ; `training_curves.png` trace leur évolution. Une exécution consignée dans le dépôt atteint un $R^2$ de test supérieur à 0,98 à plusieurs époques. Les mesures fluctuent : l'interprétation doit prendre en compte la courbe complète, et non uniquement une valeur ponctuelle.

### Régularisation et comparaison

Le modèle régularisé ajoute `BatchNorm1d` aux couches cachées et un Dropout de probabilité 0,2. Son optimiseur peut utiliser un `weight_decay` de `1e-4`, équivalent à une pénalisation L2. `compare_models.py` entraîne le baseline et le modèle régularisé avec le même découpage chronologique, puis compare leurs pertes et leurs scores $R^2$ sur les ensembles d'entraînement et de test. La figure `model_comparison.png` permet d'inspecter les courbes.

L'écart de généralisation est calculé comme la perte de test moins la perte d'entraînement. Les mécanismes de régularisation visent à limiter le surapprentissage ; leur effet doit être vérifié à partir des résultats observés et n'est pas garanti par leur seule application.

## 4. Partie 2 : Classification multi-classes — Maintenance prédictive

### Problématique et classes

La seconde tâche prédit la colonne `Failure Type` à partir de la catégorie de produit `Type` et de cinq capteurs : température de l'air, température du procédé, vitesse de rotation, couple et usure de l'outil. Les six classes sont :

| Classe |
|---|
| `No Failure` |
| `Heat Dissipation Failure` |
| `Power Failure` |
| `Overstrain Failure` |
| `Tool Wear Failure` |
| `Random Failures` |

Le dataset requis est `part2_classification/data/predictive_maintenance.csv`. Les identifiants `UDI` et `Product ID`, ainsi que la cible binaire `Target`, sont exclus : l'étiquette retenue pour cet exercice est `Failure Type`.

### EDA, prétraitement et déséquilibre

Le script `part2_classification/eda.py` affiche les statistiques descriptives, les types de données, le nombre de valeurs manquantes et les effectifs des modalités. Il enregistre la distribution des classes dans `plots/class_distribution.png` et la corrélation des cinq capteurs dans `plots/sensors_correlation.png`.

Le dataset est fortement déséquilibré : la classe `No Failure` représente environ 96,5 % des lignes dans la distribution courante. Une Accuracy élevée pourrait donc être obtenue en prédisant principalement la classe majoritaire, sans reconnaître correctement les défaillances rares. Il faut également examiner la sensibilité (Recall), la précision et le F1-score de chaque classe, le Macro F1 et la matrice de confusion.

La fonction `get_data()` de `balance_and_preprocess.py` applique le protocole suivant :

1. encode `Type` selon `L=0`, `M=1`, `H=2` et `Failure Type` avec `LabelEncoder` ;
2. sépare les observations en train et test avec `test_size=0.2`, `random_state=42` et stratification ;
3. ajuste `StandardScaler` uniquement sur le train, puis transforme le test ;
4. applique `SMOTE(random_state=42, k_neighbors=3)` exclusivement aux données d'entraînement.

SMOTE génère des exemples synthétiques pour représenter davantage les classes rares pendant l'apprentissage. Son application après la séparation, et uniquement sur le train, évite de faire entrer des exemples synthétiques dérivés des observations de test dans l'entraînement. Le test conserve sa distribution réelle afin de rester représentatif de l'évaluation.

### Modélisation et évaluation

Le MLP de base reçoit six variables et produit six logits. Il est entraîné avec `nn.CrossEntropyLoss` et Adam. La recherche d'hyperparamètres de `baseline_gridsearch.py` s'appuie sur `skorch.NeuralNetClassifier`, `GridSearchCV`, le score `f1_macro` et une validation `StratifiedKFold` à trois plis. Pour limiter le coût de calcul, la recherche peut utiliser un sous-échantillon stratifié des données équilibrées. Les meilleurs hyperparamètres sont enregistrés dans `best_params_classification.json`.

`train_and_evaluate.py` entraîne le modèle pendant 35 époques avec une boucle native PyTorch et consigne à chaque époque la Cross-Entropy et l'Accuracy d'entraînement et de test. Le script affiche ensuite l'Accuracy globale et un rapport complet par classe (précision, Recall et F1-score) ; il génère aussi `plots/classification_learning_curves.png` et `plots/confusion_matrix_baseline.png`.

Une exécution présente dans l'historique du dépôt atteint environ 97,15 % d'Accuracy de test à la dernière époque. Cette valeur seule ne permet pas de conclure sur le rappel des classes rares : le rapport par classe et la matrice de confusion sont nécessaires. Les résultats dépendent du jeu de données, des paramètres et de l'exécution.

### Modèle régularisé et comparaison

`regularized_model.py` définit un MLP régularisé avec `BatchNorm1d` après chaque couche cachée, `Dropout(p=0.3)` après ReLU et un optimiseur Adam utilisant `weight_decay=1e-4`. Batch Normalization peut aider à stabiliser l'optimisation ; Dropout et weight decay visent à réduire le surapprentissage, notamment lorsque le modèle apprend à partir d'exemples synthétiques. Leur bénéfice doit être évalué empiriquement.

`compare_models.py` entraîne le baseline et le modèle régularisé pendant le même nombre d'époques, puis affiche les métriques suivantes :

| Mesure | Signification |
|---|---|
| Train Accuracy | Proportion de prédictions correctes sur le train équilibré |
| Test Accuracy | Proportion de prédictions correctes sur le test d'origine |
| Test Macro F1-Score | Moyenne non pondérée des F1-scores des six classes |
| Perte de généralisation | Cross-Entropy de test moins Cross-Entropy d'entraînement |

Les deux matrices de confusion sont enregistrées côte à côte dans `plots/models_comparison.png`.

## 5. Synthèse et compétences acquises

Les deux exercices illustrent des étapes complémentaires d'un pipeline de Deep Learning : analyse et préparation des données, choix du protocole de validation, définition de l'architecture, entraînement, suivi des métriques et interprétation des résultats. Le travail permet notamment de consolider :

- la maîtrise des boucles d'entraînement PyTorch, de la rétropropagation et des modes `train()` / `eval()` ;
- l'intégration de modèles PyTorch dans l'écosystème Scikit-learn via `skorch` et `GridSearchCV` ;
- l'importance de respecter la chronologie en séries temporelles et de préserver la stratification en classification ;
- la prévention de la fuite de données lors de la standardisation et du sur-échantillonnage SMOTE ;
- l'interprétation de métriques adaptées aux classes déséquilibrées, en particulier le Recall et le Macro F1 ;
- l'analyse du compromis biais-variance et l'évaluation de Batch Normalization, Dropout et Weight Decay.

## 6. Guide d'exécution

### 6.1 Cloner le dépôt et placer les données

```console
git clone https://github.com/inasalilou/Atelier-1---DeepLearning.git
cd Atelier-1---DeepLearning
```

Placez les fichiers de données dans les emplacements suivants :

- `part1_regression/data/prices-split-adjusted.csv`
- `part2_classification/data/predictive_maintenance.csv`

### 6.2 Créer et activer un environnement virtuel

Dans PowerShell sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sous macOS ou Linux :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 6.3 Installer les dépendances

Chaque partie fournit un fichier de dépendances. Depuis la racine du dépôt, installez les deux ensembles :

```console
python -m pip install --upgrade pip
python -m pip install -r part1_regression/requirements.txt
python -m pip install -r part2_classification/requirements.txt
```

### 6.4 Exécuter la Partie 1 — Régression

Depuis la racine du dépôt :

```console
cd part1_regression
python eda.py
python baseline_gridsearch.py
python train_and_evaluate.py
python compare_models.py
```

L'ordre recommandé est : exploration, recherche des hyperparamètres, entraînement du baseline, puis comparaison avec le modèle régularisé. Les scripts lisent et produisent leurs fichiers dans `part1_regression/`.

### 6.5 Exécuter la Partie 2 — Classification

Ouvrez un nouveau terminal à la racine du dépôt, ou revenez à celle-ci après la Partie 1 avec `cd ..`, puis lancez :

```console
cd part2_classification
python eda.py
python balance_and_preprocess.py
python baseline_gridsearch.py
python train_and_evaluate.py
python compare_models.py
```

`balance_and_preprocess.py` peut être exécuté indépendamment pour vérifier les dimensions des partitions et les effectifs après SMOTE. Les autres scripts suivent l'ordre : exploration, vérification du prétraitement, recherche d'hyperparamètres, entraînement et comparaison. Les figures, paramètres et historiques sont enregistrés dans `part2_classification/` et ses sous-dossiers.
