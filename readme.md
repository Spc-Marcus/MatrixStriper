# MatrixStriper

## Démarrage rapide

### Installation

Assurez-vous d'avoir Python 3.8+ et les dépendances installées (voir `env_Windows.yaml` ou `setup.py`).

```bash
# (Optionnel) Créez un environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows

# Installez les dépendances
pip install -r requirements.txt  # ou utilisez conda avec env_Windows.yaml

# Installez le module en mode développement (recommandé pour le développement)
pip install -e .
```

### Lancement en ligne de commande

Pour lancer le pipeline principal sur une matrice CSV :

```bash
python -m src.MatrixStriper <input_csv> <output_txt> <output_csv> [options]
```

- `<input_csv>` : Chemin du fichier CSV d'entrée (matrice binaire)
- `<output_txt>` : Chemin du fichier texte pour les métriques
- `<output_csv>` : Chemin du fichier CSV de sortie (matrice réduite)

**Options principales :**
- `--largest_only` : Ne calcul que la plus grande sous-matrice dense
- `--min_col_quality N` : Qualité minimale des colonnes (défaut : 3)
- `--min_row_quality N` : Qualité minimale des lignes (défaut : 5)
- `--error_rate X` : Taux d'erreur toléré (défaut : 0.025)
- `--distance_thresh X` : Seuil de fusion des clusters (défaut : 0.01)
- `--certitude X` : Seuil de certitude pour la binarisation (défaut : 0.3)
- `--debug 0|1|2` : Niveau de log (0=WARNING, 1=INFO, 2=DEBUG)

**Exemple :**
```bash
python -m src.MatrixStriper data/mat_test.csv res.txt res.csv --min_col_quality 3 --min_row_quality 5 --error_rate 0.025
```

---

MatrixStriper est un module Python destiné à l'identification, la segmentation et le compactage de motifs binaires homogènes ("strips") dans des matrices de données, typiquement utilisées en bioinformatique (par exemple, matrices de variants, lectures de séquençage, etc.).

## Objectif du module

L'objectif principal de MatrixStriper est de fournir un pipeline automatisé pour :
- Prétraiter des matrices binaires afin d'identifier des régions homogènes et ambiguës.
- Appliquer des méthodes de clustering pour regrouper les colonnes et les lignes selon leur similarité.
- Extraire et sauvegarder les motifs pertinents pour des analyses ultérieures.

## Fonctionnalités principales

- Clustering hiérarchique des colonnes et des lignes.
- Détection automatique de "strips" (régions homogènes de 0 et/ou de 1).
- Gestion des régions ambiguës ou inhomogènes.
- Pipeline complet de lecture, traitement et écriture de matrices.
- Outils utilitaires pour la manipulation de matrices CSV.

## Utilisation typique

Le module peut être utilisé en ligne de commande ou intégré dans des scripts Python pour :
- Prétraiter des matrices binaires issues de données expérimentales.
- Identifier des motifs structurés dans des jeux de données volumineux.
- Générer des clusters finaux pour des analyses de souches, de variants, etc.

# Table des matières

- [Fonctions de pre_processing.py](#fonctions-de-pre_processingpy)
- [Fonctions de pipeline.py](#fonctions-de-pipelinepy)
- [Fonctions de post_processing.py](#fonctions-de-post_processingpy)
- [Fonctions de clustering.py](#fonctions-de-clusteringpy)
- [Fonctions utilitaires (utils.py)](#fonctions-utilitaires-utilspy)

---

# Fonctions de `pre_processing.py`

## `pre_processing`

La fonction `pre_processing` a pour objectif de prétraiter une matrice binaire (ou booléenne) représentant des lectures (lignes) et des colonnes (par exemple, des positions ou des features), afin d'identifier des régions homogènes appelées "strips" et de distinguer les colonnes ambiguës ou inhomogènes.

## Fonctionnement général

1. **Clustering hiérarchique des colonnes** :
   - On calcule la matrice de distances de Hamming entre les colonnes de la matrice d'entrée.
   - On applique un clustering hiérarchique (méthode linkage "complete") pour regrouper les colonnes similaires.
   - On découpe l'arbre de clustering pour obtenir des groupes de colonnes suffisamment divergents (selon le paramètre `certitude`).

2. **Filtrage des groupes** :
   - On ne conserve que les groupes de colonnes contenant au moins `min_col_quality` colonnes.

3. **Identification des strips** :
   - Pour chaque groupe, on applique un clustering hiérarchique sur les lignes (lectures) pour séparer les lignes en deux clusters.
   - On vérifie si l'un des clusters contient principalement des 0 et l'autre principalement des 1 (avec une tolérance définie par `error_rate`).
   - Si c'est le cas, le groupe est identifié comme un "strip" (région homogène).
   - Sinon, les colonnes du groupe sont considérées comme ambiguës.

4. **Gestion des colonnes non utilisées** :
   - Toute colonne n'appartenant à aucun strip ou groupe ambigu est ajoutée aux colonnes ambiguës.
   - Si aucun strip n'est trouvé, toutes les colonnes sont considérées comme ambiguës.

## Paramètres

- `input_matrix` : `np.ndarray`
  - Matrice d'entrée de forme (m, n), où m est le nombre de lectures et n le nombre de colonnes.
- `min_col_quality` : `int` (défaut : 5)
  - Nombre minimal de colonnes pour qu'un groupe soit considéré.
- `default` : `int` (défaut : 0)
  - Valeur par défaut pour les entrées incertaines (non utilisé directement dans la logique principale).
- `certitude` : `float` (défaut : 0.2)
  - Seuil de divergence pour le découpage des clusters de colonnes.
- `error_rate` : `float` (défaut : 0.025)
  - Tolérance d'erreur pour l'identification des strips (proportion d'erreurs acceptée dans un cluster homogène) (doit être < 0.5).

## Valeurs de retour

- `inhomogenious_regions` : `list[int]`
  - Liste des indices de colonnes identifiées comme inhomogènes ou ambiguës.
- `steps` : `list[tuple]`
  - Liste de tuples décrivant les strips identifiés. Chaque tuple contient :
    - Les indices des lignes du premier cluster
    - Les indices des lignes du second cluster
    - Les indices des colonnes du strip

## Exemple d'utilisation

```python
import numpy as np
from src.MatrixStriper.pre_processing import pre_processing

mat = np.random.randint(0, 2, (100, 20))
ambiguous_cols, strips = pre_processing(mat)
print("Colonnes ambiguës :", ambiguous_cols)
print("Strips identifiés :", strips)
```

## Remarques
- Cette fonction est utile pour détecter des motifs binaires homogènes dans des matrices de données.
- Les paramètres `certitude` et `error_rate` permettent d'ajuster la sensibilité de la détection des strips.

## `hamming_distance_matrix`
Calcule la matrice de distances de Hamming entre les colonnes d'une matrice binaire.

**Logique :**
- La distance de Hamming entre deux colonnes est le nombre de positions où elles diffèrent, divisé par la longueur de la colonne (i.e., la proportion de bits différents).
- La fonction convertit d'abord la matrice en binaire (0/1), puis calcule la distance de Hamming pour chaque paire de colonnes à l'aide de `pdist` de scipy, et retourne la matrice carrée des distances.
- Cela permet de quantifier la similarité entre colonnes pour le clustering.

## `is_strip`
Détermine si un groupe de colonnes forme un "strip" homogène en appliquant un clustering hiérarchique sur les lignes.

**Logique :**
- On extrait la sous-matrice correspondant aux colonnes à tester.
- Si la sous-matrice est presque entièrement composée de 0 (ou de 1), on considère qu'il s'agit d'un strip homogène (cas trivial).
- Sinon, on applique un clustering hiérarchique (AgglomerativeClustering) sur les lignes (lectures) pour les séparer en deux groupes.
- On calcule, pour chaque cluster, la proportion de 1 (ou de 0). Si un cluster est presque tout à 0 et l'autre presque tout à 1 (avec une tolérance définie par `error_rate`), alors le groupe de colonnes est considéré comme un strip.
- Sinon, il est considéré comme ambigu ou inhomogène.

# Fonctions de `post_processing.py`

## `post_processing`

La fonction `post_processing` prend le résultat du biclustering (steps) et la matrice binaire d'origine pour produire :
- les clusters finaux de reads (lignes),
- une matrice réduite représentant le profil consensus de chaque cluster sur les stripes (strips) identifiés,
- la liste des reads orphelins (non clusterisés à la fin),
- la liste des colonnes non utilisées dans les steps (colonnes non "stripes").

### Fonctionnement général

1. **Initialisation** :
   - Tous les reads sont placés dans un seul cluster initial.

2. **Application des étapes de biclustering** :
   - À chaque step, les clusters sont séparés selon les indices fournis (reads1, reads0, cols).
   - Les colonnes utilisées dans les steps sont marquées comme "stripes".

3. **Filtrage des petits clusters** :
   - Les clusters de taille inférieure à `min_reads_per_cluster` sont considérés comme orphelins (sauf si ce paramètre vaut `None`).

4. **Réaffectation des reads orphelins** :
   - Les reads orphelins sont réaffectés au cluster le plus proche (distance de Hamming sur les positions des clusters).

5. **Fusion des clusters similaires** :
   - Les clusters dont les profils consensus sont trop proches (distance de Hamming < `distance_thresh`) sont fusionnés.

6. **Construction de la matrice réduite** :
   - Pour chaque cluster, on calcule un profil consensus (moyenne arrondie) sur les colonnes de chaque step (une colonne par step, moyenne sur les colonnes de ce step).
   - La matrice réduite a donc pour shape `(nb_clusters, nb_steps)`.

7. **Exclusion des reads orphelins finaux** :
   - Les reads qui ne sont dans aucun cluster à la fin sont ignorés dans la matrice réduite et la liste des clusters, et sont listés séparément.

8. **Détection des colonnes inutilisées** :
   - Les colonnes de la matrice d'origine qui ne sont utilisées dans aucun step sont listées séparément.

### Paramètres

- `matrix` : `np.ndarray`
  - Matrice binaire d'entrée (shape `(n_reads, n_positions)`).
- `steps` : `List[Tuple[List[int], List[int], List[int]]]`
  - Liste des étapes de biclustering. Chaque step est un tuple `(reads1, reads0, cols)`.
- `read_names` : `List[str]`
  - Liste des noms de reads (doit correspondre à l'ordre des lignes de la matrice).
- `distance_thresh` : `float` (défaut : 0.1)
  - Seuil de distance de Hamming pour fusionner les clusters similaires.
- `min_reads_per_cluster` : `int | None` (défaut : 5)
  - Taille minimale d'un cluster. Si `None`, aucun filtrage n'est appliqué.

### Valeurs de retour

- `clusters` : `List[np.ndarray]`
  - Liste des clusters finaux, chaque cluster étant un tableau numpy des noms de reads.
- `reduced_matrix` : `np.ndarray`
  - Matrice réduite de shape `(nb_clusters, nb_steps)`, chaque ligne étant le profil consensus d'un cluster sur les stripes (une colonne par step).
- `orphan_reads_names` : `list[str]`
  - Liste des noms de reads non clusterisés à la fin (orphelins).
- `unused_columns` : `list[int]`
  - Liste des indices de colonnes non utilisées dans les steps (colonnes non "stripes").

### Exemple d'utilisation

```python
import numpy as np
from src.MatrixStriper.post_processing import post_processing

matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
steps = [([0, 2], [1], [0, 2]), ([1], [0, 2], [1])]
read_names = ['read1', 'read2', 'read3']
clusters, reduced, orphan_reads, unused_cols = post_processing(matrix, steps, read_names)
print("Clusters:", clusters)
print("Matrice réduite:", reduced)
print("Reads orphelins:", orphan_reads)
print("Colonnes inutilisées:", unused_cols)
```

### Remarques
- Les reads non assignés à un cluster à la fin sont listés dans `orphan_reads_names` et ne sont pas pris en compte dans la matrice réduite.
- Les colonnes non utilisées dans les steps sont listées dans `unused_columns` et ne sont pas représentées dans la matrice réduite.
- La matrice réduite permet de comparer les profils des clusters sur les stripes utilisés lors du biclustering.

---

## Sous-fonctions de `post_processing.py`

### `cluster_mean(cluster, matrix)`
Calcule le profil consensus (moyenne arrondie) d'un cluster sur toutes les colonnes de la matrice.
- **Paramètres** :
  - `cluster` : liste d'indices de reads
  - `matrix` : matrice binaire d'origine
- **Retour** : vecteur numpy de la moyenne arrondie sur chaque colonne

### `hamming_distance_with_mask(read_vec, mean_vec)`
Calcule la distance de Hamming entre deux vecteurs binaires (sans masque car la matrice ne contient que 0/1).
- **Paramètres** :
  - `read_vec` : vecteur binaire d'un read
  - `mean_vec` : vecteur consensus d'un cluster
- **Retour** : float (proportion de positions différentes)

### `merge_similar_clusters(clusters, means, distance_thresh)`
Fusionne les clusters dont les profils consensus sont trop proches selon la distance de Hamming.
- **Paramètres** :
  - `clusters` : liste de clusters (indices de reads)
  - `means` : liste des profils consensus de chaque cluster
  - `distance_thresh` : seuil de fusion
- **Retour** : nouvelle liste de clusters fusionnés

### `reassign_orphans(rem_, clusters, means, matrix, threshold=0.3)`
Réaffecte les reads orphelins au cluster le plus proche (distance de Hamming).
- **Paramètres** :
  - `rem_` : liste des indices de reads orphelins
  - `clusters` : liste de clusters courants
  - `means` : profils consensus des clusters
  - `matrix` : matrice binaire d'origine
  - `threshold` : seuil de distance pour accepter la réaffectation
- **Retour** : clusters mis à jour

---

# Fonctions de `clustering.py`

## `clustering_full_matrix`

Cette fonction applique un biclustering exhaustif et itératif sur une matrice binaire pour extraire tous les motifs significatifs (séparations de lignes selon des motifs de colonnes). Elle traite chaque région de colonnes indépendamment, applique un clustering binaire, et accumule les étapes valides.

- **Paramètres** :
  - `input_matrix` : `np.ndarray` — Matrice binaire d'entrée (0/1).
  - `regions` : `List[List[int]]` — Groupes d'indices de colonnes à traiter séparément.
  - `steps` : `List[Tuple[List[int], List[int], List[int]]]` — Résultats de clustering préexistants à préserver.
  - `min_row_quality` : `int` — Nombre minimal de lignes pour un cluster valide (défaut : 5).
  - `min_col_quality` : `int` — Nombre minimal de colonnes pour traiter une région (défaut : 3).
  - `error_rate` : `float` — Taux d'erreur toléré pour la détection de motifs (défaut : 0.025).

- **Retour** :
  - `steps` : Liste de toutes les étapes de clustering valides trouvées (triplets : indices lignes groupe 1, groupe 2, colonnes).
  - `metrics` : Dictionnaire de métriques sur le biclustering (nombre d'étapes, taille max de cluster, densité, etc).

- **Remarque** :
  - Seules les étapes où les deux groupes de lignes sont non vides et où le nombre de colonnes atteint le seuil sont conservées.

## `largest_only`

Cette fonction extrait la plus grande sous-matrice dense (quasi-biclique de 1) d'une matrice binaire, avec une densité minimale de 1 - error_rate. Elle ne fait qu'un seul passage pour trouver le cluster principal.

- **Paramètres** :
  - `input_matrix` : `np.ndarray` — Matrice binaire d'entrée (0/1).
  - `error_rate` : `float` — Densité minimale requise (défaut : 0.025).
  - `min_row_quality` : `int` — Nombre minimal de lignes pour un cluster (défaut : 5).
  - `min_col_quality` : `int` — Nombre minimal de colonnes pour un cluster (défaut : 3).

- **Retour** :
  - Tuple de listes :
    - Indices des lignes du cluster principal
    - Liste vide (pas de cluster opposé)
    - Indices des colonnes du cluster principal
  - Dictionnaire de métriques (taille, densité, indices, etc)

- **Remarque** :
  - Si aucun cluster valide n'est trouvé, retourne des listes vides et found=False dans les métriques.

## `clustering_step`

Effectue une seule étape de clustering binaire sur une matrice pour identifier une séparation significative de lignes.

- **Paramètres** :
  - `input_matrix` : `np.ndarray` — Matrice binaire d'entrée (0/1).
  - `error_rate` : `float` — Taux d'erreur toléré pour la détection de motifs (défaut : 0.025).
  - `min_row_quality` : `int` — Nombre minimal de lignes pour continuer (défaut : 5).
  - `min_col_quality` : `int` — Nombre minimal de colonnes pour continuer (défaut : 3).

- **Retour** :
  - Tuple de listes :
    - Indices des lignes à 1 (pattern positif)
    - Indices des lignes à 0 (pattern négatif)
    - Indices des colonnes où la séparation est la plus significative
  - Dictionnaire de métriques sur l'étape (taille, densité, etc)

- **Remarque** :
  - Retourne des listes vides si aucun motif significatif n'est trouvé.

---

# Fonctions de `pipeline.py`

## `compact_matrix`

Cette fonction orchestre l'ensemble du pipeline de biclustering et de compactage de matrice. Elle prend en entrée un fichier CSV de matrice binaire, applique le prétraitement, le biclustering, le post-traitement, puis sauvegarde la matrice réduite et les métriques de compression.

### Fonctionnement général

1. **Lecture de la matrice** :
   - Charge la matrice binaire et les noms de lignes depuis un fichier CSV.
2. **Prétraitement** :
   - Identifie les régions inhomogènes et les strips initiaux via `pre_processing`.
3. **Biclustering** :
   - Applique `clustering_full_matrix` pour extraire tous les motifs significatifs sur les régions inhomogènes.
4. **Post-traitement** :
   - Regroupe les reads en clusters finaux, construit la matrice réduite, identifie les reads orphelins et les colonnes inutilisées.
5. **Sauvegarde** :
   - Écrit la matrice réduite dans un CSV et sauvegarde les métriques dans un fichier texte.

### Paramètres
- `input_csv` : `str` — Chemin du fichier CSV d'entrée.
- `output_txt` : `str` — Chemin du fichier texte pour les métriques.
- `output_csv` : `str` — Chemin du fichier CSV de sortie.
- `min_col_quality` : `int` (défaut : 3) — Qualité minimale des colonnes.
- `min_row_quality` : `int` (défaut : 5) — Qualité minimale des lignes.
- `error_rate` : `float` (défaut : 0.025) — Taux d'erreur toléré.
- `distance_thresh` : `float` (défaut : 0.1) — Seuil de distance de Hamming pour fusionner les clusters.

### Valeur de retour
- `dict` : Dictionnaire de métriques de compression (nombre de clusters, ratio de compression, etc).

---

## `pipeline_ilp_largest_only`

Pipeline alternatif qui ne conserve que la plus grande sous-matrice dense (cluster principal) via ILP.

### Fonctionnement général
1. **Lecture de la matrice** depuis un CSV.
2. **Recherche du plus grand cluster** avec `largest_only`.
3. **Sauvegarde** des métriques dans un fichier texte.

### Paramètres
- `input_csv` : `str` — Chemin du fichier CSV d'entrée.
- `output_txt` : `str` — Chemin du fichier texte pour les métriques.
- `error_rate` : `float` (défaut : 0.025) — Taux d'erreur toléré.

### Valeur de retour
- `dict` : Dictionnaire de métriques sur le cluster principal.

---

# Fonctions utilitaires (`utils.py`)

## `load_csv_matrix`
Charge une matrice binaire (0/1) depuis un fichier CSV et retourne la matrice ainsi que la liste des noms de lignes.

- **Paramètres** :
  - `csv_file_path` : `str` — Chemin du fichier CSV.
- **Retour** :
  - `matrix` : `np.ndarray` — Matrice binaire.
  - `row_names` : `list` — Liste des noms de lignes.

## `write_matrix_csv`
Sauvegarde une matrice (avec noms de lignes et de colonnes) dans un fichier CSV.

- **Paramètres** :
  - `matrix` : `np.ndarray` — Matrice à sauvegarder.
  - `row_names` : `list` — Noms des lignes.
  - `col_names` : `list` — Noms des colonnes.
  - `output_csv` : `str` — Chemin du fichier de sortie.

## `save_dict_with_metadata`
Enregistre un dictionnaire dans un fichier texte, en ajoutant des métadonnées (date, heure, nombre de clés).

- **Paramètres** :
  - `data` : `dict` — Dictionnaire à sauvegarder.
  - `output_txt` : `str` — Chemin du fichier de sortie.

---

## Visualisation de matrice (matrix_visualizer.html)

Pour visualiser une matrice binaire (0/1) générée par MatrixStriper :

1. Ouvrez le fichier `matrix_visualizer.html` dans votre navigateur (double-clic ou ouvrir avec Chrome/Firefox).
2. Glissez-déposez ou sélectionnez un fichier `.csv` contenant la matrice.
   - Le CSV doit contenir uniquement des 0 et 1, séparés par des virgules.
3. Après avoir chargé le CSV, une seconde zone d'upload apparaît pour un fichier `.txt` (optionnel).
   - Ce fichier `.txt` doit être issu d'une exécution MatrixStriper avec l'option `--largest_only` (ex : `res8.txt`).
   - Il contient les indices de lignes et colonnes (`row_indices`/`col_indices` ou `list_rows_indices`/`list_cols_indices`) à afficher en priorité.
4. Glissez-déposez ou sélectionnez ce `.txt` : la matrice sera automatiquement réordonnée selon ces indices.

**Résumé** :
- Zone 1 : chargez le `.csv` (matrice).
- Zone 2 : chargez le `.txt` (optionnel, pour l'ordre des lignes/colonnes, issu de --largest_only).

Vous pouvez ainsi visualiser rapidement la structure de vos matrices et mettre en avant les sous-matrices d'intérêt extraites par MatrixStriper.

---
