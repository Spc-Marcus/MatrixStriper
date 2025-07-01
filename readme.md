# MatrixStriper

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
