# Modèles ILP du dossier `model/`

Ce dossier contient les modèles d'optimisation linéaire entière (ILP) utilisés par MatrixStriper pour l'extraction de sous-matrices denses (quasi-bicliques) dans des matrices binaires. Ces modèles sont formulés avec la bibliothèque PuLP et servent de cœur aux fonctions de clustering avancé du pipeline.

## Sommaire
- [max_Ones](#max_ones)
- [max_Ones_comp](#max_ones_comp)
- [max_e_r](#max_e_r)
- [max_e_wr](#max_e_wr)

---

## `max_Ones`

**Fichier :** `max_one_pulp.py`

Modèle ILP pour maximiser le nombre de 1 dans une sous-matrice extraite, tout en tolérant un certain taux de 0 (erreur). Il sélectionne un sous-ensemble de lignes et de colonnes pour maximiser la densité de 1.

- **Paramètres** :
  - `rows_data` : liste de tuples `(row, degree)` — lignes et leur degré (nombre de 1).
  - `cols_data` : liste de tuples `(col, degree)` — colonnes et leur degré.
  - `edges` : liste de tuples `(row, col)` — positions des 1 dans la matrice.
  - `rho` : float — taux maximal de 0 accepté dans la sous-matrice.
- **Retour** :
  - Un objet `LpProblem` PuLP prêt à être résolu.

---

## `max_Ones_comp`

**Fichier :** `max_one_pulp.py`

Variante compacte du modèle précédent, ajoutant des contraintes sur le degré des lignes et colonnes sélectionnées pour favoriser des clusters plus compacts et denses.

- **Paramètres** :
  - Identiques à `max_Ones`.
- **Retour** :
  - Un objet `LpProblem` PuLP prêt à être résolu.

---

## `max_e_r`

**Fichier :** `max_e_r_pulp.py`

Modèle ILP avancé pour extraire un quasi-biclique dense, inspiré de la littérature (Chnag et al.). Il maximise la somme des variables d'appartenance à la sous-matrice, sous contraintes de densité sur les lignes et colonnes.

- **Paramètres** :
  - `rows_data` : liste de tuples `(row, degree)` — lignes et leur degré.
  - `cols_data` : liste de tuples `(col, degree)` — colonnes et leur degré.
  - `edges` : liste de tuples `(row, col)` — positions des 1 dans la matrice.
  - `delta` : float — tolérance d'erreur pour la densité (1 - densité minimale).
- **Retour** :
  - Un objet `LpProblem` PuLP prêt à être résolu.

---

## `max_e_wr`

**Fichier :** `max_e_r_pulp.py`

Modèle ILP pour l'extension d'une solution seed (sous-matrice initiale) à l'ensemble de la matrice. Il maximise la taille du quasi-biclique tout en imposant une amélioration de l'objectif précédent et en respectant les contraintes de densité.

- **Paramètres** :
  - `rows_data`, `cols_data`, `edges` : identiques à `max_e_r`.
  - `rows_res` : liste d'indices de lignes de la solution seed.
  - `cols_res` : liste d'indices de colonnes de la solution seed.
  - `prev_obj` : float — valeur de l'objectif précédent (seed).
  - `delta` : float — tolérance d'erreur pour la densité.
- **Retour** :
  - Un objet `LpProblem` PuLP prêt à être résolu.

---

## Utilisation dans MatrixStriper

Ces modèles sont appelés par les fonctions ILP du pipeline (`ilp_pulp.py`) pour détecter et extraire automatiquement des sous-matrices denses (clusters) dans les matrices binaires d'entrée. Ils sont adaptés pour gérer le bruit et les matrices de grande taille grâce à la résolution efficace via CBC (par défaut dans PuLP).

- Pour plus de détails sur l'intégration, voir la documentation de `ilp_pulp.py` et du pipeline principal. 