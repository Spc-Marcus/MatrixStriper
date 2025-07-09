# Test Batch MatrixStriper

Ce module permet de tester le pipeline MatrixStriper (pre-processing + post-processing uniquement, sans ILP) sur un ensemble de matrices organisées par nombre d'haplotypes.

## Structure des données

Les matrices doivent être organisées dans la structure suivante :
```
data/
├── 2/
│   ├── matrix_1.csv
│   ├── matrix_2.csv
│   └── ...
├── 4/
│   ├── matrix_1.csv
│   ├── matrix_2.csv
│   └── ...
├── 6/
│   └── ...
├── 8/
│   └── ...
└── 10/
    └── ...
```

Où le nom du dossier (2, 4, 6, 8, 10) représente le nombre d'haplotypes dans les matrices.

## Scripts disponibles

### 1. `create_test_data.py`
Génère des données de test avec la structure demandée.

```bash
python create_test_data.py
```

Cela créera :
- 3 matrices par nombre d'haplotypes (2, 4, 6, 8, 10)
- Chaque matrice aura 100 reads et 50 positions
- Un niveau de bruit de 5%

### 2. `run_batch_test.py`
Script principal pour exécuter les tests de batch.

```bash
python run_batch_test.py [options]
```

#### Options disponibles :
- `--data-dir` : Dossier contenant les données (default: `data`)
- `--output` : Fichier de sortie (default: `test_results.csv`)
- `--min-col-quality` : Qualité minimale des colonnes (default: 3)
- `--min-row-quality` : Qualité minimale des lignes (default: 5)
- `--error-rate` : Taux d'erreur toléré (default: 0.025)
- `--distance-thresh` : Seuil de distance de Hamming (default: 0.1)
- `--min-reads-per-cluster` : Nombre minimum de reads par cluster (default: 5)
- `--log-level` : Niveau de logging (default: INFO)

#### Exemples d'utilisation :

```bash
# Test avec paramètres par défaut
python run_batch_test.py

# Test avec paramètres personnalisés
python run_batch_test.py --data-dir my_data --output results.csv --error-rate 0.05

# Test avec logging détaillé
python run_batch_test.py --log-level DEBUG
```

## Format de sortie

Le script génère un fichier CSV avec les colonnes suivantes :

| Colonne | Description |
|---------|-------------|
| `matrix_name` | Nom du fichier matrice |
| `matrix_path` | Chemin complet vers le fichier |
| `nb_haplotypes` | Nombre d'haplotypes (extrait du nom du dossier) |
| `nb_steps_pre_processing` | Nombre de steps trouvés par le pre-processing |
| `nb_clusters_final` | Nombre de clusters finaux après post-processing |
| `execution_time` | Temps d'exécution total (secondes) |
| `nb_unused_columns` | Nombre de colonnes non utilisées dans les steps |
| `nb_orphan_reads` | Nombre de reads non clusterisés (orphelins) |
| `original_matrix_shape` | Taille de la matrice d'entrée (reads, positions) |
| `reduced_matrix_shape` | Taille de la matrice réduite (clusters, steps) |
| `compression_ratio` | Ratio de compression |
| `percent_reads_clustered` | Pourcentage de reads clusterisés |
| `percent_positions_used` | Pourcentage de positions utilisées |
| `cluster_size_min/max/mean` | Statistiques sur la taille des clusters |
| `strip_size_min/max/mean` | Statistiques sur la taille des strips |
| `time_load/pre_processing/post_processing` | Temps par étape |

## Exemple de sortie

```csv
matrix_name,matrix_path,nb_haplotypes,nb_steps_pre_processing,nb_clusters_final,execution_time,nb_unused_columns,nb_orphan_reads,original_matrix_shape,reduced_matrix_shape,compression_ratio,percent_reads_clustered,percent_positions_used,cluster_size_min,cluster_size_max,cluster_size_mean,strip_size_min,strip_size_max,strip_size_mean,time_load,time_pre_processing,time_post_processing,min_col_quality,min_row_quality,error_rate,distance_thresh,min_reads_per_cluster
matrix_1.csv,data/2/matrix_1.csv,2,5,2,1.23,10,5,"(100, 50)","(2, 5)",0.002,0.95,0.8,45,50,47.5,3,12,8.0,0.1,0.8,0.33,3,5,0.025,0.1,5
matrix_2.csv,data/4/matrix_2.csv,4,8,4,2.45,15,8,"(100, 50)","(4, 8)",0.0064,0.92,0.7,20,28,25.0,2,8,6.25,0.12,1.2,1.13,3,5,0.025,0.1,5
```

## Pipeline utilisé

Le script utilise uniquement :
1. **Pre-processing** : Identification des strips par clustering hiérarchique
2. **Post-processing** : Création des clusters finaux et réduction de matrice

**Pas d'ILP** : Contrairement au pipeline complet, ce test n'utilise pas l'étape ILP.

## Dépendances

- numpy
- pandas
- scikit-learn
- scipy

## Utilisation avancée

Vous pouvez également utiliser directement les fonctions du module :

```python
from src.MatrixStriper.test_pipeline import test_matrix_pipeline, run_batch_test

# Test d'une seule matrice
metrics = test_matrix_pipeline("data/2/matrix_1.csv")

# Test de batch
results_df = run_batch_test(data_dir="data", output_csv="results.csv")
``` 