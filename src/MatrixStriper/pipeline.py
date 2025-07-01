from pre_processing import pre_processing
from clustering import clustering_full_matrix
from post_processing import post_processing
from utils import load_csv_matrix, write_matrix_csv

def compact_matrix(
    input_csv,
    output_csv,
    largest_only=True,
    min_col_quality=3,
    min_row_quality=5,
    error_rate=0.025,
    distance_thresh=0.1
) -> dict:
    """
    Orchestration du pipeline de biclustering et compactage de matrice.

    Args:
        input_csv (str): Chemin du fichier CSV d'entrée.
        output_csv (str): Chemin du fichier CSV de sortie.
        largest_only (bool): Si True, ne conserve que la plus grande sous-matrice.
        min_col_quality (int): Qualité minimale des colonnes.
        min_row_quality (int): Qualité minimale des lignes.
        error_rate (float): Taux d'erreur toléré.
        certitude (float): Seuil de certitude pour la binarisation.
        distance_thresh (float): Seuil de distance de Hamming pour fusionner.

    Returns:
        dict: Métriques de compression.
    """
    # 1. Lecture du CSV
    matrix, row_names, col_names = load_csv_matrix(input_csv)

    # 2. Pipeline
    inhomogenious_regions, steps = pre_processing(
        matrix, min_col_quality=min_col_quality, certitude=0.35, error_rate=error_rate
    )
    # For now, use the original matrix as binary matrix (assuming it's already binary)
    matrix_bin = matrix
    patterns = clustering_full_matrix(
        matrix_bin, inhomogenious_regions, steps,
        min_row_quality=min_row_quality, min_col_quality=min_col_quality, error_rate=error_rate
    )
    clusters = post_processing(matrix_bin, steps, row_names, distance_thresh=distance_thresh)

    # 3. Extraction de la plus grande sous-matrice (si demandé)
    if largest_only:
        # ... à implémenter : extraire le plus grand cluster ...
        pass

    # 4. Sauvegarde
    # ... à implémenter : write_matrix_csv ...

    # 5. Métriques de compression
    metrics = {
        # ... à implémenter ...
    }
    return metrics 