import numpy as np

def clustering_full_matrix(input_matrix, regions, steps, min_row_quality=5, min_col_quality=3, error_rate=0.025, **kwargs):
    """
    Biclustering exhaustif itératif pour extraire tous les patterns significatifs.

    Args:
        input_matrix (np.ndarray): Matrice binaire.
        regions (list): Régions de colonnes à traiter.
        steps (list): Étapes de division.
        min_row_quality (int): Qualité minimale des lignes.
        min_col_quality (int): Qualité minimale des colonnes.
        error_rate (float): Taux d'erreur toléré.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: Séparations trouvées.
    """
    # ... à implémenter ...
    pass 