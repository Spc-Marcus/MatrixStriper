import numpy as np
import pandas as pd

def read_matrix_csv(*args, **kwargs):
    """
    Lit un CSV avec identifiants en première ligne et colonne.

    Args:
        input_csv (str): Chemin du fichier CSV.

    Returns:
        matrix (np.ndarray): Matrice de données.
        row_names (list): Noms des lignes.
        col_names (list): Noms des colonnes.
    """
    # ... à implémenter ...
    pass

def write_matrix_csv(*args, **kwargs):
    """
    Sauvegarde une matrice avec identifiants en première ligne et colonne.

    Args:
        matrix (np.ndarray): Matrice à sauvegarder.
        row_names (list): Noms des lignes.
        col_names (list): Noms des colonnes.
        output_csv (str): Chemin du fichier de sortie.
    """
    # ... à implémenter ...
    pass 