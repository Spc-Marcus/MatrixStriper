import numpy as np
import pandas as pd
import datetime
import logging
logger = logging.getLogger(__name__)

def load_csv_matrix(csv_file_path: str) -> tuple:
    """
    Load CSV matrix with only 0s and 1s, and return also the row names.

    Parameters
    ----------
    csv_file_path : str
        Path to CSV file containing binary matrix

    Returns
    -------
    tuple
        (matrix: np.ndarray, row_names: list)
        Binary matrix with values 0 and 1, and the list of row names
    """
    # Load CSV file
    df = pd.read_csv(csv_file_path, index_col=0)

    # Ensure binary values (0, 1)
    df = df.fillna(0).astype(int)
    df = df.clip(0, 1)  # Ensure only 0 and 1 values

    matrix = df.values
    row_names = list(df.index)
    return matrix, row_names

def write_matrix_csv(matrix: np.ndarray, row_names: list, col_names: list, output_csv: str):
    """
    Sauvegarde une matrice avec identifiants en première ligne et colonne.

    Args:
        matrix (np.ndarray): Matrice à sauvegarder.
        row_names (list): Noms des lignes.
        col_names (list): Noms des colonnes.
        output_csv (str): Chemin du fichier de sortie.
    """
    # Forcer row_names et col_names à être des listes 1D
    row_names = np.array(row_names).flatten().tolist()
    col_names = np.array(col_names).flatten().tolist()
    # Forcer la matrice à être en int (0/1)
    matrix = np.array(matrix, dtype=int)
    # Create DataFrame
    df = pd.DataFrame(matrix, index=row_names, columns=col_names)
    # Save to CSV
    df.to_csv(output_csv, index=True)

def save_dict_with_metadata(data: dict, output_txt: str):
    """
    Enregistre un dictionnaire dans un fichier texte avec des métadonnées (date, heure, etc).

    Args:
        data (dict): Dictionnaire à sauvegarder.
        output_txt (str): Chemin du fichier de sortie.
    """
    now = datetime.datetime.now()
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# Sauvegarde de metrique\n")
        f.write(f"# Date et heure : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Nombre de clés : {len(data)}\n")
        f.write(f"# ---\n")
        for key, value in data.items():
            f.write(f"{key}: {value}\n")