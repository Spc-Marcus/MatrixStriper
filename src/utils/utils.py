import numpy as np
import pandas as pd

def load_csv_matrix(csv_file_path: str) -> np.ndarray:
    """
    Load CSV matrix with only 0s and 1s.
    
    Parameters
    ----------
    csv_file_path : str
        Path to CSV file containing binary matrix
        
    Returns
    -------
    np.ndarray
        Binary matrix with values 0 and 1
    """
    # Load CSV file
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Ensure binary values (0, 1)
    df = df.fillna(0).astype(int)
    df = df.clip(0, 1)  # Ensure only 0 and 1 values
    
    return df.values

def write_matrix_csv(matrix: np.ndarray, row_names: list, col_names: list, output_csv: str):
    """
    Sauvegarde une matrice avec identifiants en première ligne et colonne.

    Args:
        matrix (np.ndarray): Matrice à sauvegarder.
        row_names (list): Noms des lignes.
        col_names (list): Noms des colonnes.
        output_csv (str): Chemin du fichier de sortie.
    """
    # Create DataFrame
    df = pd.DataFrame(matrix, index=row_names, columns=col_names)
    
    # Save to CSV
    df.to_csv(output_csv, index=True)