import os
import glob
import time
import pandas as pd
import numpy as np
from .pre_processing import pre_processing
from .post_processing import post_processing
from ilp.clustering import clustering_full_matrix
from utils.utils import load_csv_matrix
import logging

logger = logging.getLogger(__name__)

def test_matrix_pipeline(
    input_csv: str,
    min_col_quality: int = 3,
    min_row_quality: int = 5,
    error_rate: float = 0.025,
    distance_thresh: float = 0.1,
    min_reads_per_cluster: int = 5
) -> dict:
    """
    Test du pipeline sur une matrice donnée (pre-processing + ILP + post-processing).
    
    Args:
        input_csv (str): Chemin du fichier CSV d'entrée.
        min_col_quality (int): Qualité minimale des colonnes.
        min_row_quality (int): Qualité minimale des lignes.
        error_rate (float): Taux d'erreur toléré.
        distance_thresh (float): Seuil de distance de Hamming pour fusionner.
        min_reads_per_cluster (int): Nombre minimum de reads par cluster.
        
    Returns:
        dict: Métriques de compression.
    """
    start_time = time.time()
    
    # 1. Lecture du CSV
    logger.info(f"Loading matrix from {input_csv}")
    matrix, row_names = load_csv_matrix(input_csv)
    logger.info(f"Matrix loaded with shape: {matrix.shape}")
    check1_time = time.time()
    load_time = check1_time - start_time
    
    # 2. Pre-processing
    logger.info(f"Starting pre-processing")
    inhomogenious_regions, steps_pre_processing = pre_processing(
        matrix, min_col_quality=min_col_quality, certitude=0.2, error_rate=error_rate
    )
    logger.info(f"Pre-processing done: {len(steps_pre_processing)} steps found")
    check2_time = time.time()
    pre_processing_time = check2_time - check1_time
    logger.info(f"Steps found: {steps_pre_processing}")
    # 3. ILP clustering
    logger.info(f"Starting ILP clustering")
    matrix_bin = matrix  # Assuming matrix is already binary
    steps_ilp, metrics_ilp = clustering_full_matrix(
        matrix_bin, inhomogenious_regions,
        min_row_quality=min_row_quality, 
        min_col_quality=min_col_quality, 
        error_rate=error_rate
    )
    logger.info(f"ILP clustering done: {len(steps_ilp)} steps found")
    check3_time = time.time()
    ilp_time = check3_time - check2_time
    
    # 4. Post-processing (avec les steps du pre-processing + ILP)
    logger.info(f"Starting post-processing")
    all_steps = steps_pre_processing + steps_ilp
    clusters, reduced_matrix, orphan_reads_names, unused_columns = post_processing(
        matrix, all_steps, row_names, 
        distance_thresh=distance_thresh, 
        min_reads_per_cluster=min_reads_per_cluster
    )
    logger.info(f"Post-processing done: {len(clusters)} clusters found")
    check4_time = time.time()
    post_processing_time = check4_time - check3_time
    
    # 5. Calcul des métriques
    nb_reads = len(matrix)
    nb_positions = len(matrix[0]) if len(matrix) > 0 else 0
    nb_clusters = len(clusters)
    nb_orphan_reads = len(orphan_reads_names)
    nb_unused_columns = len(unused_columns)
    nb_reads_clustered = nb_reads - nb_orphan_reads
    nb_positions_used = nb_positions - nb_unused_columns
    
    # Calcul de la taille de la matrice réduite
    if reduced_matrix.size > 0:
        reduced_shape = reduced_matrix.shape
    else:
        reduced_shape = (0, len(all_steps))
    
    compression_ratio = (reduced_shape[0] * reduced_shape[1]) / (nb_reads * nb_positions) if nb_reads > 0 and nb_positions > 0 else 0
    percent_reads_clustered = nb_reads_clustered / nb_reads if nb_reads > 0 else 0
    percent_positions_used = nb_positions_used / nb_positions if nb_positions > 0 else 0
    
    nb_reads_per_cluster = [len(cluster) for cluster in clusters]
    nb_positions_per_strip = [len(strip[2]) for strip in all_steps]
    
    total_time = check4_time - start_time
    
    metrics = {
        "matrix_name": os.path.basename(input_csv),
        "matrix_path": input_csv,
        "nb_haplotypes": extract_haplotype_count(input_csv),
        "nb_steps_pre_processing": len(steps_pre_processing),
        "nb_steps_ilp": len(steps_ilp),
        "nb_clusters_final": nb_clusters,
        "execution_time": total_time,
        "nb_unused_columns": nb_unused_columns,
        "nb_orphan_reads": nb_orphan_reads,
        "original_matrix_shape": (nb_reads, nb_positions),
        "reduced_matrix_shape": reduced_shape,
        "compression_ratio": compression_ratio,
        "percent_reads_clustered": percent_reads_clustered,
        "percent_positions_used": percent_positions_used,
        "cluster_size_min": min(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "cluster_size_max": max(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "cluster_size_mean": sum(nb_reads_per_cluster)/len(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "strip_size_min": min(nb_positions_per_strip) if nb_positions_per_strip else 0,
        "strip_size_max": max(nb_positions_per_strip) if nb_positions_per_strip else 0,
        "strip_size_mean": sum(nb_positions_per_strip)/len(nb_positions_per_strip) if nb_positions_per_strip else 0,
        "time_load": load_time,
        "time_pre_processing": pre_processing_time,
        "time_ilp": ilp_time,
        "time_post_processing": post_processing_time,
        "min_col_quality": min_col_quality,
        "min_row_quality": min_row_quality,
        "error_rate": error_rate,
        "distance_thresh": distance_thresh,
        "min_reads_per_cluster": min_reads_per_cluster
    }
    
    # Ajouter les métriques ILP si disponibles
    if metrics_ilp:
        metrics.update(metrics_ilp)
    
    return metrics

def extract_haplotype_count(file_path: str) -> int:
    """
    Extrait le nombre d'haplotypes depuis le chemin du fichier.
    Format attendu: data/X/YY.csv où X est le nombre d'haplotypes.
    """
    try:
        # Extraire le nom du dossier parent (X dans data/X/YY.csv)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        return int(parent_dir)
    except (ValueError, IndexError):
        logger.warning(f"Impossible d'extraire le nombre d'haplotypes depuis {file_path}")
        return -1

def run_batch_test(
    data_dir: str = "data",
    output_csv: str = "test_results.csv",
    min_col_quality: int = 3,
    min_row_quality: int = 5,
    error_rate: float = 0.025,
    distance_thresh: float = 0.1,
    min_reads_per_cluster: int = 5,
    max_matrices_per_folder: int = 20
) -> pd.DataFrame:
    """
    Lance les tests sur toutes les matrices dans le dossier data.
    
    Args:
        data_dir (str): Dossier contenant les sous-dossiers d'haplotypes.
        output_csv (str): Fichier de sortie pour les résultats.
        min_col_quality (int): Qualité minimale des colonnes.
        min_row_quality (int): Qualité minimale des lignes.
        error_rate (float): Taux d'erreur toléré.
        distance_thresh (float): Seuil de distance de Hamming pour fusionner.
        min_reads_per_cluster (int): Nombre minimum de reads par cluster.
        max_matrices_per_folder (int): Nombre maximum de matrices à traiter par dossier.
        
    Returns:
        pd.DataFrame: DataFrame contenant tous les résultats.
    """
    logger.info(f"Starting batch test on directory: {data_dir}")
    
    # Chercher tous les fichiers CSV dans data/X/YY.csv
    pattern = os.path.join(data_dir, "*", "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Grouper les fichiers par dossier et limiter à max_matrices_per_folder par dossier
    files_by_folder = {}
    for csv_file in csv_files:
        folder = os.path.dirname(csv_file)
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        files_by_folder[folder].append(csv_file)
    
    # Limiter le nombre de fichiers par dossier
    limited_csv_files = []
    for folder, files in files_by_folder.items():
        # Trier les fichiers pour avoir un ordre déterministe
        files.sort()
        # Prendre seulement les max_matrices_per_folder premiers
        limited_files = files[:max_matrices_per_folder]
        limited_csv_files.extend(limited_files)
        logger.info(f"Folder {folder}: processing {len(limited_files)} out of {len(files)} matrices")
    
    logger.info(f"Total matrices to process after limiting: {len(limited_csv_files)}")
    
    # Créer le fichier CSV avec les en-têtes
    columns_order = [
        "matrix_name", "matrix_path", "nb_haplotypes", 
        "nb_steps_pre_processing", "nb_steps_ilp", "nb_clusters_final", "execution_time",
        "nb_unused_columns", "nb_orphan_reads", "original_matrix_shape"
    ]
    
    # Créer un DataFrame vide avec les colonnes pour définir l'ordre
    empty_df = pd.DataFrame(columns=columns_order)
    empty_df.to_csv(output_csv, index=False)
    logger.info(f"Created CSV file with headers: {output_csv}")
    
    results = []
    
    for i, csv_file in enumerate(limited_csv_files):
        logger.info(f"Processing file {i+1}/{len(limited_csv_files)}: {csv_file}")
        
        try:
            metrics = test_matrix_pipeline(
                csv_file,
                min_col_quality=min_col_quality,
                min_row_quality=min_row_quality,
                error_rate=error_rate,
                distance_thresh=distance_thresh,
                min_reads_per_cluster=min_reads_per_cluster
            )
            results.append(metrics)
            logger.info(f"Successfully processed {csv_file}")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            # Ajouter une ligne d'erreur
            error_metrics = {
                "matrix_name": os.path.basename(csv_file),
                "matrix_path": csv_file,
                "nb_haplotypes": extract_haplotype_count(csv_file),
                "nb_steps_pre_processing": -1,
                "nb_steps_ilp": -1,
                "nb_clusters_final": -1,
                "execution_time": -1,
                "nb_unused_columns": -1,
                "nb_orphan_reads": -1,
                "original_matrix_shape": (-1, -1),
                "reduced_matrix_shape": (-1, -1),
                "compression_ratio": -1,
                "percent_reads_clustered": -1,
                "percent_positions_used": -1,
                "cluster_size_min": -1,
                "cluster_size_max": -1,
                "cluster_size_mean": -1,
                "strip_size_min": -1,
                "strip_size_max": -1,
                "strip_size_mean": -1,
                "time_load": -1,
                "time_pre_processing": -1,
                "time_ilp": -1,
                "time_post_processing": -1,
                "error": str(e)
            }
            results.append(error_metrics)
        
        # Sauvegarder au fur et à mesure
        if results:
            # Créer un DataFrame temporaire avec les résultats actuels
            temp_df = pd.DataFrame(results)
            
            # Réorganiser les colonnes selon l'ordre défini
            other_columns = [col for col in temp_df.columns if col not in columns_order]
            final_columns = columns_order + other_columns
            
            # S'assurer que toutes les colonnes existent
            for col in final_columns:
                if col not in temp_df.columns:
                    temp_df[col] = None
            
            temp_df = temp_df[final_columns]
            
            # Sauvegarder dans le CSV
            temp_df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv} ({len(results)} matrices processed so far)")
    
    # Créer le DataFrame final
    df = pd.DataFrame(results)
    
    # Réorganiser les colonnes selon la demande
    other_columns = [col for col in df.columns if col not in columns_order]
    final_columns = columns_order + other_columns
    
    # S'assurer que toutes les colonnes existent
    for col in final_columns:
        if col not in df.columns:
            df[col] = None
    
    df = df[final_columns]
    
    logger.info(f"Final results saved to {output_csv}")
    logger.info(f"Total matrices processed: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Lancer les tests
    results_df = run_batch_test()
    print(f"Test completed. Processed {len(results_df)} matrices.")
    print(f"Results saved to test_results.csv") 