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
        largest_only (bool): Si True, ne conserve que la plus grande sous-matrice (cluster principal) lors du post-traitement. Si False, conserve tous les clusters détectés.
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
    steps, metrics_ilp = clustering_full_matrix(
        matrix_bin, inhomogenious_regions, steps,
        min_row_quality=min_row_quality, min_col_quality=min_col_quality, error_rate=error_rate
    )
    clusters, reduced_matrix, orphan_reads_names, unused_columns = post_processing(matrix_bin, steps, row_names, distance_thresh=distance_thresh)
    row_names = ["read_" + str(i) for i in range(len(reduced_matrix))]
    col_names = ["strip_" + str(i) for i in range(len(reduced_matrix[0]))]
    # 3. Sauvegarde
    write_matrix_csv(output_csv, reduced_matrix, row_names, col_names)

    # 4. Métriques de compression
    nb_reads = len(matrix)
    nb_positions = len(matrix[0]) if len(matrix) > 0 else 0
    nb_clusters = len(clusters)
    nb_steps = len(steps)
    nb_orphan_reads = len(orphan_reads_names)
    nb_unused_columns = len(unused_columns)
    nb_reads_clustered = nb_reads - nb_orphan_reads
    nb_positions_used = nb_positions - nb_unused_columns
    reduced_shape = reduced_matrix.shape if hasattr(reduced_matrix, 'shape') else (len(reduced_matrix), len(reduced_matrix[0]) if reduced_matrix else 0)
    compression_ratio = (reduced_shape[0] * reduced_shape[1]) / (nb_reads * nb_positions) if nb_reads > 0 and nb_positions > 0 else 0
    percent_reads_clustered = nb_reads_clustered / nb_reads if nb_reads > 0 else 0
    percent_positions_used = nb_positions_used / nb_positions if nb_positions > 0 else 0
    nb_reads_per_cluster = [len(cluster) for cluster in clusters]
    nb_positions_per_strip = [len(strip[2]) for strip in steps]
    metrics = {
        "nb_clusters": nb_clusters,
        "nb_inhomogenious_regions": len(inhomogenious_regions),
        "nb_steps": nb_steps,
        "nb_reads_per_cluster": nb_reads_per_cluster,
        "nb_positions_per_strip": nb_positions_per_strip,
        "orphan_reads_names": orphan_reads_names,
        "unused_columns": unused_columns,
        "original_matrix_shape": (nb_reads, nb_positions),
        "reduced_matrix_shape": reduced_shape,
        "compression_ratio": compression_ratio,
        "nb_orphan_reads": nb_orphan_reads,
        "nb_unused_columns": nb_unused_columns,
        "percent_reads_clustered": percent_reads_clustered,
        "percent_positions_used": percent_positions_used,
        "cluster_size_min": min(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "cluster_size_max": max(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "cluster_size_mean": sum(nb_reads_per_cluster)/len(nb_reads_per_cluster) if nb_reads_per_cluster else 0,
        "strip_size_min": min(nb_positions_per_strip) if nb_positions_per_strip else 0,
        "strip_size_max": max(nb_positions_per_strip) if nb_positions_per_strip else 0,
        "strip_size_mean": sum(nb_positions_per_strip)/len(nb_positions_per_strip) if nb_positions_per_strip else 0,
    }
    # Ajout des métriques ILP
    if metrics_ilp:
        metrics.update(metrics_ilp)
    return metrics 