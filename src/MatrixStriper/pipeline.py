from .pre_processing import pre_processing
from ilp.clustering import clustering_full_matrix, largest_only
from .post_processing import post_processing
from utils.utils import load_csv_matrix, write_matrix_csv, save_dict_with_metadata
import logging
logger = logging.getLogger(__name__)
import time
def compact_matrix(
    input_csv,
    output_txt,
    output_csv,
    min_col_quality=3,
    min_row_quality=5,
    error_rate=0.025,
    distance_thresh_post=0.1,
    distance_thresh_CH=0.05
) -> dict:
    """
    Orchestrate the full biclustering + compaction pipeline and produce summary metrics.

    High-level overview
    - Load a binary matrix (CSV) and optional row names.
    - Pre-process to detect inhomogeneous column regions.
    - Run ILP-based biclustering (clustering_full_matrix) on detected regions.
    - Post-process to compact the matrix into strips/stripes and produce clusters.
    - Persist the reduced matrix (CSV) and pipeline metrics (JSON/TXT with metadata).

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file containing the matrix to compact. The function expects
        a numeric/binary matrix where rows are reads/observations and columns are positions.
    output_txt : str
        Path where pipeline metrics and metadata will be saved (JSON-like TXT).
    output_csv : str
        Path where the compacted/reduced matrix (CSV) will be written.
    min_col_quality : int, optional
        Minimum number of columns required for a strip to be considered valid (default: 3).
    min_row_quality : int, optional
        Minimum number of rows required for a cluster to be considered valid (default: 5).
    error_rate : float, optional
        Tolerance for quasi-biclique detection (default: 0.025).
    distance_thresh_post : float, optional
        Hamming distance threshold used during post-processing to merge similar rows/clusters
        (default: 0.1).

    Returns
    -------
    dict
        A dictionary containing:
        - pipeline timing information (load, pre-processing, ILP, post-processing)
        - counts (clusters, orphan reads, unused columns, strips)
        - per-cluster and per-strip sizes
        - lists of read names per cluster and list of columns per strip
        - ILP metrics if available
        This dictionary is also saved to output_txt via save_dict_with_metadata.

    Example
    -------
    metrics = compact_matrix(
        "input_reads.csv",
        "pipeline_metrics.txt",
        "compacted_matrix.csv",
        min_col_quality=3,
        min_row_quality=5,
        error_rate=0.02,
        distance_thresh_post=0.08
    )

    Notes
    -----
    - The function does not change the original input file.
    - The input matrix is assumed to be already binarized; no additional binarization is performed.
    - ILP steps can be time-consuming; tune error_rate, TimeLimit and min_*_quality for performance.
    - Returned metrics contain lists that can be large (per-cluster read lists) — saved for reproducibility.
    """
    start_time = time.time()
    # 1. Lecture du CSV
    logger.info(f"Loading matrix from {input_csv}")
    matrix, row_names = load_csv_matrix(input_csv)
    logger.info(f"Matrix loaded with shape: {matrix.shape}")
    check1_time = time.time()
    load_time = check1_time - start_time
    # 2. Pipeline
    logger.info(f"Starting pre-processing")
    inhomogenious_regions, steps = pre_processing(
        matrix, min_col_quality=min_col_quality, certitude=distance_thresh_CH, error_rate=error_rate
    )
    steps_pre_processing = steps
    logger.info(f"Pre-processing done")
    check2_time = time.time()
    pre_processing_time = check2_time - check1_time
    logger.info(f"Starting ILP")
    # For now, use the original matrix as binary matrix (assuming it's already binary)
    matrix_bin = matrix
    steps_ilp, metrics_ilp = clustering_full_matrix(
        matrix_bin, inhomogenious_regions,
        min_row_quality=min_row_quality, min_col_quality=min_col_quality, error_rate=error_rate
    )
    steps = steps_pre_processing + steps_ilp
    logger.info(f"ILP done")
    check3_time = time.time()
    ilp_time = check3_time - check2_time
    logger.info(f"Starting post-processing")
    clusters, reduced_matrix, orphan_reads_names, unused_columns = post_processing(
        matrix_bin, steps, row_names, distance_thresh=distance_thresh_post)
    logger.info(f"Post-processing done")
    row_names = ["read_" + str(i) for i in range(len(clusters))]
    col_names = ["strip_" + str(i) for i in range(len(steps))]
    logger.debug(f"Row names: {row_names}")
    logger.debug(f"Col names: {col_names}")
    logger.debug(f"Reduced matrix: {reduced_matrix}")
    check4_time = time.time()
    post_processing_time = check4_time - check3_time
    # 3. Sauvegarde
    write_matrix_csv(reduced_matrix, row_names, col_names, output_csv)
    logger.info(f"Matrix saved to {output_csv}")
    # 4. Métriques de compression
    nb_reads = len(matrix)
    nb_positions = len(matrix[0]) if len(matrix) > 0 else 0
    nb_clusters = len(clusters)
    nb_orphan_reads = len(orphan_reads_names)
    nb_unused_columns = len(unused_columns)
    nb_reads_clustered = nb_reads - nb_orphan_reads
    nb_positions_used = nb_positions - nb_unused_columns
    reduced_shape = reduced_matrix.shape if hasattr(reduced_matrix, 'shape') else (len(reduced_matrix), len(reduced_matrix[0]) if reduced_matrix else 0)
    percent_reads_clustered = nb_reads_clustered / nb_reads if nb_reads > 0 else 0
    percent_positions_used = nb_positions_used / nb_positions if nb_positions > 0 else 0
    nb_reads_per_cluster = [len(cluster) for cluster in clusters]
    nb_positions_per_strip = [len(strip[2]) for strip in steps]

    metrics = {
        "nb_clusters": nb_clusters,
        "nb_inhomogenious_regions": len(inhomogenious_regions),
        "nb_steps_pre_processing": len(steps_pre_processing),
        "nb_steps_ilp": len(steps_ilp),
        "nb_reads_per_cluster": nb_reads_per_cluster,
        "nb_positions_per_strip": nb_positions_per_strip,
        "orphan_reads_names": orphan_reads_names,
        "unused_columns": unused_columns,
        "original_matrix_shape": (nb_reads, nb_positions),
        "reduced_matrix_shape": reduced_shape,
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
        "list_of_steps_pre_processing": steps_pre_processing,
        "list_of_steps_ilp": steps_ilp,
    }
    metrics["time_load"] = load_time
    metrics["time_pre_processing"] = pre_processing_time
    metrics["time_ilp"] = ilp_time
    metrics["time_post_processing"] = post_processing_time
    metrics["name"] = "pipeline_compact_matrix"
    metrics["input_csv"] = input_csv
    metrics["output_txt"] = output_txt
    metrics["output_csv"] = output_csv
    metrics["min_col_quality"] = min_col_quality
    metrics["min_row_quality"] = min_row_quality
    metrics["error_rate"] = error_rate
    metrics["distance_thresh_post"] = distance_thresh_post
    # Ajout des noms de reads pour chaque cluster, formaté joliment
    for i, cluster in enumerate(clusters):
        read_names_list = [str(name) for name in cluster]
        metrics[f"read_{i}"] = read_names_list
    for i, strip in enumerate(steps):
        metrics[f"strip_{i}"] = list(strip[2])
    # Ajout des métriques ILP
    if metrics_ilp:
        metrics.update(metrics_ilp)
    save_dict_with_metadata(metrics, output_txt)
    return metrics 


def pipeline_ilp_largest_only(
    input_csv,
    output_txt,
    error_rate=0.025
) -> dict:
    """
    Run a minimal pipeline that only computes the largest quasi-biclique via ILP and returns metrics.

    Description
    - Loads the matrix from input_csv.
    - Calls largest_only(...) to find the single largest dense submatrix (quasi-biclique).
    - Saves timing and ILP metrics to output_txt and returns the metrics dict.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV matrix.
    output_txt : str
        Path where resulting metrics and metadata will be saved.
    error_rate : float, optional
        Tolerance for the density check in largest_only (default: 0.025).

    Returns
    -------
    dict
        Metrics returned by largest_only augmented with timings, input/output info and a name field.

    Example
    -------
    metrics = pipeline_ilp_largest_only("reads.csv", "largest_metrics.txt", error_rate=0.02)

    Notes
    -----
    - Intended for quick inspection or benchmarking: it does not perform pre/post-processing.
    - The returned metrics contain the row/column indices of the detected submatrix and density info.
    """
    # 1. Lecture du CSV
    start_time = time.time()
    matrix, read_names = load_csv_matrix(input_csv)
    check1_time = time.time()
    load_time = check1_time - start_time
    # 2. ILP
    res , metrics = largest_only(matrix, error_rate=error_rate)
    check2_time = time.time()
    ilp_time = check2_time - check1_time
    metrics["time_load"] = load_time
    metrics["time_ilp"] = ilp_time
    metrics["time_total"] = ilp_time + load_time
    metrics["name"] = "pipeline_ilp_largest_only"
    metrics["input_csv"] = input_csv
    # 3. Sauvegarde
    save_dict_with_metadata(metrics, output_txt)
    return metrics