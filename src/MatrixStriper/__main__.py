import argparse
import logging
from .pipeline import compact_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de biclustering et compactage de matrice.")
    parser.add_argument("input_csv", type=str, help="Chemin du fichier CSV d'entrée.")
    parser.add_argument("output_csv", type=str, help="Chemin du fichier CSV de sortie.")
    parser.add_argument("--largest_only", action="store_true", help="Ne conserve que la plus grande sous-matrice.")
    parser.add_argument("--min_col_quality", type=int, default=3, help="Qualité minimale des colonnes.")
    parser.add_argument("--min_row_quality", type=int, default=5, help="Qualité minimale des lignes.")
    parser.add_argument("--error_rate", type=float, default=0.025, help="Taux d'erreur toléré.")
    parser.add_argument("--distance_thresh", type=float, default=0.1, help="Seuil de distance de Hamming pour fusionner.")
    parser.add_argument("--certitude", type=float, default=0.3, help="Seuil de certitude pour la binarisation.")
    parser.add_argument("--debug", type=int, default=0, choices=[0,1,2], help="Niveau de debug: 0=WARNING, 1=INFO, 2=DEBUG")
    args = parser.parse_args()

    # Setup logger
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(level=log_levels.get(args.debug, logging.WARNING), format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("MatrixStriper")

    logger.info("Début du pipeline MatrixStriper")
    logger.debug(f"Arguments: {args}")

    metrics = compact_matrix(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        largest_only=args.largest_only,
        min_col_quality=args.min_col_quality,
        min_row_quality=args.min_row_quality,
        error_rate=args.error_rate,
        distance_thresh=args.distance_thresh
    )

    logger.info("Pipeline terminé.")
    logger.info(f"Métriques de compression : {metrics}")
    print("Métriques de compression :", metrics) 