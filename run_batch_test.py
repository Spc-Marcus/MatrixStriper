#!/usr/bin/env python3
"""
Script principal pour exécuter les tests de batch sur les matrices.
Usage: python run_batch_test.py [options]
"""

import sys
import os
import argparse
import logging

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from MatrixStriper.test_pipeline import run_batch_test

def main():
    parser = argparse.ArgumentParser(description='Test batch du pipeline MatrixStriper (pre-processing + post-processing)')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Dossier contenant les sous-dossiers d\'haplotypes (default: data)')
    
    parser.add_argument('--output', type=str, default='test_results.csv',
                       help='Fichier de sortie pour les résultats (default: test_results.csv)')
    
    parser.add_argument('--min-col-quality', type=int, default=3,
                       help='Qualité minimale des colonnes (default: 3)')
    
    parser.add_argument('--min-row-quality', type=int, default=5,
                       help='Qualité minimale des lignes (default: 5)')
    
    parser.add_argument('--error-rate', type=float, default=0.025,
                       help='Taux d\'erreur toléré (default: 0.025)')
    
    parser.add_argument('--distance-thresh', type=float, default=0.1,
                       help='Seuil de distance de Hamming pour fusionner (default: 0.1)')
    
    parser.add_argument('--min-reads-per-cluster', type=int, default=5,
                       help='Nombre minimum de reads par cluster (default: 5)')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Niveau de logging (default: INFO)')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Vérifier que le dossier data existe
    if not os.path.exists(args.data_dir):
        logger.error(f"Le dossier {args.data_dir} n'existe pas!")
        sys.exit(1)
    
    logger.info("Démarrage des tests de batch")
    logger.info(f"Dossier de données: {args.data_dir}")
    logger.info(f"Fichier de sortie: {args.output}")
    logger.info(f"Paramètres: min_col_quality={args.min_col_quality}, "
                f"min_row_quality={args.min_row_quality}, error_rate={args.error_rate}, "
                f"distance_thresh={args.distance_thresh}, min_reads_per_cluster={args.min_reads_per_cluster}")
    
    try:
        # Lancer les tests
        results_df = run_batch_test(
            data_dir=args.data_dir,
            output_csv=args.output,
            min_col_quality=args.min_col_quality,
            min_row_quality=args.min_row_quality,
            error_rate=args.error_rate,
            distance_thresh=args.distance_thresh,
            min_reads_per_cluster=args.min_reads_per_cluster
        )
        
        logger.info(f"Tests terminés avec succès!")
        logger.info(f"Nombre de matrices traitées: {len(results_df)}")
        logger.info(f"Résultats sauvegardés dans: {args.output}")
        
        # Afficher un résumé des résultats
        if len(results_df) > 0:
            print("\n=== RÉSUMÉ DES RÉSULTATS ===")
            print(f"Nombre total de matrices: {len(results_df)}")
            
            # Statistiques par nombre d'haplotypes
            haplotype_stats = results_df.groupby('nb_haplotypes').agg({
                'nb_steps_pre_processing': ['mean', 'std'],
                'nb_clusters_final': ['mean', 'std'],
                'execution_time': ['mean', 'std'],
                'nb_unused_columns': ['mean', 'std'],
                'nb_orphan_reads': ['mean', 'std']
            }).round(2)
            
            print("\nStatistiques par nombre d'haplotypes:")
            print(haplotype_stats)
            
            # Temps total d'exécution
            total_time = results_df['execution_time'].sum()
            print(f"\nTemps total d'exécution: {total_time:.2f} secondes")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 