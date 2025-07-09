#!/usr/bin/env python3
"""
Script simple pour analyser les résultats de test MatrixStriper.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plot.analyze_results import ResultsAnalyzer

def main():
    """Fonction principale."""
    # Vérifier si le fichier test_results.csv existe
    csv_file = "test_results.csv"
    
    if not Path(csv_file).exists():
        print(f"Erreur: Le fichier {csv_file} n'existe pas!")
        print("Assurez-vous d'avoir exécuté le test batch d'abord.")
        return
    
    print("=== ANALYSE DES RÉSULTATS MATRIXSTRIPER ===\n")
    
    # Lancer l'analyse
    analyzer = ResultsAnalyzer(csv_file)
    analyzer.run_complete_analysis("analysis_results")
    
    print("\n=== ANALYSE TERMINÉE ===")
    print("Consultez le dossier 'analysis_results' pour voir tous les résultats.")

if __name__ == "__main__":
    main() 