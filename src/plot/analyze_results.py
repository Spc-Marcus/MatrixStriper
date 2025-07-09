#!/usr/bin/env python3
"""
Script d'analyse et de visualisation des résultats de test batch MatrixStriper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ResultsAnalyzer:
    def __init__(self, csv_path):
        """
        Initialise l'analyseur avec le fichier CSV des résultats.
        
        Args:
            csv_path (str): Chemin vers le fichier CSV des résultats
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Charge et prépare les données."""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Données chargées: {len(self.df)} matrices")
            print(f"Colonnes disponibles: {list(self.df.columns)}")
            
            # Conversion des colonnes de forme de matrice
            self.df['original_rows'] = self.df['original_matrix_shape'].str.extract(r'\((\d+),').astype(int)
            self.df['original_cols'] = self.df['original_matrix_shape'].str.extract(r', (\d+)\)').astype(int)
            self.df['reduced_rows'] = self.df['reduced_matrix_shape'].str.extract(r'\((\d+),').astype(int)
            self.df['reduced_cols'] = self.df['reduced_matrix_shape'].str.extract(r', (\d+)\)').astype(int)
            
            # Calcul de la taille totale des matrices
            self.df['original_size'] = self.df['original_rows'] * self.df['original_cols']
            self.df['reduced_size'] = self.df['reduced_rows'] * self.df['reduced_cols']
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            raise
    
    def basic_statistics(self):
        """Affiche les statistiques de base."""
        print("\n=== STATISTIQUES DE BASE ===")
        print(f"Nombre total de matrices: {len(self.df)}")
        print(f"Nombre d'haplotypes testés: {self.df['nb_haplotypes'].unique()}")
        
        print("\nRépartition par nombre d'haplotypes:")
        haplotype_counts = self.df['nb_haplotypes'].value_counts().sort_index()
        for haplotype, count in haplotype_counts.items():
            print(f"  {haplotype} haplotypes: {count} matrices")
        
        print(f"\nTemps d'exécution total: {self.df['execution_time'].sum():.2f} secondes")
        print(f"Temps d'exécution moyen: {self.df['execution_time'].mean():.3f} secondes")
        print(f"Temps d'exécution médian: {self.df['execution_time'].median():.3f} secondes")
        
        print(f"\nTaille moyenne des matrices originales: {self.df['original_size'].mean():.0f} éléments")
        print(f"Taille moyenne des matrices réduites: {self.df['reduced_size'].mean():.0f} éléments")
    
    def plot_execution_time_by_haplotype(self, save_path=None):
        """Graphique du temps d'exécution par nombre d'haplotypes."""
        plt.figure(figsize=(12, 6))
        
        # Box plot
        plt.subplot(1, 2, 1)
        ax1 = sns.boxplot(data=self.df, x='nb_haplotypes', y='execution_time', color='#4C72B0')
        plt.title('Temps d\'exécution par nombre d\'haplotypes')
        plt.xlabel('Nombre d\'haplotypes')
        plt.ylabel('Temps d\'exécution (secondes)')
        # Annotation moyenne
        means = self.df.groupby('nb_haplotypes')['execution_time'].mean()
        for xtick, mean in enumerate(means):
            plt.text(xtick, mean, f"{mean:.2f}", color='red', ha='center', va='bottom', fontweight='bold')
        
        # Violin plot
        plt.subplot(1, 2, 2)
        ax2 = sns.violinplot(data=self.df, x='nb_haplotypes', y='execution_time', color='#55A868')
        plt.title('Distribution du temps d\'exécution')
        plt.xlabel('Nombre d\'haplotypes')
        plt.ylabel('Temps d\'exécution (secondes)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clustering_results(self, save_path=None):
        """Graphique des résultats de clustering."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Nombre de clusters finaux
        axes[0, 0].hist(self.df['nb_clusters_final'], bins=20, alpha=0.8, edgecolor='black', color='#4C72B0')
        axes[0, 0].set_title('Distribution du nombre de clusters finaux')
        axes[0, 0].set_xlabel('Nombre de clusters finaux')
        axes[0, 0].set_ylabel('Nombre de matrices')
        
        # Nombre de steps de pre-processing
        axes[0, 1].hist(self.df['nb_steps_pre_processing'], bins=20, alpha=0.8, edgecolor='black', color='#55A868')
        axes[0, 1].set_title('Distribution du nombre de steps de pré-traitement')
        axes[0, 1].set_xlabel('Nombre de steps de pré-traitement')
        axes[0, 1].set_ylabel('Nombre de matrices')
        
        # Pourcentage de reads clusterisés
        percent_reads = self.df['percent_reads_clustered'] * 100
        axes[1, 0].hist(percent_reads, bins=20, alpha=0.8, edgecolor='black', color='#C44E52')
        axes[1, 0].set_title('Distribution du pourcentage de reads clusterisés')
        axes[1, 0].set_xlabel('Pourcentage de reads clusterisés (%)')
        axes[1, 0].set_ylabel('Nombre de matrices')
        
        # Pourcentage de positions utilisées
        percent_pos = self.df['percent_positions_used'] * 100
        axes[1, 1].hist(percent_pos, bins=20, alpha=0.8, edgecolor='black', color='#8172B2')
        axes[1, 1].set_title('Distribution du pourcentage de positions utilisées')
        axes[1, 1].set_xlabel('Pourcentage de positions utilisées (%)')
        axes[1, 1].set_ylabel('Nombre de matrices')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_compression_analysis(self, save_path=None):
        """Analyse de la compression."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ratio de compression
        axes[0, 0].hist(self.df['compression_ratio'], bins=30, alpha=0.8, edgecolor='black', color='#4C72B0')
        axes[0, 0].set_title('Distribution du ratio de compression')
        axes[0, 0].set_xlabel('Ratio de compression')
        axes[0, 0].set_ylabel('Nombre de matrices')
        mean = self.df['compression_ratio'].mean()
        median = self.df['compression_ratio'].median()
        axes[0, 0].axvline(mean, color='red', linestyle='--', label=f'Moyenne: {mean:.4f}')
        axes[0, 0].axvline(median, color='green', linestyle=':', label=f'Médiane: {median:.4f}')
        axes[0, 0].legend()
        
        # Taille des matrices vs compression
        axes[0, 1].scatter(self.df['original_size'], self.df['compression_ratio'], alpha=0.6, color='#55A868')
        axes[0, 1].set_title('Ratio de compression vs Taille de matrice originale')
        axes[0, 1].set_xlabel('Taille de matrice originale (nb éléments)')
        axes[0, 1].set_ylabel('Ratio de compression')
        
        # Nombre d'haplotypes vs compression
        box_data = [self.df[self.df['nb_haplotypes'] == h]['compression_ratio'] for h in sorted(self.df['nb_haplotypes'].unique())]
        axes[1, 0].boxplot(box_data, labels=sorted(self.df['nb_haplotypes'].unique()), patch_artist=True, boxprops=dict(facecolor='#C44E52'))
        axes[1, 0].set_title('Ratio de compression par nombre d\'haplotypes')
        axes[1, 0].set_xlabel('Nombre d\'haplotypes')
        axes[1, 0].set_ylabel('Ratio de compression')
        
        # Reads orphelins vs colonnes inutilisées
        axes[1, 1].scatter(self.df['nb_unused_columns'], self.df['nb_orphan_reads'], alpha=0.6, color='#8172B2')
        axes[1, 1].set_title('Reads orphelins vs Colonnes inutilisées')
        axes[1, 1].set_xlabel('Nombre de colonnes inutilisées')
        axes[1, 1].set_ylabel('Nombre de reads orphelins')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_haplotype_comparison(self, save_path=None):
        """Comparaison détaillée par nombre d'haplotypes."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        haplotype_groups = self.df.groupby('nb_haplotypes')
        labels = sorted(self.df['nb_haplotypes'].unique())
        # Temps d'exécution
        execution_times = [group['execution_time'].values for _, group in haplotype_groups]
        axes[0, 0].boxplot(execution_times, labels=labels, patch_artist=True, boxprops=dict(facecolor='#4C72B0'))
        axes[0, 0].set_title('Temps d\'exécution par haplotype')
        axes[0, 0].set_ylabel('Temps (secondes)')
        # Nombre de clusters
        cluster_counts = [group['nb_clusters_final'].values for _, group in haplotype_groups]
        axes[0, 1].boxplot(cluster_counts, labels=labels, patch_artist=True, boxprops=dict(facecolor='#55A868'))
        axes[0, 1].set_title('Nombre de clusters finaux par haplotype')
        axes[0, 1].set_ylabel('Nombre de clusters')
        # Nombre de steps
        step_counts = [group['nb_steps_pre_processing'].values for _, group in haplotype_groups]
        axes[0, 2].boxplot(step_counts, labels=labels, patch_artist=True, boxprops=dict(facecolor='#C44E52'))
        axes[0, 2].set_title('Nombre de steps de pré-traitement par haplotype')
        axes[0, 2].set_ylabel('Nombre de steps')
        # Ratio de compression
        compression_ratios = [group['compression_ratio'].values for _, group in haplotype_groups]
        axes[1, 0].boxplot(compression_ratios, labels=labels, patch_artist=True, boxprops=dict(facecolor='#8172B2'))
        axes[1, 0].set_title('Ratio de compression par haplotype')
        axes[1, 0].set_ylabel('Ratio de compression')
        # Pourcentage de reads clusterisés
        read_percentages = [group['percent_reads_clustered'].values * 100 for _, group in haplotype_groups]
        axes[1, 1].boxplot(read_percentages, labels=labels, patch_artist=True, boxprops=dict(facecolor='#937860'))
        axes[1, 1].set_title('Pourcentage de reads clusterisés par haplotype')
        axes[1, 1].set_ylabel('Pourcentage de reads clusterisés (%)')
        # Pourcentage de positions utilisées
        pos_percentages = [group['percent_positions_used'].values * 100 for _, group in haplotype_groups]
        axes[1, 2].boxplot(pos_percentages, labels=labels, patch_artist=True, boxprops=dict(facecolor='#DA8BC3'))
        axes[1, 2].set_title('Pourcentage de positions utilisées par haplotype')
        axes[1, 2].set_ylabel('Pourcentage de positions utilisées (%)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, save_path=None):
        """Matrice de corrélation des variables numériques."""
        numeric_cols = [
            'nb_haplotypes', 'nb_steps_pre_processing', 'nb_clusters_final',
            'execution_time', 'nb_unused_columns', 'nb_orphan_reads',
            'compression_ratio', 'percent_reads_clustered', 'percent_positions_used',
            'original_size', 'reduced_size'
        ]
        corr = self.df[numeric_cols].corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Matrice de corrélation des variables numériques')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, output_path=None):
        """Génère un rapport de synthèse."""
        report = []
        report.append("=== RAPPORT D'ANALYSE DES RÉSULTATS MATRIXSTRIPER ===\n")
        
        # Statistiques générales
        report.append("1. STATISTIQUES GÉNÉRALES")
        report.append(f"   - Nombre total de matrices: {len(self.df)}")
        report.append(f"   - Haplotypes testés: {sorted(self.df['nb_haplotypes'].unique())}")
        report.append(f"   - Temps total d'exécution: {self.df['execution_time'].sum():.2f} secondes")
        report.append(f"   - Temps moyen par matrice: {self.df['execution_time'].mean():.3f} secondes\n")
        
        # Analyse par haplotype
        report.append("2. ANALYSE PAR NOMBRE D'HAPLOTYPES")
        for haplotype in sorted(self.df['nb_haplotypes'].unique()):
            subset = self.df[self.df['nb_haplotypes'] == haplotype]
            report.append(f"   {haplotype} haplotypes ({len(subset)} matrices):")
            report.append(f"     - Temps moyen: {subset['execution_time'].mean():.3f}s")
            report.append(f"     - Clusters moyens: {subset['nb_clusters_final'].mean():.1f}")
            report.append(f"     - Steps moyens: {subset['nb_steps_pre_processing'].mean():.1f}")
            report.append(f"     - Compression moyenne: {subset['compression_ratio'].mean():.4f}")
            report.append(f"     - Reads clusterisés: {subset['percent_reads_clustered'].mean():.1%}")
            report.append(f"     - Positions utilisées: {subset['percent_positions_used'].mean():.1%}\n")
        
        # Performances
        report.append("3. PERFORMANCES")
        report.append(f"   - Ratio de compression moyen: {self.df['compression_ratio'].mean():.4f}")
        report.append(f"   - Pourcentage moyen de reads clusterisés: {self.df['percent_reads_clustered'].mean():.1%}")
        report.append(f"   - Pourcentage moyen de positions utilisées: {self.df['percent_positions_used'].mean():.1%}")
        report.append(f"   - Nombre moyen de clusters finaux: {self.df['nb_clusters_final'].mean():.1f}")
        report.append(f"   - Nombre moyen de steps de pre-processing: {self.df['nb_steps_pre_processing'].mean():.1f}\n")
        
        # Observations
        report.append("4. OBSERVATIONS CLÉS")
        report.append(f"   - Matrices avec 0 steps: {len(self.df[self.df['nb_steps_pre_processing'] == 0])} ({len(self.df[self.df['nb_steps_pre_processing'] == 0])/len(self.df)*100:.1f}%)")
        report.append(f"   - Matrices avec 1 cluster final: {len(self.df[self.df['nb_clusters_final'] == 1])} ({len(self.df[self.df['nb_clusters_final'] == 1])/len(self.df)*100:.1f}%)")
        report.append(f"   - Matrices avec compression parfaite (ratio=0): {len(self.df[self.df['compression_ratio'] == 0])} ({len(self.df[self.df['compression_ratio'] == 0])/len(self.df)*100:.1f}%)")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Rapport sauvegardé dans: {output_path}")
        
        print(report_text)
        return report_text
    
    def run_complete_analysis(self, output_dir="analysis_results"):
        """Lance l'analyse complète et sauvegarde tous les résultats."""
        # Créer le dossier de sortie
        Path(output_dir).mkdir(exist_ok=True)
        
        print("=== DÉBUT DE L'ANALYSE COMPLÈTE ===\n")
        
        # Statistiques de base
        self.basic_statistics()
        
        # Générer le rapport
        self.generate_summary_report(f"{output_dir}/summary_report.txt")
        
        # Graphiques
        print("\nGénération des graphiques...")
        self.plot_execution_time_by_haplotype(f"{output_dir}/execution_time.png")
        self.plot_clustering_results(f"{output_dir}/clustering_results.png")
        self.plot_compression_analysis(f"{output_dir}/compression_analysis.png")
        self.plot_haplotype_comparison(f"{output_dir}/haplotype_comparison.png")
        self.plot_correlation_matrix(f"{output_dir}/correlation_matrix.png")
        
        print(f"\nAnalyse terminée! Résultats sauvegardés dans: {output_dir}/")

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse des résultats de test MatrixStriper')
    parser.add_argument('csv_file', help='Fichier CSV des résultats')
    parser.add_argument('--output-dir', default='analysis_results', 
                       help='Dossier de sortie pour les résultats (default: analysis_results)')
    
    args = parser.parse_args()
    
    # Vérifier que le fichier existe
    if not Path(args.csv_file).exists():
        print(f"Erreur: Le fichier {args.csv_file} n'existe pas!")
        return
    
    # Lancer l'analyse
    analyzer = ResultsAnalyzer(args.csv_file)
    analyzer.run_complete_analysis(args.output_dir)

if __name__ == "__main__":
    main() 