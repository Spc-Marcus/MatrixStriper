import pytest
import numpy as np
from src.MatrixStriper.post_processing import post_processing

class TestPostProcessing:
    def test_empty_matrix(self):
        """
        Vérifie que la fonction gère correctement une matrice vide.
        """
        matrix = np.empty((0, 0))
        steps = []
        read_names = []
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names)
        assert clusters == []
        assert reduced.size == 0
        assert orphan_reads == []
        assert unused_columns == []

    def test_single_read(self):
        """
        Cas limite : une seule lecture, un seul cluster.
        """
        matrix = np.array([[1, 0, 1]])
        steps = []
        read_names = ['read1']
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names,min_reads_per_cluster=None)
        assert len(clusters) == 1
        assert clusters[0][0] == 'read1'
        assert reduced.size == 0
        assert orphan_reads == []
        assert unused_columns == [0, 1, 2]

    def test_no_split(self):
        """
        Aucun step : tous les reads dans un seul cluster.
        """
        matrix = np.array([[1, 0], [0, 1], [1, 0]])
        steps = []
        read_names = ['r1', 'r2', 'r3']
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names,min_reads_per_cluster=None)
        assert len(clusters) == 1
        assert set(clusters[0]) == {'r1', 'r2', 'r3'}
        assert reduced.size == 0
        assert orphan_reads == []
        assert unused_columns == [0, 1]

    def test_simple_split(self):
        """
        Un step qui sépare les reads en deux clusters.
        """
        matrix = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        steps = [([0, 2], [1, 3], [0, 1])]
        read_names = ['a', 'b', 'c', 'd']
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names, min_reads_per_cluster=None)
        assert len(clusters) == 2
        all_reads = set(np.concatenate(clusters))
        assert all_reads == {'a', 'b', 'c', 'd'}
        assert reduced.shape == (2, 1)
        assert orphan_reads == []
        assert unused_columns == []

    def test_min_reads_filtering(self):
        """
        Teste le filtrage des petits clusters.
        """
        matrix = np.array([[0, 0], [1, 1], [0, 0], [1, 1], [1, 1]])
        steps = [([0, 2], [1, 3, 4], [0, 1])]
        read_names = ['a', 'b', 'c', 'd', 'e']
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names, min_reads_per_cluster=3)
        # Un cluster de taille 2 doit être filtré
        assert all(len(clust) >= 3 for clust in clusters)
        assert reduced.shape[0] == 1
        assert reduced.shape[1] == 1
        assert set(orphan_reads) == {'a','c'}
        assert unused_columns == []

    def test_orphan_reassignment(self):
        """
        Teste la réaffectation d'un read orphelin au cluster le plus proche si la distance est inférieure à 0.3.
        """
        matrix = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0],[0, 0, 1]])
        steps = [([0, 2], [1], [0, 1, 2])]
        read_names = ['a', 'b', 'c', 'd']
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names, min_reads_per_cluster=None, distance_thresh=0.5)
        # Le read 'b' doit être réaffecté à un cluster
        all_reads = set(np.concatenate(clusters))
        assert 'd' in all_reads
        assert reduced.shape[0] == 2
        assert orphan_reads == []
        assert unused_columns == []

    def test_merge_clusters(self):
        """
        Teste la fusion de clusters similaires (distance_thresh élevée).
        """
        matrix = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        steps = [([0, 1], [2, 3], [0, 1])]
        read_names = ['a', 'b', 'c', 'd']
        # Avec un seuil élevé, tout doit fusionner
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names, distance_thresh=1.0, min_reads_per_cluster=None)
        assert len(clusters) == 1
        assert set(clusters[0]) == {'a', 'b', 'c', 'd'}
        assert orphan_reads == []
        assert unused_columns == []

    def test_return_types(self):
        """
        Vérifie les types de retour.
        """
        matrix = np.random.randint(0, 2, size=(6, 4))
        steps = [([0, 1, 2], [3, 4, 5], [0, 1, 2, 3])]
        read_names = [f'read{i}' for i in range(6)]
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names)
        assert isinstance(clusters, list)
        assert all(isinstance(cl, np.ndarray) for cl in clusters)
        assert isinstance(reduced, np.ndarray)
        assert isinstance(orphan_reads, list)
        assert isinstance(unused_columns, list)
    
    def test_reduce_matrix(self):
        """
        Teste la réduction de la matrice.
        """
        matrix = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
        ], dtype=int)
        steps =[
            ([0,1,2,3,4,5],[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[0,1,2,3,18]),
            ([0,1,2,3,4,5,6,7,8,9,10,11,12,13],[14,15,16,17,18,19,20,21],[4,5,6,7,8,9,21]),
            ([14,15,16,17,18,19,20,21],[0,1,2,3,4,5,6,7,8,9,10,11,12,13],[10,11,12,13,14,15,16,22]),
            ([],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[17,19,20,23,24,25])
        ]
        read_names = [f'read{i}' for i in range(23)]
        clusters, reduced, orphan_reads, unused_columns = post_processing(matrix, steps, read_names, min_reads_per_cluster=None)
        assert len(clusters) == 3
        assert reduced.shape == (len(clusters), len(steps))
        assert unused_columns == [26]
        assert orphan_reads == ['read22']