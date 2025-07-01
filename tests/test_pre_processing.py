import pytest
import numpy as np
from src.MatrixStriper.pre_processing import pre_processing, hamming_distance_matrix, is_strip
import logging
from sklearn.cluster import AgglomerativeClustering


class TestPreProcessing:
    def test_empty_matrix(self):
        """
        Vérifie que la fonction gère correctement une matrice vide.
        - Doit retourner la structure attendue.
        """
        matrix = np.array([[]])
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert inhomogeneous_regions == []
        assert steps == []

    def test_single_element_matrix(self):
        """
        Teste le cas limite d'une matrice 1x1.
        - Les régions doivent contenir l'unique colonne.
        - Aucun step ne doit être généré.
        """
        matrix = np.array([[1]])
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert inhomogeneous_regions == [0]
        assert steps == []

    def test_small_matrix_no_clustering(self):
        """
        Vérifie qu'aucun clustering n'est effectué pour une petite matrice.
        - Une seule région doit être détectée.
        - Aucun step ne doit être généré.
        """
        matrix = np.array([[1, 0], [0, 1], [1, 0]])
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=5)
        assert sorted(inhomogeneous_regions) == [0, 1]
        assert steps == []

    def test_error_rate_validation(self):
        """
        Vérifie la validation du paramètre certitude.
        - Doit lever une ValueError pour des valeurs hors [0, 0.5).
        """
        matrix = np.array([[1, 0]])
        with pytest.raises(ValueError):
            pre_processing(matrix, error_rate=-0.1)
        with pytest.raises(ValueError):
            pre_processing(matrix, error_rate=0.5)
        with pytest.raises(ValueError):
            pre_processing(matrix, error_rate=1.0)

    def test_homogeneous_region_identification(self):
        """
        Vérifie l'identification de régions homogènes (clustering en deux groupes bien séparés).
        - Les steps doivent contenir des clusters bien définis.
        - Chaque step doit être un tuple (cluster1, cluster0, region).
        - Les régions doivent être bien définies.

        """
        matrix = np.zeros((15, 20), dtype=int)
        matrix[:7, :10] = 1
        matrix[7:, 10:] = 1
        inhomogeneous_regions, steps = pre_processing(matrix)
        for step in steps:
            assert len(step) == 3
            cluster1, cluster0, region = step
            assert isinstance(cluster1, list)
            assert isinstance(cluster0, list)
            assert all(i not in cluster0 for i in cluster1)
            assert all(i not in cluster1 for i in cluster0)
            assert isinstance(region, list)
        used = set()
        for _, _, region in steps:
            used.update(region)
        all_cols = set(range(matrix.shape[1]))
        expected = sorted(list(all_cols - used))
        assert sorted(inhomogeneous_regions) == expected

    def test_heterogeneous_region_identification(self):
        """
        Vérifie l'identification de régions hétérogènes (mélange de 0 et 1 sans séparation franche).
        - Doit détecter au moins une région inhomogène ou un step.
        """
        matrix = np.zeros((15, 20), dtype=int)
        for j in range(20):
            for i in range(15):
                matrix[i, j] = (i + j) % 2
        inhomogeneous_regions, steps = pre_processing(matrix)
        total_regions_processed = len(inhomogeneous_regions) + sum(len(region) for _, _, region in steps)
        assert total_regions_processed > 0

    def test_large_region_subdivision(self):
        """
        Vérifie la subdivision des grandes régions (>15 colonnes).
        - Tous les éléments doivent rester binaires.
        """
        matrix = np.ones((10, 25), dtype=int)
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert inhomogeneous_regions == []
        assert len(steps) == 1
        assert len(steps[0][1]) == 10
        assert len(steps[0][2]) == 25

    
    def test_return_types(self):
        """
        Vérifie que les types de retour sont corrects et compatibles pour la suite du pipeline.
        - inhomogeneous_regions une liste d'entiers
        - steps une liste de tuples (cluster1, cluster0, region)
        """
        matrix = np.random.randint(0, 2, size=(10, 16))
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert isinstance(inhomogeneous_regions, list)
        assert isinstance(steps, list)
        assert all(isinstance(i, int) for i in inhomogeneous_regions)
        for step in steps:
            assert isinstance(step, tuple)
            assert len(step) == 3

    def test_pre_processing_with_default_values(self):
        """
        Vérifie que la fonction gère correctement les valeurs par défaut.
        """
        matrix = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        inhomogeneous_regions, steps = pre_processing(matrix)
        # With default min_col_quality=5, all columns should be inhomogeneous
        assert sorted(inhomogeneous_regions) == [0, 1, 2]
        assert steps == []
    
    def test_pre_processing_with_min_col_quality(self):
        """
        Vérifie que la fonction gère correctement le paramètre min_col_quality.
        """
        matrix = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=2)
        # The result depends on clustering, but should be valid
        assert inhomogeneous_regions == [1]
        assert len(steps) == 1
        assert len(steps[0][1]) == 3
        assert len(steps[0][2]) == 2
    
    def test_pre_processing_invalid_argument(self):
        """
        Vérifie que la fonction lève une exception ou gère proprement un argument d'entrée non valide (ex: tableau 1D).
        - Doit lever une exception ou retourner une structure attendue sans planter.
        """
        # Cas d'un tableau 1D (pas de shape (m, n))
        arr = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            pre_processing(arr)
    
    def test_inhomogeneous_region_threshold(self):
        """
        Verifie que la fonction gère correctement les seuils d'inhomogénéité.
        - Doit retourner une liste d'inhomogénéités.
        - Doit retourner une steps.
        """
        matrix = np.array([
            [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1],
            [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1],
            [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1],
            [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,1,1],
            [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1],
            [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1],
            [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1],
            [1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1],
            [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1],
            [1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,1,1,1],
            [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1],
            [1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1],
            [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1],
            [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,1,1,1],
            [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,1,1],
            [1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0],
            [1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,0,0],
            [0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0],
        ], dtype=int)
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=3)
        assert inhomogeneous_regions == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        assert len(steps) == 1
        assert sorted(steps[0][2]) == [17, 18, 19]
        assert sorted(steps[0][1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        assert sorted(steps[0][0]) == [16, 17, 18, 19]

    
    def test_hamming_distance_matrix(self):
        """
        Teste la fonction hamming_distance_matrix.
        """
        matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        distance_matrix = hamming_distance_matrix(matrix)
        assert distance_matrix.shape == (3, 3)
        assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetric
        assert np.allclose(np.diag(distance_matrix), 0)  # Diagonal is 0

    def test_is_strip_clear_pattern(self):
        """
        Teste la fonction is_strip avec un pattern clair de strip.
        """
        # Create a clear strip pattern: first 5 rows have 1s in first 3 columns, rest have 0s
        matrix = np.zeros((10, 5))
        matrix[:5, :3] = 1
        matrix[5:, 3:] = 1
        
        is_strip_result, strip_info = is_strip(matrix, [0, 1, 2], error_rate=0.3)
        assert is_strip_result
        assert strip_info is not None
        cluster1, cluster0, columns = strip_info
        assert columns == [0, 1, 2]
        assert len(cluster1) == 5
        assert len(cluster0) == 5
        

    def test_is_strip_no_pattern(self):
        """
        Teste la fonction is_strip avec un pattern qui n'est pas un strip.
        """
        # Create a random pattern that shouldn't form a strip
        np.random.seed(42)
        matrix = np.random.choice([0, 1], size=(10, 5))
        
        is_strip_result, strip_info = is_strip(matrix, [0, 1, 2], error_rate=0.3)
        assert not is_strip_result
        assert strip_info is None

    def test_is_strip_single_column(self):
        """
        Teste la fonction is_strip avec une seule colonne.
        """
        matrix = np.array([[1], [0], [1], [0]])
        is_strip_result, strip_info = is_strip(matrix, [0], error_rate=0.3)
        assert not is_strip_result
        assert strip_info is None

    def test_matrix_with_all_ones(self):
        """
        Teste une matrice remplie de 1s.
        Avec petite erreur.
        """
        matrix = np.ones((10, 15))
        matrix[0,5] = 0
        matrix[8,8] = 0
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert inhomogeneous_regions == []
        assert steps[0][0] == []
        assert sorted(steps[0][1]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert sorted(steps[0][2]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def test_matrix_with_all_zeros(self):
        """
        Teste une matrice remplie de 0s.
        Avec petite erreur.
        """
        matrix = np.zeros((10, 15))
        matrix[0, 0] = 1
        inhomogeneous_regions, steps = pre_processing(matrix)
        assert inhomogeneous_regions == []
        assert steps[0][0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert sorted(steps[0][1]) == []
        assert sorted(steps[0][2]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def test_perfect_strip_pattern(self):
        """
        Teste un pattern de strip parfait.
        """
        # Create a perfect strip: first half of rows have 1s in first half of columns
        matrix = np.zeros((20, 20))
        matrix[:10, :10] = 1
        matrix[10:, 10:] = 1
        
        inhomogeneous_regions, steps = pre_processing(matrix, error_rate=0.3)
        # Should identify strips
        assert len(steps) == 2
        for step in steps:
            cluster1, cluster0, columns = step
            assert len(cluster1) == 10
            assert len(cluster0) == 10
            assert len(columns) == 10

    def test_clear_strip(self):
        """
        Matrix with a clear strip: first 5 rows, first 3 columns are 1, rest are 0.
        Should identify a strip and no ambiguous columns.
        """
        
        matrix = np.zeros((10, 5), dtype=int)
        matrix[:5, :3] = 1
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=3, error_rate=0.3)
        print('matrix:', matrix)
        print('inhomogeneous_regions:', inhomogeneous_regions)
        print('steps:', steps)
        # Should find a strip in columns [0,1,2]
        assert set(inhomogeneous_regions) == {3, 4}
        assert len(steps) == 1
        assert set(steps[0][2]) == {0, 1, 2}

    def test_all_ambiguous(self):
        """
        Matrix with no clear strip: random 0/1, should return all columns as ambiguous.
        """
        np.random.seed(0)
        matrix = np.random.randint(0, 2, size=(8, 4))
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=2, error_rate=0.3)
        assert sorted(inhomogeneous_regions) == [0, 1, 2, 3]
        assert len(steps) == 0

    def test_mixed_case(self):
        """
        Matrix with a strip in first 2 columns, ambiguous in last 2.
        """
        matrix = np.zeros((6, 4), dtype=int)
        matrix[:3, :2] = 1  # clear strip in first 2 columns
        # Make ambiguous columns deterministic
        matrix[:, 2] = [0, 1, 0, 1, 0, 1]
        matrix[:, 3] = [1, 0, 1, 0, 1, 0]
        inhomogeneous_regions, steps = pre_processing(matrix, min_col_quality=2,certitude=0.1)
        assert sorted(inhomogeneous_regions) == [2, 3]
        assert len(steps) == 1
        assert sorted(steps[0][2]) == [0, 1]

