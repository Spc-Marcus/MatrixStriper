import logging
import os
import numpy as np
from typing import List, Tuple
# Configuration cloud Gurobi : ces variables d'environnement doivent être définies AVANT l'import de gurobipy
# os.environ['GRB_WLSACCESSID'] = 'af4b8280-70cd-47bc-aeef-69ecf14ecd10'
# os.environ['GRB_WLSSECRET'] = '04da6102-8eb3-4e38-ba06-660ea8f87bf2'
# os.environ['GRB_LICENSEID'] = '2669217'
from model.max_one_grb import max_Ones_gurobi
from model.max_e_r_grb import MaxERSolver
import contextlib
import sys
import gurobipy as grb
from model.max_e_r_V2_grb import MaxERModel

logger = logging.getLogger(__name__)

# Suppress ALL Gurobi output
gurobi_logger = logging.getLogger('gurobipy')
gurobi_logger.setLevel(logging.CRITICAL)
gurobi_logger.propagate = False

@contextlib.contextmanager
def suppress_gurobi_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def find_quasi_dens_matrix_max_onesV2(
    input_matrix: np.ndarray,
    error_rate: float = 0.025
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using integer linear programming optimization (Gurobi).
    Utilise max_Ones_comp_gurobi pour la résolution du modèle à chaque étape.
    """
    X_problem = input_matrix.copy()
    n_rows, n_cols = X_problem.shape
    if n_rows == 0 or n_cols == 0:
        return [], [], False
    
    try:
        # --- PHASE 1: SEED - Sélection des lignes d'abord ---
        # Calculer les degrés des lignes et colonnes
        row_degrees = np.sum(X_problem == 1, axis=1)
        col_degrees = np.sum(X_problem == 1, axis=0)
        
        # Trier les colonnes par degré décroissant pour le seeding
        cols_sums = X_problem.sum(axis=0)
        cols_sorted = np.argsort(cols_sums)[::-1]
        
        # Déterminer le nombre de colonnes pour le seeding (comme dans V2)
        seed_cols = min(max(n_cols // 3, min(n_cols, 5)), 50)
        seed_col_indices = cols_sorted[:seed_cols]
        
        # Créer les données pour le modèle avec toutes les lignes et les colonnes de seed
        rows_data = [(int(r), int(row_degrees[r])) for r in range(n_rows)]
        cols_data = [(int(c), int(col_degrees[c])) for c in seed_col_indices]
        
        # Créer les arêtes pour la matrice de seed
        edges = []
        for r in range(n_rows):
            for c in seed_col_indices:
                if X_problem[r, c] == 1:
                    edges.append((int(r), int(c)))
        
        # Résoudre le modèle de seed avec densité maximale (error_rate = 0)
        model = max_Ones_gurobi(rows_data, cols_data, edges, 0)
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0.05)
        model.setParam('TimeLimit', 20)
        model.optimize()
        
        status = model.Status
        if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
            return [], [], False
        elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
            rw = []
            cl = []
            for v in model.getVars():
                if v.VarName.startswith('row_') and v.X > 0.5:
                    rw.append(int(v.VarName.split('_')[1]))
                elif v.VarName.startswith('col_') and v.X > 0.5:
                    cl.append(int(v.VarName.split('_')[1]))
        else:
            return [], [], False
        
        if not rw or not cl:
            return [], [], False
        
        # --- PHASE 2: EXTENSION COLONNES - Ajouter des colonnes sur les lignes sélectionnées ---
        # Trouver les colonnes restantes qui ne sont pas dans la solution de seed
        rem_cols = [c for c in range(n_cols) if c not in cl]
        
        if len(rw) > 0 and len(rem_cols) > 0:
            # Calculer les scores des colonnes restantes basés sur les lignes sélectionnées
            rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
            potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
            
            if potential_cols:
                # Ajouter les colonnes potentielles à la solution
                all_col_indices = cl + potential_cols
                
                # Recalculer les degrés pour les nouvelles colonnes
                row_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=1)
                col_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=0)
                
                rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rw)]
                cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(all_col_indices)]
                
                # Créer les nouvelles arêtes
                edges = []
                for i, r in enumerate(rw):
                    for j, c in enumerate(all_col_indices):
                        if X_problem[r, c] == 1:
                            edges.append((int(r), int(c)))
                
                # Résoudre le modèle étendu avec le taux d'erreur spécifié
                model = max_Ones_gurobi(rows_data, cols_data, edges, error_rate)
                model.setParam('OutputFlag', 0)
                model.setParam('MIPGap', 0.05)
                model.setParam('TimeLimit', 180)
                model.optimize()
                
                status = model.Status
                if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                    return [], [], False
                elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                    rw = []
                    cl = []
                    for v in model.getVars():
                        if v.VarName.startswith('row_') and v.X > 0.5:
                            rw.append(int(v.VarName.split('_')[1]))
                        elif v.VarName.startswith('col_') and v.X > 0.5:
                            cl.append(int(v.VarName.split('_')[1]))
                else:
                    return [], [], False
        
        # Vérifier le statut final
        if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
            return [], [], False
        elif status == grb.GRB.TIME_LIMIT:
            return rw, cl, True
        elif status != grb.GRB.OPTIMAL:
            return [], [], False
        
        return rw, cl, True
    except Exception as e:
        logger.error(f"[GRB] Critical error in quasi-biclique detection: {str(e)}")
        return [], [], False

def find_quasi_dens_matrix_max_ones(
    input_matrix: np.ndarray,
    error_rate: float = 0.025
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using integer linear programming optimization (Gurobi).
    Utilise max_Ones_comp_gurobi pour la résolution du modèle à chaque étape.
    """
    X_problem = input_matrix.copy()
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    m = len(rows_sorted)
    n = len(cols_sorted)
    if m == 0 or n == 0:
        return [], [], False
    seed_rows = max(m // 3, 2)
    seed_cols = max(n // 3, 2)
    if n > 50:
        step_n = 10
    else:
        step_n = 2
    for x in range(m // 3, m, 10):
        for y in range(seed_cols, n, step_n):
            nb_of_ones = 0
            for row in rows_sorted[:x]:
                for col in cols_sorted[:y]:
                    nb_of_ones += X_problem[row, col]
            ratio_ones = nb_of_ones / (x * y) if (x * y) > 0 else 0
            if ratio_ones > 0.99:
                seed_rows = x
                seed_cols = y
    try:
        # --- PHASE 1: SEED ---
        seed_row_indices = rows_sorted
        seed_col_indices = cols_sorted[:seed_cols]
        row_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=1)
        col_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=0)
        rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(seed_row_indices)]
        cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(seed_col_indices)]
        edges = []
        for i, r in enumerate(seed_row_indices):
            for j, c in enumerate(seed_col_indices):
                if X_problem[r, c] == 1:
                    edges.append((int(r), int(c)))
        model = max_Ones_gurobi(rows_data, cols_data, edges, 0)
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0.05)
        model.setParam('TimeLimit', 20)
        model.optimize()
        status = model.Status
        if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
            return [], [], False
        elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
            rw = []
            cl = []
            for v in model.getVars():
                if v.VarName.startswith('row_') and v.X > 0.5:
                    rw.append(int(v.VarName.split('_')[1]))
                elif v.VarName.startswith('col_') and v.X > 0.5:
                    cl.append(int(v.VarName.split('_')[1]))
        else:
            return [], [], False
        
        # --- PHASE 3: EXTENSION COLONNES ---
        rem_cols = [c for c in cols_sorted if c not in cl]
        if len(rw) > 0:
            rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
            potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
        else:
            potential_cols = []
        if potential_cols:
            all_col_indices = cl + potential_cols
            row_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=1)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rw)]
            col_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=0)
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(all_col_indices)]
            edges = []
            for i, r in enumerate(rw):
                for j, c in enumerate(all_col_indices):
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            model = max_Ones_gurobi(rows_data, cols_data, edges, error_rate)
            model.setParam('OutputFlag', 0)
            model.setParam('MIPGap', 0.05)
            model.setParam('TimeLimit', 180)
            model.optimize()
        
            status = model.Status
            if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                return [], [], False
            elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                rw = []
                cl = []
                for v in model.getVars():
                    if v.VarName.startswith('row_') and v.X > 0.5:
                        rw.append(int(v.VarName.split('_')[1]))
                    elif v.VarName.startswith('col_') and v.X > 0.5:
                        cl.append(int(v.VarName.split('_')[1]))
            else:
                return [], [], False
            # --- PHASE 2: EXTENSION LIGNES ---
        rem_rows = [r for r in rows_sorted if r not in rw]
        if len(cl) > 0:
            rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
            potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
        else:
            potential_rows = []
        if potential_rows:
            all_row_indices = rw + potential_rows
            row_degrees = np.sum(X_problem[all_row_indices, :][:, cl] == 1, axis=1)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(all_row_indices)]
            col_degrees = np.sum(X_problem[all_row_indices, :][:, cl] == 1, axis=0)
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(cl)]
            edges = []
            for i, r in enumerate(all_row_indices):
                for j, c in enumerate(cl):
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            model = max_Ones_gurobi(rows_data, cols_data, edges, error_rate)
            model.setParam('OutputFlag', 0)
            model.setParam('MIPGap', 0.05)
            model.setParam('TimeLimit', 20)
            model.optimize()
            status = model.Status
            if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                return [], [], False
            elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                rw = []
                cl = []
                for v in model.getVars():
                    if v.VarName.startswith('row_') and v.X > 0.5:
                        rw.append(int(v.VarName.split('_')[1]))
                    elif v.VarName.startswith('col_') and v.X > 0.5:
                        cl.append(int(v.VarName.split('_')[1]))
                logger.debug(f"[DEBUG] PHASE 2: rw={rw}, cl={cl}")
            else:
                return [], [], False
        status = model.Status
        if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
            return [], [], False
        elif status == grb.GRB.TIME_LIMIT:
            return rw, cl, True
        elif status != grb.GRB.OPTIMAL:
            return [], [], False
        return rw, cl, True
    except Exception as e:
        logger.error(f"[GRB] Critical error in quasi-biclique detection: {str(e)}")
        return [], [], False


def find_quasi_biclique_max_e_r_wr(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using Gurobi with seed-and-extend strategy.
    Utilise MaxERSolver (max_e_r, max_e_wr).
    """
    X_problem = input_matrix.copy()
    n_rows, n_cols = X_problem.shape
    if n_rows == 0 or n_cols == 0:
        logger.debug("[GRB] Empty matrix provided to quasi-biclique detection")
        return [], [], False
    ones_count = np.sum(X_problem == 1)
    zeros_count = np.sum(X_problem == 0)
    initial_density = ones_count / X_problem.size
    logger.debug(f"[GRB] Initial matrix stats: {ones_count} ones, {zeros_count} zeros, density={initial_density:.4f}")
    
    try:
        row_sums = X_problem.sum(axis=1)
        col_sums = X_problem.sum(axis=0)
        cols_sorted = np.argsort(col_sums)[::-1]
        rows_sorted = np.argsort(row_sums)[::-1]
        seed_cols = max(n_cols // 3, 2)
        logger.debug(f"[GRB] Seed region size: {seed_cols} columns")
        seed_row_indices = rows_sorted
        seed_col_indices = cols_sorted[:seed_cols]
        seed_matrix = X_problem[np.ix_(seed_row_indices, seed_col_indices)]
        row_degrees = np.sum(seed_matrix == 1, axis=1)
        col_degrees = np.sum(seed_matrix == 1, axis=0)
        seed_density = np.sum(seed_matrix == 1) / seed_matrix.size
        logger.debug(f"[GRB] Seed matrix density: {seed_density:.4f}")
        rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(seed_row_indices)]
        cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(seed_col_indices)]
        edges = []
        for i, r in enumerate(seed_row_indices):
            for j, c in enumerate(seed_col_indices):
                if seed_matrix[i, j] == 1:
                    edges.append((int(r), int(c)))
        solver = MaxERSolver()
        seed_model = solver.max_e_r(rows_data, cols_data, edges, error_rate)
        seed_model.setParam('OutputFlag', 0)
        seed_model.optimize()
        rw = []
        cl = []
        for v in seed_model.getVars():
            if v.VarName.startswith('row_') and v.X > 0.5:
                rw.append(int(v.VarName.split('_')[1]))
            elif v.VarName.startswith('col_') and v.X > 0.5:
                cl.append(int(v.VarName.split('_')[1]))
        if not rw or not cl:
            logger.debug("[GRB] No solution found in seed phase")
            return [], [], False
        seed_solution_matrix = X_problem[np.ix_(rw, cl)]
        seed_solution_density = np.sum(seed_solution_matrix == 1) / seed_solution_matrix.size
        logger.debug(f"[GRB] Seed solution density: {seed_solution_density:.4f}")
        # --- PHASE 2: EXTENSION COLONNES ---
        row_degrees = np.sum(X_problem[rw, :] == 1, axis=1)
        rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rw)]
        col_degrees = np.sum(X_problem[rw, :] == 1, axis=0)
        cols_data = [(int(c), int(col_degrees[c])) for c in range(n_cols)]
        edges = []
        for i, r in enumerate(rw):
            for c in range(n_cols):
                if X_problem[r, c] == 1:
                    edges.append((int(r), int(c)))
        prev_obj = seed_model.objVal
        full_model = solver.max_e_wr(
            rows_data,
            cols_data,
            edges,
            rw,
            cl,
            prev_obj,
            error_rate
        )
        full_model.setParam('OutputFlag', 0)
        full_model.optimize()
        if full_model.status == 2:  # GRB.OPTIMAL
            rw = []
            cl = []
            for v in full_model.getVars():
                if v.VarName.startswith('row_') and v.X > 0.5:
                    rw.append(int(v.VarName.split('_')[1]))
                elif v.VarName.startswith('col_') and v.X > 0.5:
                    cl.append(int(v.VarName.split('_')[1]))
        if rw and cl:
            selected = X_problem[np.ix_(rw, cl)]
            density = np.sum(selected == 1) / selected.size
            logger.debug(f"[GRB] Final quasi-biclique: {len(rw)} rows, {len(cl)} columns, density={density:.4f}")
            return rw, cl, True
        logger.debug("[GRB] No valid solution found")
        return [], [], False
    except Exception as e:
        logger.error(f"[GRB] Critical error in quasi-biclique detection: {str(e)}")
        return [], [], False

def find_quasi_biclique_max_e_r_V2(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using Gurobi with seed-and-extend strategy.
    Utilise MaxERModel (max_e_r, max_e_wr).
    """
    X_problem = input_matrix.copy()
    n_rows, n_cols = X_problem.shape
    if n_rows == 0 or n_cols == 0:
        logger.debug("[GRB] Empty matrix provided to quasi-biclique detection")
        return [], [], False
    ones_count = np.sum(X_problem == 1)
    zeros_count = np.sum(X_problem == 0)
    initial_density = ones_count / X_problem.size
    logger.debug(f"[GRB] Initial matrix stats: {ones_count} ones, {zeros_count} zeros, density={initial_density:.4f}")

    try:
        # --- PHASE 1: SEED ---
        row_degrees = np.sum(X_problem == 1, axis=1)
        rows_data = [(r, int(row_degrees[r])) for r in range(n_rows)]
        col_degrees = np.sum(X_problem == 1, axis=0)
        cols_data = [(int(c), int(col_degrees[c])) for c in range(n_cols)]
        edges = []
        cols_sums = X_problem.sum(axis=0)
        cols_sorted = np.argsort(cols_sums)[::-1]
        seed_cols = min(max(n_cols // 3, min(n_cols, 5)), 50)
        no_use_cols_seed = cols_sorted[seed_cols:]

        for r in range(n_rows):
            for c in range(n_cols):
                if X_problem[r, c] == 1:
                    edges.append((int(r), int(c)))
        model = MaxERModel(rows_data, cols_data, edges)
        model.setParam('OutputFlag', 1)
        model.build_max_e_r(3, 2)
        model.add_density_constraints(0)
        model.add_forced_cols_zero(no_use_cols_seed)
        model.optimize()

        if model.status == 2:
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()
        else:
            return [], [], False
        """
        elif model.status == 3:
            model.computeIIS()
            model.write("model.ilp")
            logger.debug("[GRB] IIS written to model.ilp")
            return [], [], False
        """
        
        # --- PHASE 2: EXTENSION COLONNES ---
        no_use_rows_seed = [r for r in range(n_rows) if r not in rw]
        model.remove_forced_cols_zero(no_use_cols_seed)
        model.add_forced_rows_zero(no_use_rows_seed)
        model.add_improvement_constraint(model.ObjVal)
        model.update_density_constraints(error_rate)
        model.optimize()
        if model.status == 2:
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()
        """
        elif model.status == 3:
            model.computeIIS()
            model.write("model.ilp")
            logger.debug("[GRB] IIS written to model.ilp")"""
        
        if rw and cl:
            selected = X_problem[np.ix_(rw, cl)]
            density = np.sum(selected == 1) / selected.size
            logger.debug(f"[GRB] Final quasi-biclique: {len(rw)} rows, {len(cl)} columns, density={density:.4f}")
            return rw, cl, True
        logger.debug("[GRB] No valid solution found")
        return [], [], False
    except Exception as e:
        logger.error(f"[GRB] Critical error in quasi-biclique detection: {str(e)}")
        return [], [], False
