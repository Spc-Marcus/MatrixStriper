import logging
import os
import numpy as np
from typing import List, Tuple
# Configuration cloud Gurobi : ces variables d'environnement doivent être définies AVANT l'import de gurobipy
os.environ['GRB_WLSACCESSID'] = 'af4b8280-70cd-47bc-aeef-69ecf14ecd10'
os.environ['GRB_WLSSECRET'] = '04da6102-8eb3-4e38-ba06-660ea8f87bf2'
os.environ['GRB_LICENSEID'] = '2669217'
from model.max_one_grb import max_Ones_comp_gurobi
from model.max_e_r_grb import MaxERSolver
import contextlib
import sys
import gurobipy as grb

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

def find_quasi_biclique_max_ones_comp(
    input_matrix: np.ndarray,
    error_rate: float = 0.025
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using integer linear programming optimization (Gurobi).
    Identique à la version PuLP mais utilise max_Ones_comp_gurobi.
    """
    X_problem = input_matrix.copy()
    n_rows, n_cols = X_problem.shape
    if n_rows == 0 or n_cols == 0:
        logger.debug("Empty matrix provided to quasi-biclique detection")
        return [], [], False
    logger.debug(f"[GRB] Starting quasi-biclique detection on {n_rows}x{n_cols} matrix")
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    seed_rows = n_rows // 3
    seed_cols = n_cols // 3
    step_n = 10 if n_cols > 50 else 2
    for x in range(n_rows // 3, n_rows, 10):
        for y in range(seed_cols, n_cols, step_n):
            if x == 0 or y == 0:
                continue
            nb_of_ones = 0
            for row in rows_sorted[:x]:
                for col in cols_sorted[:y]:
                    nb_of_ones += X_problem[row, col]
            ratio_ones = nb_of_ones / (x * y)
            if ratio_ones > 0.99:
                seed_rows = x
                seed_cols = y
    logger.debug(f"[GRB] Using seed region: {seed_rows} rows x {seed_cols} columns")
    seed_row_indices = rows_sorted[:seed_rows]
    seed_col_indices = cols_sorted[:seed_cols]
    seed_matrix = X_problem[np.ix_(seed_row_indices, seed_col_indices)]
    row_degrees = np.sum(seed_matrix == 1, axis=1)
    col_degrees = np.sum(seed_matrix == 1, axis=0)
    rows_data = [(i, int(row_degrees[idx])) for idx, i in enumerate(seed_row_indices)]
    cols_data = [(j, int(col_degrees[idx])) for idx, j in enumerate(seed_col_indices)]
    edges = []
    for i_idx, i in enumerate(seed_row_indices):
        for j_idx, j in enumerate(seed_col_indices):
            if seed_matrix[i_idx, j_idx] == 1:
                edges.append((i, j))
    try:
        model = max_Ones_comp_gurobi(rows_data, cols_data, edges, error_rate)
        model.setParam('OutputFlag', 0)
        model.optimize()
        rw = []
        cl = []
        for v in model.getVars():
            if v.varName.startswith('row_') and v.x > 0.5:
                rw.append(int(v.varName.split('_')[1]))
            elif v.varName.startswith('col_') and v.x > 0.5:
                cl.append(int(v.varName.split('_')[1]))
        logger.debug(f"[GRB] Initial seed solution: {len(rw)} rows, {len(cl)} columns")
        rem_rows = [r for r in range(n_rows) if r not in seed_row_indices]
        if len(cl) > 0:
            rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
            potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
        else:
            potential_rows = []
        if potential_rows and len(rw) > 0 and len(cl) > 0:
            logger.debug(f"[GRB] Extending with {len(potential_rows)} compatible rows")
            extended_rows = rw + potential_rows
            extended_row_degrees = np.sum(X_problem[extended_rows][:, cl] == 1, axis=1)
            extended_rows_data = [(r, int(extended_row_degrees[idx])) for idx, r in enumerate(extended_rows)]
            extended_cols_data = [(c, int(np.sum(X_problem[extended_rows, c] == 1))) for c in cl]
            extended_edges = []
            for i in extended_rows:
                for j in cl:
                    if X_problem[i, j] == 1:
                        extended_edges.append((i, j))
            model = max_Ones_comp_gurobi(extended_rows_data, extended_cols_data, extended_edges, error_rate)
            model.setParam('OutputFlag', 0)
            model.optimize()
            rw = []
            cl = []
            for v in model.getVars():
                if v.varName.startswith('row_') and v.x > 0.5:
                    rw.append(int(v.varName.split('_')[1]))
                elif v.varName.startswith('col_') and v.x > 0.5:
                    cl.append(int(v.varName.split('_')[1]))
        rem_cols = [c for c in range(n_cols) if c not in cl]
        if len(rw) > 0:
            rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
            potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
        else:
            potential_cols = []
        if potential_cols and len(rw) > 0:
            logger.debug(f"[GRB] Extending with {len(potential_cols)} compatible columns")
            final_cols = cl + potential_cols
            final_row_degrees = np.sum(X_problem[rw][:, final_cols] == 1, axis=1)
            final_col_degrees = np.sum(X_problem[rw][:, final_cols] == 1, axis=0)
            final_rows_data = [(r, int(final_row_degrees[idx])) for idx, r in enumerate(rw)]
            final_cols_data = [(c, int(np.sum(X_problem[rw, c] == 1))) for c in final_cols]
            final_edges = []
            for i in rw:
                for j in final_cols:
                    if X_problem[i, j] == 1:
                        final_edges.append((i, j))
            model = max_Ones_comp_gurobi(final_rows_data, final_cols_data, final_edges, error_rate)
            model.setParam('OutputFlag', 0)
            model.optimize()
            rw = []
            cl = []
            for v in model.getVars():
                if v.varName.startswith('row_') and v.x > 0.5:
                    rw.append(int(v.varName.split('_')[1]))
                elif v.varName.startswith('col_') and v.x > 0.5:
                    cl.append(int(v.varName.split('_')[1]))
        logger.debug(f"[GRB] Final quasi-biclique: {len(rw)} rows, {len(cl)} columns")
        if len(rw) > 0 and len(cl) > 0:
            selected = X_problem[np.ix_(rw, cl)]
            density = np.sum(selected == 1) / selected.size
            logger.debug(f"[GRB] Final density: {density:.4f}")
            return rw, cl, True
        else:
            return [], [], False
    except Exception as e:
        logger.error(f"[GRB] Error solving optimization problem: {e}")
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
        seed_rows = max(n_rows // 3, 2)
        seed_cols = max(n_cols // 3, 2)
        logger.debug(f"[GRB] Seed region size: {seed_rows} rows x {seed_cols} columns")
        seed_row_indices = rows_sorted[:seed_rows]
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
        seed_model = solver.max_e_r(rows_data, cols_data, edges, 0.0025)
        seed_model.setParam('OutputFlag', 0)
        seed_model.optimize()
        rw = []
        cl = []
        for v in seed_model.getVars():
            if v.varName.startswith('row_') and v.x > 0.5:
                rw.append(int(v.varName.split('_')[1]))
            elif v.varName.startswith('col_') and v.x > 0.5:
                cl.append(int(v.varName.split('_')[1]))
        if not rw or not cl:
            logger.debug("[GRB] No solution found in seed phase")
            return [], [], False
        seed_solution_matrix = X_problem[np.ix_(rw, cl)]
        seed_solution_density = np.sum(seed_solution_matrix == 1) / seed_solution_matrix.size
        logger.debug(f"[GRB] Seed solution density: {seed_solution_density:.4f}")
        all_row_degrees = np.sum(X_problem == 1, axis=1)
        all_col_degrees = np.sum(X_problem == 1, axis=0)
        all_rows_data = [(int(r), int(all_row_degrees[r])) for r in range(n_rows)]
        all_cols_data = [(int(c), int(all_col_degrees[c])) for c in range(n_cols)]
        all_edges = []
        for r in range(n_rows):
            for c in range(n_cols):
                if X_problem[r, c] == 1:
                    all_edges.append((int(r), int(c)))
        prev_obj = seed_model.objVal
        full_model = solver.max_e_wr(
            all_rows_data,
            all_cols_data,
            all_edges,
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
                if v.varName.startswith('row_') and v.x > 0.5:
                    rw.append(int(v.varName.split('_')[1]))
                elif v.varName.startswith('col_') and v.x > 0.5:
                    cl.append(int(v.varName.split('_')[1]))
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

