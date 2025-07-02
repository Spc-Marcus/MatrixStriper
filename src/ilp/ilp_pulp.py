from model.max_e_r_pulp import max_e_r, max_e_wr
from model.max_one_pulp import max_Ones_comp
import pulp as plp
import numpy as np
from typing import List, Tuple
import logging
import contextlib
import sys
import os
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_pulp_output():
    """Context manager to suppress PuLP output"""
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
    Find a quasi-biclique in a binary matrix using integer linear programming optimization.
    
    This function identifies the largest dense sub-matrix where most elements are 1s,
    with tolerance for noise defined by the error rate. It uses a three-phase approach:
    seeding with a high-density region, then iteratively extending by rows and columns
    to maximize the objective while maintaining density constraints.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows and columns represent 
        data points and features respectively. Values indicate feature presence (1) 
        or absence (0).
    error_rate : float, optional
        Maximum fraction of 0s allowed in the quasi-biclique, defining noise tolerance.
        A value of 0.025 means up to 2.5% of cells can be 0s. Default is 0.025.
    
    Returns
    -------
    Tuple[List[int], List[int], bool]
        Triple containing the quasi-biclique results:
        - [0] : List[int] - Row indices included in the quasi-biclique
        - [1] : List[int] - Column indices included in the quasi-biclique  
        - [2] : bool - Success status (True if valid solution found, False otherwise)
        
        Empty lists are returned when no significant quasi-biclique is found or
        optimization fails.
    """
        
    
    # Copy input matrix to avoid modifying original
    X_problem = input_matrix.copy()
    
    # Get matrix dimensions
    n_rows, n_cols = X_problem.shape
    
    # Handle edge case: empty matrix
    if n_rows == 0 or n_cols == 0:
        logger.debug("Empty matrix provided to quasi-biclique detection")
        return [], [], False
    
    logger.debug(f"Starting quasi-biclique detection on {n_rows}x{n_cols} matrix")
    
    # Sort rows and columns by decreasing number of 1s (highest density first)
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    
    # Phase 1: Seed region selection - find dense area to start optimization
    seed_rows = n_rows // 3
    seed_cols = n_cols // 3
    
    # Adjust step size based on matrix width
    step_n = 10 if n_cols > 50 else 2
    
    # Search for the largest sub-region with >99% density of 1s
    for x in range(n_rows // 3, n_rows, 10):
        for y in range(seed_cols, n_cols, step_n):
            nb_of_ones = 0
            for row in rows_sorted[:x]:
                for col in cols_sorted[:y]:
                    nb_of_ones += X_problem[row, col]
            ratio_ones = nb_of_ones / (x * y)
            if ratio_ones > 0.99:
                seed_rows = x
                seed_cols = y
    
    logger.debug(f"Using seed region: {seed_rows} rows x {seed_cols} columns")
    
    # Calculate degree for each row and column in the seed region
    seed_row_indices = rows_sorted[:seed_rows]
    seed_col_indices = cols_sorted[:seed_cols]
    
    # Calculate degree for seed rows/columns (number of 1s in each row/column)
    seed_matrix = X_problem[np.ix_(seed_row_indices, seed_col_indices)]
    row_degrees = np.sum(seed_matrix == 1, axis=1)
    col_degrees = np.sum(seed_matrix == 1, axis=0)
    
    # Create rows_data and cols_data for seed regions
    rows_data = [(i, int(row_degrees[idx])) for idx, i in enumerate(seed_row_indices)]
    cols_data = [(j, int(col_degrees[idx])) for idx, j in enumerate(seed_col_indices)]
    
    # Create edges: list of (row, col) pairs where the matrix has a 1
    edges = []
    for i_idx, i in enumerate(seed_row_indices):
        for j_idx, j in enumerate(seed_col_indices):
            if seed_matrix[i_idx, j_idx] == 1:
                edges.append((i, j))
    
    # Solve initial optimization problem with seed rows and columns
    try:
        with suppress_pulp_output():
            # Create the model
            model = max_Ones_comp(rows_data, cols_data, edges, error_rate)
            # Solve the model
            model.solve(plp.PULP_CBC_CMD(msg=0))
        
        # Extract seed solution results
        rw = []
        cl = []
        for var in model.variables():
            if var.name.startswith('row_') and var.varValue > 0.5:
                rw.append(int(var.name.split('_')[1]))
            elif var.name.startswith('col_') and var.varValue > 0.5:
                cl.append(int(var.name.split('_')[1]))
        
        logger.debug(f"Initial seed solution: {len(rw)} rows, {len(cl)} columns")
        
        # Phase 2: Row extension - add compatible rows
        rem_rows = [r for r in range(n_rows) if r not in seed_row_indices]
        if len(cl) > 0:
            # Find rows with >50% compatibility with selected columns
            rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
            potential_rows = [r for idx, r in enumerate(rem_rows) 
                             if rem_rows_sum[idx] > 0.5 * len(cl)]
        else:
            potential_rows = []
        
        # If there are potential rows to add, create a new optimization
        if potential_rows and len(rw) > 0 and len(cl) > 0:
            logger.debug(f"Extending with {len(potential_rows)} compatible rows")
            
            # All rows to consider (selected rows + potential rows)
            extended_rows = rw + potential_rows
            
            # Calculate degrees for extended rows
            extended_row_degrees = np.sum(X_problem[extended_rows][:, cl] == 1, axis=1)
            
            # Create new data for extended optimization
            extended_rows_data = [(r, int(extended_row_degrees[idx])) for idx, r in enumerate(extended_rows)]
            extended_cols_data = [(c, int(np.sum(X_problem[extended_rows, c] == 1))) for c in cl]
            
            # Create edges for extended optimization
            extended_edges = []
            for i in extended_rows:
                for j in cl:
                    if X_problem[i, j] == 1:
                        extended_edges.append((i, j))
            
            # Run extended optimization with rows
            with suppress_pulp_output():
                extended_model = max_Ones_comp(extended_rows_data, extended_cols_data, extended_edges, error_rate)
                extended_model.solve(plp.PULP_CBC_CMD(msg=0))
            
            # Extract results after row extension
            rw = []
            cl = []
            for var in extended_model.variables():
                if var.name.startswith('row_') and var.varValue > 0.5:
                    rw.append(int(var.name.split('_')[1]))
                elif var.name.startswith('col_') and var.varValue > 0.5:
                    cl.append(int(var.name.split('_')[1]))
        
        # Phase 3: Column extension - add compatible columns
        rem_cols = [c for c in range(n_cols) if c not in cl]
        if len(rw) > 0:
            # Find columns with >90% compatibility with selected rows
            rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
            potential_cols = [c for idx, c in enumerate(rem_cols) 
                             if rem_cols_sum[idx] > 0.9 * len(rw)]
        else:
            potential_cols = []
        
        # If there are potential columns to add, create a final optimization
        if potential_cols and len(rw) > 0:
            logger.debug(f"Extending with {len(potential_cols)} compatible columns")
            
            # All columns to consider (selected columns + potential columns)
            final_cols = cl + potential_cols
            
            # Calculate degrees for final optimization
            final_row_degrees = np.sum(X_problem[rw][:, final_cols] == 1, axis=1)
            final_col_degrees = np.sum(X_problem[rw][:, final_cols] == 1, axis=0)
            
            # Create data for final optimization
            final_rows_data = [(r, int(final_row_degrees[idx])) for idx, r in enumerate(rw)]
            final_cols_data = [(c, int(np.sum(X_problem[rw, c] == 1))) for c in final_cols]
            
            # Create edges for final optimization
            final_edges = []
            for i in rw:
                for j in final_cols:
                    if X_problem[i, j] == 1:
                        final_edges.append((i, j))
            
            # Run final optimization
            with suppress_pulp_output():
                final_model = max_Ones_comp(final_rows_data, final_cols_data, final_edges, error_rate)
                final_model.solve(plp.PULP_CBC_CMD(msg=0))
            
            # Extract final results
            rw = []
            cl = []
            for var in final_model.variables():
                if var.name.startswith('row_') and var.varValue > 0.5:
                    rw.append(int(var.name.split('_')[1]))
                elif var.name.startswith('col_') and var.varValue > 0.5:
                    cl.append(int(var.name.split('_')[1]))
        
        logger.debug(f"Final quasi-biclique: {len(rw)} rows, {len(cl)} columns")
        
        # Check if we found a valid solution
        if len(rw) > 0 and len(cl) > 0:
            # Calculate density of final selection
            selected = X_problem[np.ix_(rw, cl)]
            density = np.sum(selected == 1) / selected.size
            logger.debug(f"Final density: {density:.4f}")
            return rw, cl, True
        else:
            return [], [], False
        
    except Exception as e:
        logger.error(f"Error solving optimization problem: {e}")
        return [], [], False






def find_quasi_biclique_max_e_r_wr(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using PuLP with seed-and-extend strategy.
    Uses max_e_r for initial seed detection and max_e_wr for extension phases.
    """
    # Copy input matrix to avoid modifying original
    X_problem = input_matrix.copy()
    # Get matrix dimensions
    n_rows, n_cols = X_problem.shape
    # Handle edge case: empty matrix
    if n_rows == 0 or n_cols == 0:
        logger.debug("[QUASI-BICLIQUE] Empty matrix provided to quasi-biclique detection")
        return [], [], False
    # Calculate and log initial matrix statistics
    ones_count = np.sum(X_problem == 1)
    zeros_count = np.sum(X_problem == 0)
    initial_density = ones_count / X_problem.size
    logger.debug(f"[QUASI-BICLIQUE] Initial matrix stats: {ones_count} ones, {zeros_count} zeros, density={initial_density:.4f}")
    logger.debug(f"[QUASI-BICLIQUE](1-error_rate) Created PuLP solver with debug output suppressed")
    try:
        # Sort rows and columns by decreasing number of 1s (highest density first)
        row_sums = X_problem.sum(axis=1)
        col_sums = X_problem.sum(axis=0)
        cols_sorted = np.argsort(col_sums)[::-1]
        rows_sorted = np.argsort(row_sums)[::-1]
        # Phase 1: Select seed region
        seed_rows = max(n_rows // 3, 2)
        seed_cols = max(n_cols // 3, 2)
        logger.debug(f"[QUASI-BICLIQUE] Seed region size: {seed_rows} rows x {seed_cols} columns")
        # Select dense seed rows and columns
        seed_row_indices = rows_sorted[:seed_rows]
        seed_col_indices = cols_sorted[:seed_cols]
        logger.debug(f"[QUASI-BICLIQUE] Selected seed row indices: {seed_row_indices[:10]}{'...' if len(seed_row_indices) > 10 else ''}")
        logger.debug(f"[QUASI-BICLIQUE] Selected seed col indices: {seed_col_indices[:10]}{'...' if len(seed_col_indices) > 10 else ''}")
        # Calculate row and column degrees in seed region
        seed_matrix = X_problem[np.ix_(seed_row_indices, seed_col_indices)]
        row_degrees = np.sum(seed_matrix == 1, axis=1)
        col_degrees = np.sum(seed_matrix == 1, axis=0)
        seed_density = np.sum(seed_matrix == 1) / seed_matrix.size
        logger.debug(f"[QUASI-BICLIQUE] Seed matrix density: {seed_density:.4f}")
        logger.debug(f"[QUASI-BICLIQUE] Row degrees range: [{np.min(row_degrees)}, {np.max(row_degrees)}], mean: {np.mean(row_degrees):.2f}")
        logger.debug(f"[QUASI-BICLIQUE] Col degrees range: [{np.min(col_degrees)}, {np.max(col_degrees)}], mean: {np.mean(col_degrees):.2f}")
        # Prepare data for max_e_r
        rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(seed_row_indices)]
        cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(seed_col_indices)]
        # Create edges (positions of 1s in the matrix)
        edges = []
        for i, r in enumerate(seed_row_indices):
            for j, c in enumerate(seed_col_indices):
                if seed_matrix[i, j] == 1:
                    edges.append((int(r), int(c)))
        logger.debug(f"[QUASI-BICLIQUE] Prepared data for max_e_r: {len(rows_data)} rows, {len(cols_data)} cols, {len(edges)} edges")
        # PHASE 1: Create seed model using max_e_r
        logger.debug("[QUASI-BICLIQUE] ========== STARTING PHASE 1: SEED DETECTION ==========")
        # Force CBC solver to avoid Gurobi license issues
        try:
            logger.debug("[QUASI-BICLIQUE] Calling max_e_r...")
            seed_model = max_e_r(rows_data, cols_data, edges, 0.0025)
            if seed_model is None:
                logger.error("[QUASI-BICLIQUE] max_e_r returned None")
                return [], [], False
            logger.debug("[QUASI-BICLIQUE] Starting seed model optimization...")
            # Use CBC solver explicitly with complete output suppression
            with suppress_pulp_output():
                # Force CBC solver and suppress all output
                solver = pl.PULP_CBC_CMD(
                    msg=0,
                    keepFiles=0,
                    gapRel=0.01,
                    timeLimit=300,
                    options=[
                        '-log', '0',
                        '-printingOptions', 'none',
                        '-preprocess', 'off'
                    ]
                )
                seed_model.solve(solver)
            logger.debug(f"[QUASI-BICLIQUE] Seed optimization completed with status: {pl.LpStatus[seed_model.status]}")
            # Check optimization status
            if seed_model.status != pl.LpStatusOptimal:
                logger.error(f"[QUASI-BICLIQUE] Seed optimization failed with status {pl.LpStatus[seed_model.status]}")
                return [], [], False
            # Extract seed solution
            logger.debug("[QUASI-BICLIQUE] Extracting seed solution variables...")
            rw = []
            cl = []
            for var in seed_model.variables():
                if var.varValue and var.varValue > 0.5:
                    if var.name.startswith("row_"):
                        rw.append(int(var.name.split("_")[1]))
                    elif var.name.startswith("col_"):
                        cl.append(int(var.name.split("_")[1]))
            seed_obj = pl.value(seed_model.objective)
            logger.debug(f"[QUASI-BICLIQUE] Seed solution extracted: {len(rw)} rows, {len(cl)} columns, obj={seed_obj}")
        except Exception as e:
            logger.error(f"[QUASI-BICLIQUE] Error in seed phase: {str(e)}")
            return [], [], False
        # If seed phase found no solution, return empty
        if not rw or not cl:
            logger.debug("[QUASI-BICLIQUE] No solution found in seed phase")
            return [], [], False
        # Calculate seed solution density
        seed_solution_matrix = X_problem[np.ix_(rw, cl)]
        seed_solution_density = np.sum(seed_solution_matrix == 1) / seed_solution_matrix.size
        logger.debug(f"[QUASI-BICLIQUE] Seed solution density: {seed_solution_density:.4f}")
        # PHASE 2: Extension using max_e_wr on the entire matrix
        logger.debug("[QUASI-BICLIQUE] ========== STARTING PHASE 2: FULL MATRIX EXTENSION ==========")
        # Prepare data for max_e_wr on the entire matrix
        all_row_degrees = np.sum(X_problem == 1, axis=1)
        all_col_degrees = np.sum(X_problem == 1, axis=0)
        all_rows_data = [(int(r), int(all_row_degrees[r])) for r in range(n_rows)]
        all_cols_data = [(int(c), int(all_col_degrees[c])) for c in range(n_cols)]
        # Create edges for the entire matrix (positions of 1s)
        all_edges = []
        for r in range(n_rows):
            for c in range(n_cols):
                if X_problem[r, c] == 1:
                    all_edges.append((int(r), int(c)))
        logger.debug(f"[QUASI-BICLIQUE] Full matrix data: {len(all_rows_data)} rows, {len(all_cols_data)} cols, {len(all_edges)} edges")
        # Use max_e_wr for extension on the entire matrix
        try:
            logger.debug("[QUASI-BICLIQUE] Calling max_e_wr on entire matrix...")
            full_model = max_e_wr(
                all_rows_data,
                all_cols_data,
                all_edges,
                rw,  # previous rows from seed
                cl,  # previous columns from seed
                seed_obj,  # previous objective from seed
                error_rate
            )
            if full_model is None:
                logger.error("[QUASI-BICLIQUE] max_e_wr returned None for full matrix extension")
            else:
                logger.debug("[QUASI-BICLIQUE] Starting full matrix extension optimization...")
                # Use CBC solver explicitly with complete output suppression
                with suppress_pulp_output():
                    solver = pl.PULP_CBC_CMD(
                        msg=0,
                        keepFiles=0,
                        gapRel=0.01,
                        timeLimit=300,
                        options=[
                            '-log', '0',
                            '-printingOptions', 'none',
                            '-preprocess', 'off'
                        ]
                    )
                    full_model.solve(solver)
                logger.debug(f"[QUASI-BICLIQUE] Full matrix extension completed with status: {pl.LpStatus[full_model.status]}")
                # Check optimization status
                if full_model.status == pl.LpStatusOptimal:
                    # Extract full matrix extension solution
                    logger.debug("[QUASI-BICLIQUE] Extracting full matrix extension solution...")
                    rw = []
                    cl = []
                    for var in full_model.variables():
                        if var.varValue and var.varValue > 0.5:
                            if var.name.startswith("row_"):
                                rw.append(int(var.name.split("_")[1]))
                            elif var.name.startswith("col_"):
                                cl.append(int(var.name.split("_")[1]))
                    full_obj = pl.value(full_model.objective)
                    logger.debug(f"[QUASI-BICLIQUE] Full matrix extension solution: {len(rw)} rows, {len(cl)} columns, obj={full_obj}")
                else:
                    logger.debug(f"[QUASI-BICLIQUE] Full matrix extension failed with status {pl.LpStatus[full_model.status]}")
                    # Keep seed solution if extension failed
        except Exception as e:
            logger.error(f"[QUASI-BICLIQUE] Error in full matrix extension: {str(e)}")
            # Continue with seed solution if extension failed
            pass
        # Check if we found a valid solution
        logger.debug("[QUASI-BICLIQUE] ========== FINAL SOLUTION VALIDATION ==========")
        if rw and cl:
            # Calculate final density
            selected = X_problem[np.ix_(rw, cl)]
            density = np.sum(selected == 1) / selected.size
            ones_in_solution = np.sum(selected == 1)
            zeros_in_solution = np.sum(selected == 0)
            logger.debug(f"[QUASI-BICLIQUE] Final quasi-biclique found:")
            logger.debug(f"[QUASI-BICLIQUE]   - Dimensions: {len(rw)} rows x {len(cl)} columns")
            logger.debug(f"[QUASI-BICLIQUE]   - Density: {density:.4f}")
            logger.debug(f"[QUASI-BICLIQUE]   - Ones: {ones_in_solution}, Zeros: {zeros_in_solution}")
            logger.debug(f"[QUASI-BICLIQUE]   - Row indices: {rw}")
            logger.debug(f"[QUASI-BICLIQUE]   - Column indices: {cl}")
            logger.debug("[QUASI-BICLIQUE] Successfully returning valid solution")
            return rw, cl, True
        # No solution found
        logger.debug("[QUASI-BICLIQUE] No valid solution found")
        return [], [], False
    except Exception as e:
        logger.error(f"[QUASI-BICLIQUE] Critical error in quasi-biclique detection: {str(e)}")
        return [], [], False