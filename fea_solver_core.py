# fea_solver_core.py
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator
import numba

@numba.jit(nopython=True, cache=True)
def _get_element_stiffness_tet4(node_coords, D_matrix):
    """
    Calculates the stiffness matrix for a single linear tetrahedral element (TET4).
    """
    M = np.ones((4, 4))
    M[:, 1:] = node_coords
    
    volume = np.linalg.det(M) / 6.0
    if volume <= 1.0e-18: 
        return np.zeros((12, 12)), False

    dN_dx = np.linalg.inv(M).T 

    B = np.zeros((6, 12))
    
    for i in range(4):
        dN_i_dx, dN_i_dy, dN_i_dz = dN_dx[i, 1], dN_dx[i, 2], dN_dx[i, 3]
        col = i * 3
        B[0, col]=dN_i_dx; B[1, col+1]=dN_i_dy; B[2, col+2]=dN_i_dz
        B[3, col]=dN_i_dy; B[3, col+1]=dN_i_dx
        B[4, col+1]=dN_i_dz; B[4, col+2]=dN_i_dy
        B[5, col]=dN_i_dz; B[5, col+2]=dN_i_dx

    ke = B.T @ D_matrix @ B * volume
    return ke, True

@numba.jit(nopython=True, cache=True, parallel=True)
def _assemble_k_global_data(elements, nodes, D_matrix):
    """
    (Replaces batching) Calculates all data for the global stiffness matrix in a
    single, parallel pass. This is a "Direct-to-COO" assembly method.
    """
    num_elements = len(elements)
    entries_per_element = 144
    total_max_entries = num_elements * entries_per_element

    data = np.zeros(total_max_entries, dtype=np.float64)
    rows = np.zeros(total_max_entries, dtype=np.int64)
    cols = np.zeros(total_max_entries, dtype=np.int64)
    valid_elements_mask = np.zeros(num_elements, dtype=np.bool_)
    
    for i in numba.prange(num_elements):
        node_ids = elements[i]
        
        ke, is_valid = _get_element_stiffness_tet4(nodes[node_ids], D_matrix)
        valid_elements_mask[i] = is_valid

        if is_valid:
            dof_indices = np.empty(12, dtype=np.int64)
            for j in range(4):
                node_id = node_ids[j]
                dof_indices[j*3]   = node_id * 3
                dof_indices[j*3+1] = node_id * 3 + 1
                dof_indices[j*3+2] = node_id * 3 + 2

            write_pos_start = i * entries_per_element
            k = 0
            for r in range(12):
                for c in range(12):
                    idx = write_pos_start + k
                    data[idx] = ke[r, c]
                    rows[idx] = dof_indices[r]
                    cols[idx] = dof_indices[c]
                    k += 1
                    
    return data, rows, cols, valid_elements_mask

@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_stress_for_valid_elements(valid_elements, nodes, displacements, D_matrix):
    """
    Stage 3 (Parallel): Calculates stresses at the centroid of valid elements.
    """
    num_valid_elements = len(valid_elements)
    element_stresses = np.zeros((num_valid_elements, 6))

    for i in numba.prange(num_valid_elements):
        node_ids = valid_elements[i]
        element_nodes_coords = nodes[node_ids]

        M = np.ones((4, 4))
        M[:, 1:] = element_nodes_coords
        
        if abs(np.linalg.det(M)) < 1e-18:
            continue

        element_displacements = displacements[node_ids].flatten()
        
        dN_dx = np.linalg.inv(M).T
        B = np.zeros((6, 12))
        for j in range(4):
            dN_j_dx, dN_j_dy, dN_j_dz = dN_dx[j, 1], dN_dx[j, 2], dN_dx[j, 3]
            col = j * 3
            B[0, col] = dN_j_dx
            B[1, col + 1] = dN_j_dy
            B[2, col + 2] = dN_j_dz
            B[3, col] = dN_j_dy;     B[3, col + 1] = dN_j_dx
            B[4, col + 1] = dN_j_dz; B[4, col + 2] = dN_j_dy
            B[5, col] = dN_j_dz;     B[5, col + 2] = dN_j_dx
        element_stresses[i] = D_matrix @ B @ element_displacements
        
    return element_stresses

@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_strain_energy_for_valid_elements(valid_elements, nodes, displacements, D_matrix):
    """
    Calculates the strain energy for each valid element. Ue = 0.5 * u.T * ke * u
    """
    num_valid_elements = len(valid_elements)
    element_strain_energy = np.zeros(num_valid_elements)

    for i in numba.prange(num_valid_elements):
        node_ids = valid_elements[i]
        element_nodes_coords = nodes[node_ids]
        
        # Re-calculate the element stiffness matrix (ke) for this element
        ke, is_valid = _get_element_stiffness_tet4(element_nodes_coords, D_matrix)
        
        if is_valid:
            # Extract the 12 DOFs for this element from the global displacement vector
            element_displacements_vec = displacements[node_ids].flatten()
            
            # Strain Energy Ue = 0.5 * ue.T * ke * ue
            energy = 0.5 * (element_displacements_vec.T @ ke @ element_displacements_vec)
            element_strain_energy[i] = energy
            
    return element_strain_energy

@numba.jit(nopython=True, cache=True)
def _average_element_stress_to_nodes(num_nodes, valid_elements, element_von_mises):
    """
    Averages the element-centric von Mises stress to the nodes for smooth visualization.
    """
    nodal_stress_sum = np.zeros(num_nodes, dtype=np.float64)
    nodal_contribution_count = np.zeros(num_nodes, dtype=np.int32)

    for i in range(len(valid_elements)):
        element_nodes = valid_elements[i]
        stress = element_von_mises[i]
        for node_id in element_nodes:
            nodal_stress_sum[node_id] += stress
            nodal_contribution_count[node_id] += 1
            
    nodal_contribution_count[nodal_contribution_count == 0] = 1
    
    nodal_avg_stress = nodal_stress_sum / nodal_contribution_count
    return nodal_avg_stress

def _create_id_to_index_map(mesh: pv.UnstructuredGrid) -> dict:
    if 'persistent_ids' not in mesh.point_data:
        raise ValueError("Mesh is missing 'persistent_ids' in point_data.")
    persistent_ids = mesh.point_data['persistent_ids']
    return {pid: i for i, pid in enumerate(persistent_ids)}

def _check_for_disconnected_components(elements, num_nodes, fixed_indices):
    if num_nodes == 0 or len(fixed_indices) == 0:
        return False
    adj = [[] for _ in range(num_nodes)]
    for element in elements:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = element[i], element[j]
                adj[u].append(v)
                adj[v].append(u)
    
    adj_list_unique = [np.unique(np.array(neighbors, dtype=np.int64)) for neighbors in adj]
    
    return _run_bfs_check(num_nodes, adj_list_unique, np.array(fixed_indices, dtype=np.int64))


@numba.jit(nopython=True, cache=True)
def _run_bfs_check(num_nodes, adj_list, fixed_indices):
    if num_nodes == 0 or len(fixed_indices) == 0:
        return False
        
    visited = np.zeros(num_nodes, dtype=np.bool_)
    q = np.empty(num_nodes, dtype=np.int64)
    q_start, q_end = 0, 0
    
    start_node = fixed_indices[0]
    visited[start_node] = True
    q[q_end] = start_node
    q_end += 1

    while q_start < q_end:
        u = q[q_start]
        q_start += 1
        for v in adj_list[u]:
            if not visited[v]:
                visited[v] = True
                q[q_end] = v
                q_end += 1
    
    return np.all(visited)


def run_native_fea(mesh: pv.UnstructuredGrid, material: dict, fixed_node_indices: list, 
                   loaded_node_indices: list, force: tuple, log_func=print, 
                   progress_callback=None, stress_percentile_threshold=90.0,
                   disp_node_data: dict = None):
    """
    Performs a high-performance, memory-efficient FEA analysis using a direct
    matrix solution (partitioning) method for boundary conditions.
    """
    log_func("Step 0: Translating frontend IDs to solver indices...")
    try:
        id_map = _create_id_to_index_map(mesh)
        fixed_indices_solver = [id_map[fid] for fid in fixed_node_indices if fid in id_map]
        loaded_indices_solver = [id_map[lid] for lid in loaded_node_indices if lid in id_map]
        disp_data_solver = {}
        if disp_node_data:
            for pid, disp_vector in disp_node_data.items():
                if pid in id_map:
                    disp_data_solver[id_map[pid]] = disp_vector
        log_func("...ID translation successful.")
    except (ValueError, KeyError) as e:
        log_func(f"FATAL ERROR during ID translation: {e}"); raise e

    def _update_progress(percent, message):
        log_func(message)
        if progress_callback: progress_callback(percent, message)

    _update_progress(0, "Step 1: Preparing data for solver (using SI units)...")
    nodes = mesh.points
    num_nodes = nodes.shape[0]
    elements = mesh.cells_dict.get(pv.CellType.TETRA)
    if elements is None: raise TypeError("Mesh does not contain tetrahedral (TET4) cells.")
    
    E = material["ex"]; nu = material["prxy"]
    c1=E/((1+nu)*(1-2*nu)); c2=(1-nu); c3=nu; c4=(1-2*nu)/2
    D_matrix = c1*np.array([[c2,c3,c3,0,0,0],[c3,c2,c3,0,0,0],[c3,c3,c2,0,0,0],[0,0,0,c4,0,0],[0,0,0,0,c4,0],[0,0,0,0,0,c4]])
    
    _update_progress(10, "Step 2: Assembling Stiffness Matrix...")
    num_dofs = num_nodes * 3
    data, rows, cols, all_valid_elements_mask = _assemble_k_global_data(elements, nodes, D_matrix)
    num_invalid_elements = len(elements) - np.sum(all_valid_elements_mask)
    if num_invalid_elements > 0:
        valid_entries_mask = np.repeat(all_valid_elements_mask, 144)
        data = data[valid_entries_mask]
        rows = rows[valid_entries_mask]
        cols = cols[valid_entries_mask]
    
    K_original = coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()
    _update_progress(50, "Assembly complete.")

    if num_invalid_elements > 0:
        log_func(f"WARNING: Found and discarded {num_invalid_elements} invalid elements (zero/negative volume).")
    
    # ---------------------- DIRECT MATRIX SOLUTION (WITH INDEXING CORRECTION) ----------------------
    
    _update_progress(51, "Step 3: Applying BCs using direct matrix partitioning...")

    F_applied = np.zeros(num_dofs)
    if loaded_indices_solver:
        force_per_node = np.array(force, dtype=np.float64) / len(loaded_indices_solver)
        for node_id in loaded_indices_solver:
            F_applied[node_id*3:node_id*3+3] = force_per_node

    U_p = np.zeros(num_dofs)
    if disp_data_solver:
        for node_id, disp_vector in disp_data_solver.items():
            for i in range(3):
                U_p[node_id*3 + i] = disp_vector[i]
    
    fixed_dofs = np.array([dof for node_id in fixed_indices_solver for dof in (node_id*3, node_id*3+1, node_id*3+2)], dtype=np.int64)
    disp_dofs = np.array([dof for node_id in disp_data_solver.keys() for dof in (node_id*3, node_id*3+1, node_id*3+2)], dtype=np.int64)
    prescribed_dofs = np.union1d(fixed_dofs, disp_dofs)
    
    all_dofs = np.arange(num_dofs)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs, assume_unique=True)
    
    log_func(f"...Identified {len(free_dofs)} free DOFs and {len(prescribed_dofs)} prescribed DOFs.")
    
    U_p_values = U_p[prescribed_dofs]
    
    log_func("...Partitioning system matrices K_ff and K_fp.")
    K_ff = K_original[free_dofs, :][:, free_dofs]
    K_fp = K_original[free_dofs, :][:, prescribed_dofs]
    
    F_f = F_applied[free_dofs]

    log_func("...Adjusting force vector for prescribed displacements.")
    F_solver = F_f - K_fp.dot(U_p_values)
    
    f_norm = np.linalg.norm(F_solver)
    log_func(f"...Norm of solver force vector F_solver: {f_norm}")
    if f_norm < 1e-9 and len(free_dofs) > 0:
        log_func("WARNING: Solver force vector is near-zero. Resulting free displacements will also be zero.")

    # ---------------------- END OF BC SECTION ----------------------

    _update_progress(59, "Step 3.5: Checking model stability...")
    all_constrained_indices = np.union1d(fixed_indices_solver, np.array(list(disp_data_solver.keys()), dtype=np.int64))
    if all_constrained_indices.size == 0:
        raise RuntimeError("Solver aborted: Model is not stable. At least one node must be fixed.")
    is_stable = _check_for_disconnected_components(elements, num_nodes, all_constrained_indices)
    if not is_stable:
        raise RuntimeError("Solver aborted: Model is not stable. The mesh may have disconnected components or fixed BCs are insufficient.")
    log_func("...Model stability check passed.")

    _update_progress(60, "Step 4: Solving the reduced system of equations...")
    
    U_f = np.zeros(len(free_dofs))
    if len(free_dofs) > 0 and f_norm > 1e-9:
        log_func(f"...Solving for {len(U_f)} unknown displacements.")
        M_diag = K_ff.diagonal()
        M_diag[np.abs(M_diag) < 1e-12] = 1.0
        M_inv = 1.0 / M_diag
        preconditioner = LinearOperator((len(free_dofs), len(free_dofs)), matvec=lambda v: M_inv * v)
        
        U_f, info = cg(K_ff, F_solver, rtol=1e-4, maxiter=500, M=preconditioner)
        
        if info != 0:
            log_func(f"WARNING: Solver did not converge in 500 iterations.")
    
    _update_progress(80, "System solved successfully.")

    log_func("...Reconstructing full displacement vector.")
    U = np.zeros(num_dofs)
    U[prescribed_dofs] = U_p_values
    U[free_dofs] = U_f
    displacements = U.reshape((num_nodes, 3))

    _update_progress(81, "Step 5: Post-processing results...")
    valid_elements = elements[all_valid_elements_mask]
    
    element_stresses = _calculate_stress_for_valid_elements(valid_elements, nodes, displacements, D_matrix)
    element_von_mises = np.sqrt(0.5*((element_stresses[:,0]-element_stresses[:,1])**2 + (element_stresses[:,1]-element_stresses[:,2])**2 + (element_stresses[:,2]-element_stresses[:,0])**2) + 3*(element_stresses[:,3]**2 + element_stresses[:,4]**2 + element_stresses[:,5]**2))
    
    # --- START: New Stress Filtering Logic ---
    if element_von_mises.size > 1:
        # Use the percentile from the function's arguments
        percentile_value = np.percentile(element_von_mises, stress_percentile_threshold)
        
        # Find values *above* this percentile
        high_stress_values = element_von_mises[element_von_mises > percentile_value]
        
        if high_stress_values.size > 0:
            # Calculate the average of *only* those high values
            filter_threshold = np.mean(high_stress_values)
            log_func(f"...Stress filter activated: Values above {stress_percentile_threshold}th percentile ({percentile_value:.2e} Pa) will be capped at their average ({filter_threshold:.2e} Pa).")
            
            # Cap all values in element_von_mises at this new threshold
            np.clip(element_von_mises, a_min=None, a_max=filter_threshold, out=element_von_mises)
        else:
            log_func(f"...Stress filter: No values found above {stress_percentile_threshold}th percentile. No capping applied.")
    # --- END: New Stress Filtering Logic ---
            
    log_func("...Calculating elemental strain energy.")
    element_strain_energy = _calculate_strain_energy_for_valid_elements(valid_elements, nodes, displacements, D_matrix)

    _update_progress(90, "Interpolating stress field to nodes...")
    nodal_von_mises = _average_element_stress_to_nodes(num_nodes, valid_elements, element_von_mises)

    # --- START OF MODIFIED REACTION FORCE CALCULATION ---

    log_func("...Calculating reaction forces at constraints.")
    F_internal = K_original.dot(U)
    
    # In a pure displacement case, F_applied is zero, so F_reaction = F_internal.
    # For correctness in mixed-load cases, we should use: F_reaction = F_internal - F_applied
    F_reaction_vec = F_internal - F_applied # F_applied was defined in the BC section

    # --- NEW: Calculate reaction force on FIXED nodes only ---
    total_reaction_force_fixed = np.zeros(3)
    if fixed_dofs.size > 0:
        # Select the reaction forces corresponding to the fixed DOFs
        reactions_at_fixed_dofs = F_reaction_vec[fixed_dofs]
        # Reshape into (num_fixed_nodes, 3) and sum along the columns (axis=0)
        total_reaction_force_fixed = reactions_at_fixed_dofs.reshape(-1, 3).sum(axis=0)

    # --- NEW: Calculate reaction force on DISPLACEMENT LOAD nodes only ---
    total_reaction_force_disp = np.zeros(3)
    if disp_dofs.size > 0:
        # Select the reaction forces corresponding to the displacement-loaded DOFs
        reactions_at_disp_dofs = F_reaction_vec[disp_dofs]
        # Reshape and sum
        total_reaction_force_disp = reactions_at_disp_dofs.reshape(-1, 3).sum(axis=0)

    # Calculate the total reaction force for equilibrium check (should be ~zero)
    total_reaction_force_system = F_reaction_vec.reshape(-1, 3).sum(axis=0)
    
    # --- END OF MODIFIED REACTION FORCE CALCULATION ---

    _update_progress(99, "Finalizing results...")
    result_mesh = mesh.copy()
    result_mesh.point_data["Displacements"] = displacements
    result_mesh.point_data["displacement"] = np.linalg.norm(displacements, axis=1)
    result_mesh.point_data["von_mises_stress"] = nodal_von_mises
    result_mesh.set_active_vectors('Displacements')

    # Store elemental (cell) data by mapping results for valid elements back to the full cell array
    full_strain_energy = np.zeros(len(elements))
    full_strain_energy[all_valid_elements_mask] = element_strain_energy
    result_mesh.cell_data["strain_energy"] = full_strain_energy
    
    full_von_mises = np.zeros(len(elements))
    full_von_mises[all_valid_elements_mask] = element_von_mises
    result_mesh.cell_data["von_mises_stress"] = full_von_mises

    # --- NEW: Calculate and store total strain energy ---
    total_strain_energy = np.sum(element_strain_energy)
    result_mesh.field_data["total_strain_energy_J"] = total_strain_energy

    # Store the new, more specific results
    result_mesh.field_data["total_reaction_force_fixed_N"] = total_reaction_force_fixed
    result_mesh.field_data["total_reaction_force_disp_load_N"] = total_reaction_force_disp
    result_mesh.field_data["total_reaction_force_system_N"] = total_reaction_force_system

    log_func("--- Force & Energy Analysis ---")
    log_func(f"Reaction Force on FIXED nodes (X,Y,Z):   {total_reaction_force_fixed} N")
    log_func(f"Reaction Force on DISP LOAD nodes (X,Y,Z): {total_reaction_force_disp} N")
    log_func(f"Total System Reaction Force (X,Y,Z):   {total_reaction_force_system} N")
    log_func(f"Total Strain Energy in model:            {total_strain_energy:.4f} J")

    _update_progress(100, "FEA simulation completed successfully.")

    return result_mesh
