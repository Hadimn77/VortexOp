# fea_solver_core.py
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import numba

@numba.jit(nopython=True, cache=True)
def _get_element_stiffness_tet4(node_coords, D_matrix):
    """
    Calculates the stiffness matrix for a single linear tetrahedral element (TET4).
    """
    M = np.ones((4, 4))
    M[:, 1:] = node_coords
    
    volume = np.linalg.det(M) / 6.0
    if volume <= 1e-12: # Use a small tolerance for floating point comparisons
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
def _calculate_batch_element_data(batch_elements, nodes, D_matrix, batch_kes, batch_dof_indices, batch_is_valid):
    """
    Stage 1 (Parallel): Calculates Ke and DOF indices for a BATCH of elements in parallel.
    """
    for i in numba.prange(len(batch_elements)):
        node_ids = batch_elements[i]
        
        ke, is_valid = _get_element_stiffness_tet4(nodes[node_ids], D_matrix)
        batch_is_valid[i] = is_valid

        if is_valid:
            batch_kes[i] = ke
            dof_indices = np.empty(12, dtype=np.int64)
            for j in range(4):
                node_id = node_ids[j]
                dof_indices[j*3] = node_id * 3
                dof_indices[j*3+1] = node_id * 3 + 1
                dof_indices[j*3+2] = node_id * 3 + 2
            batch_dof_indices[i] = dof_indices

@numba.jit(nopython=True, cache=True, parallel=True)
def _calculate_stress_for_valid_elements(valid_elements, nodes, displacements, D_matrix):
    """
    Stage 3 (Parallel): Calculates stresses and centroids for valid elements.
    """
    num_valid_elements = len(valid_elements)
    element_stresses = np.zeros((num_valid_elements, 6))
    element_centroids = np.zeros((num_valid_elements, 3))

    for i in numba.prange(num_valid_elements):
        node_ids = valid_elements[i]
        element_nodes_coords = nodes[node_ids]

        M = np.ones((4, 4))
        M[:, 1:] = element_nodes_coords
        
        if abs(np.linalg.det(M)) < 1e-12:
            continue

        element_displacements = displacements[node_ids].flatten()
        
        sum_of_coords = np.sum(element_nodes_coords, axis=0)
        element_centroids[i] = sum_of_coords / 4.0

        dN_dx = np.linalg.inv(M).T
        B = np.zeros((6, 12))
        for j in range(4):
            dN_j_dx, dN_j_dy, dN_j_dz = dN_dx[j, 1], dN_dx[j, 2], dN_dx[j, 3]
            col = j * 3
            B[0, col]=dN_j_dx; B[1, col+1]=dN_j_dy; B[2, col+2]=dN_j_dz
            B[3, col]=dN_j_dy; B[3, col+1]=dN_j_dx
            # --- MODIFICATION: Corrected the shear term calculation ---
            B[4, col+1]=dN_j_dz; B[4, col+2]=dN_j_dy
            B[5, col]=dN_j_dz; B[5, col+2]=dN_j_dx
        element_stresses[i] = D_matrix @ B @ element_displacements
        
    return element_stresses, element_centroids

def _create_id_to_index_map(mesh: pv.UnstructuredGrid) -> dict:
    """
    Creates a mapping from original, persistent node IDs to the 0-based
    array index used by the solver.
    """
    if 'persistent_ids' not in mesh.point_data:
        raise ValueError(
            "Mesh is missing 'persistent_ids' in point_data. "
            "Cannot map frontend IDs to solver indices. Please regenerate the mesh."
        )
    
    persistent_ids = mesh.point_data['persistent_ids']
    id_to_index_map = {pid: i for i, pid in enumerate(persistent_ids)}
    
    return id_to_index_map

@numba.jit(nopython=True, cache=True)
def _check_for_disconnected_components(elements, num_nodes, fixed_indices):
    """
    Checks if all nodes in the mesh are connected to at least one fixed node
    through a path of elements. Prevents free-body motion failures.
    """
    if num_nodes == 0 or len(fixed_indices) == 0:
        return False

    adj = [np.empty(0, dtype=np.int64) for _ in range(num_nodes)]
    for i in range(len(elements)):
        element = elements[i]
        for j in range(4):
            for k in range(j + 1, 4):
                u, v = element[j], element[k]
                adj[u] = np.append(adj[u], v)
                adj[v] = np.append(adj[v], u)

    visited = np.zeros(num_nodes, dtype=np.bool_)
    q = np.empty(num_nodes, dtype=np.int64)
    q_start, q_end = 0, 0

    for node_idx in fixed_indices:
        if not visited[node_idx]:
            visited[node_idx] = True
            q[q_end] = node_idx
            q_end += 1

    while q_start < q_end:
        u = q[q_start]
        q_start += 1
        neighbors = np.unique(adj[u])
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                q[q_end] = v
                q_end += 1

    return np.all(visited)


def run_native_fea(mesh: pv.UnstructuredGrid, material: dict, fixed_node_indices: list, 
                   loaded_node_indices: list, force: tuple, log_func=print, 
                   progress_callback=None, stress_percentile_threshold=99.0):
    """
    Performs a high-performance, memory-efficient FEA analysis using batch processing.
    All calculations are performed in SI units (meters, Newtons, Pascals).
    """
    log_func("Step 0: Translating frontend IDs to solver indices...")
    try:
        id_map = _create_id_to_index_map(mesh)
        
        fixed_indices_solver = [id_map[fid] for fid in fixed_node_indices if fid in id_map]
        loaded_indices_solver = [id_map[lid] for lid in loaded_node_indices if lid in id_map]
        log_func("...ID translation successful.")

    except (ValueError, KeyError) as e:
        log_func(f"FATAL ERROR during ID translation: {e}")
        raise e

    def _update_progress(percent, message):
        log_func(message)
        if progress_callback: progress_callback(percent, message)

    _update_progress(0, "Step 1: Preparing data for solver (using SI units)...")
    
    nodes = mesh.points
    num_nodes = nodes.shape[0]
    elements = mesh.cells_dict.get(pv.CellType.TETRA)
    if elements is None: raise TypeError("Mesh does not contain any tetrahedral (TET4) cells.")
    num_elements = len(elements)
    
    E = material["ex"]; nu = material["prxy"]
    c1=E/((1+nu)*(1-2*nu)); c2=(1-nu); c3=nu; c4=(1-2*nu)/2
    D_matrix = c1*np.array([[c2,c3,c3,0,0,0],[c3,c2,c3,0,0,0],[c3,c3,c2,0,0,0],[0,0,0,c4,0,0],[0,0,0,0,c4,0],[0,0,0,0,0,c4]])
    
    _update_progress(10, "Step 2: Assembling Stiffness Matrix in batches...")
    num_dofs = num_nodes * 3
    
    K_global = csr_matrix((num_dofs, num_dofs), dtype=np.float64)
    batch_size = 100000 
    num_batches = (num_elements + batch_size - 1) // batch_size
    all_valid_elements_mask = np.zeros(num_elements, dtype=np.bool_)

    num_invalid_elements = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_elements)
        batch_elements = elements[start_idx:end_idx]
        current_batch_size = len(batch_elements)
        
        batch_kes = np.zeros((current_batch_size, 12, 12), dtype=np.float64)
        batch_dof_indices = np.zeros((current_batch_size, 12), dtype=np.int64)
        batch_is_valid = np.zeros(current_batch_size, dtype=np.bool_)
        _calculate_batch_element_data(batch_elements, nodes, D_matrix, batch_kes, batch_dof_indices, batch_is_valid)
        
        num_invalid_in_batch = current_batch_size - np.sum(batch_is_valid)
        num_invalid_elements += num_invalid_in_batch
        
        all_valid_elements_mask[start_idx:end_idx] = batch_is_valid

        data_list, row_list, col_list = [], [], []
        valid_batch_indices = np.where(batch_is_valid)[0]
        for j in valid_batch_indices:
            ke = batch_kes[j]
            dof_indices = batch_dof_indices[j]
            rows, cols = np.meshgrid(dof_indices, dof_indices, indexing='ij')
            data_list.append(ke.flatten())
            row_list.append(rows.flatten())
            col_list.append(cols.flatten())
        
        if data_list:
            data = np.concatenate(data_list)
            rows = np.concatenate(row_list)
            cols = np.concatenate(col_list)
            K_global += coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs))

    K_original = K_global.copy()
    _update_progress(50, "Assembly complete.")

    if num_invalid_elements > 0:
        log_func(f"WARNING: Found and discarded {num_invalid_elements} invalid elements (zero/negative volume). "
                 f"High numbers can indicate poor mesh quality and may lead to instability.")
    
    _update_progress(51, "Step 3: Applying boundary conditions and loads...")
    F_global = np.zeros(num_dofs)
    if loaded_indices_solver:
        force_per_node = np.array(force, dtype=np.float64) / len(loaded_indices_solver)
        for node_id in loaded_indices_solver: 
            F_global[node_id*3:node_id*3+3] = force_per_node
            
    fixed_dofs = np.array([dof for node_id in fixed_indices_solver for dof in (node_id*3,node_id*3+1,node_id*3+2)], dtype=np.int64)
    K_global = K_global.tocsr()
    for dof in fixed_dofs:
        K_global.data[K_global.indptr[dof]:K_global.indptr[dof+1]] = 0
    transpose = K_global.transpose().tocsr()
    for dof in fixed_dofs:
        transpose.data[transpose.indptr[dof]:transpose.indptr[dof+1]] = 0
    K_global = transpose.transpose()
    K_global[fixed_dofs, fixed_dofs] = 1.0
    F_global[fixed_dofs] = 0.0

    _update_progress(59, "Step 3.5: Checking model stability...")
    is_stable = _check_for_disconnected_components(elements, num_nodes, np.array(fixed_indices_solver))
    if not is_stable:
        raise RuntimeError("Solver aborted: Model is not stable. "
                           "This is likely due to insufficient constraints (free body motion) "
                           "or disconnected element groups. Please check your fixed BCs.")
    log_func("...Model stability check passed.")

    _update_progress(60, "Step 4: Solving the system of equations...")
    max_iterations = 300
    iteration_counter = [0] 
    def solver_callback(xk):
        iteration_counter[0] += 1
        progress_percent = 60 + (iteration_counter[0] / max_iterations) * 20
        if progress_callback: _update_progress(int(progress_percent), f"Solving... (Iteration {iteration_counter[0]})")
    
    U, info = cg(K_global, F_global, rtol=1e-6, maxiter=max_iterations, callback=solver_callback)
    
    if info == 0: 
        _update_progress(80, "System solved successfully.")
    elif info > 0:
        log_func(f"WARNING: Solver did not converge to the desired tolerance within {max_iterations} iterations (info={info}). Results may be inaccurate.")
    else:
        raise RuntimeError(f"Solver failed due to a numerical issue (info={info}). This often indicates an unstable model (insufficient constraints).")
        
    displacements = U.reshape((num_nodes, 3))

    _update_progress(81, "Step 5: Post-processing results...")
    valid_elements = elements[all_valid_elements_mask]
    
    element_stresses, _ = _calculate_stress_for_valid_elements(
        valid_elements, nodes, displacements, D_matrix
    )
    
    s_elem = element_stresses
    element_von_mises = np.sqrt(0.5*((s_elem[:,0]-s_elem[:,1])**2 + (s_elem[:,1]-s_elem[:,2])**2 + (s_elem[:,2]-s_elem[:,0])**2) + 3*(s_elem[:,3]**2 + s_elem[:,4]**2 + s_elem[:,5]**2))
    
    valid_indices = np.where(all_valid_elements_mask)[0]
    
    if stress_percentile_threshold < 100.0 and element_stresses.size > 0:
        log_func(f"...Applying stress outlier filter at {stress_percentile_threshold}th percentile.")
        if element_von_mises.size > 0:
            stress_limit = np.percentile(element_von_mises, stress_percentile_threshold)
            filter_mask = element_von_mises < stress_limit
            element_stresses = element_stresses[filter_mask]
            element_von_mises = element_von_mises[filter_mask]
            final_valid_indices = valid_indices[filter_mask]
    else:
        final_valid_indices = valid_indices

    _update_progress(95, "Calculating principal stresses for each element...")
    num_final_elements = element_stresses.shape[0]
    stress_tensors = np.zeros((num_final_elements, 3, 3))
    
    s_elem_final = element_stresses
    stress_tensors[:, 0, 0] = s_elem_final[:, 0]; stress_tensors[:, 1, 1] = s_elem_final[:, 1]; stress_tensors[:, 2, 2] = s_elem_final[:, 2]
    stress_tensors[:, 0, 1] = stress_tensors[:, 1, 0] = s_elem_final[:, 3]
    stress_tensors[:, 1, 2] = stress_tensors[:, 2, 1] = s_elem_final[:, 4]
    stress_tensors[:, 0, 2] = stress_tensors[:, 2, 0] = s_elem_final[:, 5]
    
    principal_stresses = np.sort(np.linalg.eigvalsh(stress_tensors), axis=1)[:, ::-1]

    F_reaction_vec = K_original.dot(U)
    total_reaction_force = np.zeros(3)
    for dof in fixed_dofs: total_reaction_force[dof % 3] += F_reaction_vec[dof]
    
    _update_progress(99, "Finalizing results...")
    result_mesh = mesh.copy()
    
    result_mesh.point_data["displacement"] = np.linalg.norm(displacements, axis=1)
    result_mesh.point_data['Displacements'] = displacements

    full_element_von_mises = np.zeros(num_elements)
    full_principal_s1 = np.zeros(num_elements)
    full_principal_s2 = np.zeros(num_elements)
    full_principal_s3 = np.zeros(num_elements)
    
    if len(final_valid_indices) == principal_stresses.shape[0]:
        full_element_von_mises[final_valid_indices] = element_von_mises
        full_principal_s1[final_valid_indices] = principal_stresses[:, 0]
        full_principal_s2[final_valid_indices] = principal_stresses[:, 1]
        full_principal_s3[final_valid_indices] = principal_stresses[:, 2]

    result_mesh.cell_data["von_mises_stress"] = full_element_von_mises
    result_mesh.cell_data["principal_s1"] = full_principal_s1
    result_mesh.cell_data["principal_s2"] = full_principal_s2
    result_mesh.cell_data["principal_s3"] = full_principal_s3
    
    result_mesh.set_active_vectors('Displacements')
    result_mesh.field_data["total_reaction_force_N"] = total_reaction_force
    log_func(f"Total Reaction Force (X,Y,Z): {total_reaction_force} N")
    _update_progress(100, "FEA simulation completed successfully.")

    return result_mesh
