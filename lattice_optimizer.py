import os
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from lattice_utils import generate_infill_inside
from fea_utils import create_robust_volumetric_mesh
from fea_solver_core import run_native_fea
from unit_utils import UnitManager

# Bayesian Optimization requires the scikit-optimize library
try:
    from skopt import Optimizer
    from skopt.space import Real, Categorical
    from skopt.plots import plot_convergence
except ImportError:
    raise ImportError("Bayesian optimization requires 'scikit-optimize'. Please install it using: pip install scikit-optimize")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

## --------------------------------------------------------------------------
## Helper Functions
## --------------------------------------------------------------------------

def get_robust_max_displacement(mesh):
    """Calculates a robust maximum displacement value from a mesh."""
    if "displacement" in mesh.point_data and mesh.n_points > 0:
        displacements = mesh.point_data["displacement"]
        if displacements.size == 0: return 0.0
        # Use 99.5th percentile to avoid extreme outliers from solver artifacts
        return np.percentile(displacements, 99.5)
    return 1e12 # Return a large number if displacement data is not found

def get_robust_max_stress(mesh):
    """Calculates a robust maximum stress value from a mesh."""
    if "von_mises_stress" in mesh.point_data and mesh.n_points > 0 and mesh.point_data["von_mises_stress"].size > 0:
        return np.max(mesh.point_data['von_mises_stress'])
    elif "von_mises_stress" in mesh.cell_data and mesh.n_cells > 0 and mesh.cell_data["von_mises_stress"].size > 0:
        return np.max(mesh.cell_data["von_mises_stress"])
    return 1e12 # Return a large number if stress data is not found

def _remap_indices_by_proximity(new_mesh: pv.UnstructuredGrid, original_coords: np.ndarray, search_radius: float):
    """
    Finds the new node indices on a new mesh that correspond to original coordinates
    within a specified search radius.
    """
    if original_coords.size == 0:
        return []
    if 'persistent_ids' not in new_mesh.point_data:
        new_mesh.point_data['persistent_ids'] = np.arange(new_mesh.n_points)
    
    kdtree = KDTree(new_mesh.points)
    distances, indices = kdtree.query(original_coords, distance_upper_bound=search_radius)
    
    valid_indices_mask = distances != np.inf
    
    if not np.any(valid_indices_mask):
        return []
    
    valid_original_indices = indices[valid_indices_mask]
    
    return list(new_mesh.point_data['persistent_ids'][valid_original_indices])

def _create_normalized_scalar_field(control_points_coords, control_point_values, full_target_points):
    """Interpolates control point values to a full field and ensures values are [0, 1]."""
    full_scalar_values = griddata(
        control_points_coords, control_point_values, full_target_points,
        method='linear', fill_value=np.mean(control_point_values)
    )

    np.clip(full_scalar_values, 0.0, 1.0, out=full_scalar_values)

    max_scalar_value = np.max(full_scalar_values)
    min_scalar_value = np.min(full_scalar_values)
    
    if (max_scalar_value - min_scalar_value) < 1e-9:
        normalized_scalar_field = np.full_like(full_scalar_values, 0.5)
    else:
        normalized_scalar_field = (full_scalar_values - min_scalar_value) / (max_scalar_value - min_scalar_value)

    return normalized_scalar_field
    
def smooth_scalar_field(points, scalar_values, k_neighbors=10, num_iterations=5):
    """
    Smooths a scalar field using k-nearest neighbors averaging.
    """
    if points.shape[0] < k_neighbors:
        return scalar_values
        
    kdtree = KDTree(points)
    smoothed_values = scalar_values.copy()
    
    for _ in range(num_iterations):
        _, nearest_indices = kdtree.query(points, k=k_neighbors)
        temp_values = np.mean(smoothed_values[nearest_indices], axis=1)
        smoothed_values = temp_values
        
    return np.round(smoothed_values, 2)

def _map_fea_result_to_control_values(fea_mesh, control_points_coords, unit_manager, log_func, scalar_field_source='von_mises_stress'):
    """Takes a mesh with FEA data and maps it to normalized [0,1] control point values."""
    if fea_mesh is None:
        log_func("No FEA mesh provided for mapping; using uniform 0.5.", "warning")
        return [0.5] * len(control_points_coords)

    source_data = None
    source_points = None
    unit_type = ''

    if scalar_field_source == 'strain_energy':
        log_func("...mapping Strain Energy field to control points.")
        if 'strain_energy' in fea_mesh.cell_data and fea_mesh.cell_data['strain_energy'].size > 0:
            source_data = fea_mesh.cell_data['strain_energy']
            source_points = fea_mesh.cell_centers().points
            unit_type = 'energy'
        else:
            log_func("Could not find strain energy data, falling back to stress.", "warning")
            scalar_field_source = 'von_mises_stress' # Explicitly set for fallback
    
    if scalar_field_source == 'von_mises_stress':
        if source_data is None: # Check if we fell back from strain energy
            log_func("...mapping von Mises Stress field to control points.")
        
        if 'von_mises_stress' in fea_mesh.point_data and fea_mesh.point_data['von_mises_stress'].size > 0:
            source_data = fea_mesh.point_data['von_mises_stress']
            source_points = fea_mesh.points
            unit_type = 'pressure'
        elif 'von_mises_stress' in fea_mesh.cell_data and fea_mesh.cell_data['von_mises_stress'].size > 0:
            source_data = fea_mesh.cell_data['von_mises_stress']
            source_points = fea_mesh.cell_centers().points
            unit_type = 'pressure'

    if source_data is None or source_points is None:
        log_func(f"No valid data found for '{scalar_field_source}'; using uniform 0.5.", "warning")
        return [0.5] * len(control_points_coords)

    values_solver = unit_manager.convert_to_solver(np.asarray(source_data), unit_type)
    
    global_min, global_max = np.min(values_solver), np.max(values_solver)
    if (global_max - global_min) < 1e-9:
        normalized_values = np.full_like(values_solver, 0.5)
    else:
        normalized_values = (values_solver - global_min) / (global_max - global_min)

    k_neighbors = min(10, len(source_points))
    if k_neighbors > 0:
        kdtree = KDTree(source_points)
        _, nearest_indices = kdtree.query(control_points_coords, k=k_neighbors)
        if k_neighbors > 1 and nearest_indices.ndim > 1:
            return list(np.mean(normalized_values[nearest_indices], axis=1))
        else:
            return list(normalized_values[nearest_indices])
            
    return [0.5] * len(control_points_coords)

## --------------------------------------------------------------------------
## Main Optimization Loop
## --------------------------------------------------------------------------

def run_optimization_loop(
    initial_fea_mesh: pv.UnstructuredGrid,
    original_shell: pv.PolyData,
    lattice_params: dict,
    remesh_params: dict,
    meshing_params: dict,
    fea_params: dict,
    optim_params: dict,
    unit_manager: UnitManager,
    log_func,
    progress_callback
):
    """
    Runs a deterministic Bayesian optimization, saving all intermediate results.
    """
    log_func("--- Starting Bayesian Lattice Optimization ---")

    output_dir = "optimization_steps"
    os.makedirs(output_dir, exist_ok=True)
    log_func(f"Intermediate results will be saved in '{output_dir}/'")

    # Initialize a list to store log data for each iteration
    optimization_log = []
    
    if 'persistent_ids' not in initial_fea_mesh.point_data:
        initial_fea_mesh.point_data['persistent_ids'] = np.arange(initial_fea_mesh.n_points)
        
    # --- 1. Initial Setup ---
    search_radius = meshing_params.get('detail_size', 1.0) * 2.0
    initial_ids = initial_fea_mesh.point_data['persistent_ids']
    fixed_mask = np.isin(initial_ids, fea_params.get("fixed_node_indices", []))
    loaded_mask = np.isin(initial_ids, fea_params.get("loaded_node_indices", []))
    original_fixed_coords = initial_fea_mesh.points[fixed_mask]
    original_loaded_coords = initial_fea_mesh.points[loaded_mask]
    
    # Store original displacement definitions (coord + UI vector) for remapping
    original_disp_definitions = []
    disp_node_data_ui = fea_params.get("disp_node_data", {})
    if disp_node_data_ui:
        id_to_idx = {pid: i for i, pid in enumerate(initial_ids)}
        for pid, vector_ui in disp_node_data_ui.items():
            if pid in id_to_idx:
                node_index = id_to_idx[pid]
                original_disp_definitions.append({
                    "coord": initial_fea_mesh.points[node_index],
                    "vector_ui": vector_ui 
                })
        log_func(f"Stored {len(original_disp_definitions)} prescribed displacement definitions for remapping.")

    patience_counter = 0
    last_best_score = np.inf
    patience = optim_params.get('patience', 10)
    tolerance = optim_params.get('tolerance', 1e-4)
    
    bounds = initial_fea_mesh.bounds
    grid_points = pv.ImageData(
        dimensions=(10, 10, 10), 
        spacing=((bounds[1]-bounds[0])/9, (bounds[3]-bounds[2])/9, (bounds[5]-bounds[4])/9),
        origin=(bounds[0], bounds[2], bounds[4])
    ).points
    _, nearest_node_indices = KDTree(initial_fea_mesh.points).query(grid_points)
    control_point_indices = np.unique(nearest_node_indices)
    control_points_coords = initial_fea_mesh.points[control_point_indices]
    full_target_points = initial_fea_mesh.points

    initial_max_stress_ui = get_robust_max_stress(initial_fea_mesh)
    initial_total_strain_energy = initial_fea_mesh.field_data.get("total_strain_energy_J", 1.0)
    if initial_total_strain_energy < 1e-9: initial_total_strain_energy = 1.0

    try:
        initial_volume = original_shell.volume
        if initial_volume < 1e-9: initial_volume = 1.0
    except Exception:
        initial_volume = 1.0

    # --- 2. Define Optimization Search Space ---
    user_min_limit = optim_params.get('min_thickness', 0.5)
    user_max_limit = optim_params.get('max_thickness', 5.0)
    available_lattice_types = optim_params.get('available_lattice_types', ['gyroid'])
    
    search_space = [
        Real(user_min_limit, user_max_limit, name='max_thick'),
        Real(0.0, 1.0, name = 'thickness_ratio'),
        Categorical(available_lattice_types, name='lattice_type')
    ]
    
    # seed = optim_params.get('random_state', 11)
    optimizer = Optimizer(
        dimensions=search_space, 
        base_estimator="RF", 
        acq_func="EI"
        #random_state=seed
    )

    # --- 3. Define the Objective Function to Minimize ---
    iteration_data = {
        "counter": 1, "best_score": np.inf, "best_params": {},
        "best_mesh": initial_fea_mesh.copy(), "last_successful_mesh": initial_fea_mesh.copy(), 'best_mesh_iteration':1,
        "results": {}
    }
    
    def objective(params):
        iteration = iteration_data["counter"]
        iteration_data["counter"] += 1
        
        iteration_paths = {
            'scalar_path': '', 'lattice_path': '', 'fea_result_path': ''
        }
        
        # --- FIX 1: Unpack all THREE parameters ---
        max_thick_suggested, thickness_ratio, lattice_type_suggested = params
        
        log_entry = {
            "iteration": iteration,
            "lattice_type": lattice_type_suggested,
            "volume_fraction": "N/A",
            "max_stress": "N/A",
            "max_displacement": "N/A",
            "total_strain_energy": "N/A",
            "objective_score": 1e12
        }

        # --- FIX 2: 'thickness_ratio' variable is now defined and used correctly ---
        min_thick_bound = optim_params.get('min_thickness', 0.5)
        min_thick_calculated = max_thick_suggested - thickness_ratio * (max_thick_suggested - min_thick_bound)
        min_thick_calculated = max(min_thick_calculated, min_thick_bound)
        
        # Enforce min_thick < max_thick (same logic as before)
        if min_thick_calculated >= max_thick_suggested:
            min_thick_calculated = max_thick_suggested - 0.1 # Nudge it down
            if min_thick_calculated < min_thick_bound:
                 min_thick_calculated = min_thick_bound # Don't go below the absolute bound
        
        scalar_field_source = optim_params.get('scalar_field_source', 'von_mises_stress')

        log_func(f"\n--- Iteration {iteration} ---")
        log_func(f"Suggested: max_thick={max_thick_suggested:.3f}, thickness_ratio={thickness_ratio:.3f}, type={lattice_type_suggested}")
        log_func(f"Calculated min_thick (from ratio): {min_thick_calculated:.3f}")

        control_values = _map_fea_result_to_control_values(
            iteration_data["best_mesh"], control_points_coords, unit_manager, log_func, scalar_field_source
        )
        full_scalar_values = _create_normalized_scalar_field(control_points_coords, control_values, full_target_points)
        
        smoothed_scalar_values = smooth_scalar_field(full_target_points, full_scalar_values)
        
        try:
            scalar_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_scalar.txt")
            scalar_data_to_save = np.hstack([full_target_points, smoothed_scalar_values[:, np.newaxis]])
            np.savetxt(scalar_filename, scalar_data_to_save, header='X Y Z NormalizedValue', fmt='%.6f')
            iteration_paths['scalar_path'] = scalar_filename
        except Exception as e:
            log_func(f"WARNING: Could not save scalar field for iteration {iteration}. Reason: {e}", "warning")

        current_lattice_params = {**lattice_params,
            'external_scalar': (full_target_points, smoothed_scalar_values), 'use_scalar_for_thickness': True,
            'solidify': True, 'min_thickness_bound': min_thick_calculated,
            'max_thickness_bound': max_thick_suggested, 'lattice_type': lattice_type_suggested
        }
        
        # Remove keys that are for MANUAL generation, as they conflict
        # with the optimizer's parameters.
        current_lattice_params.pop('thickness', None)
        current_lattice_params.pop('use_volume_fraction', None)
        current_lattice_params.pop('target_volume_fraction', None)
        
        new_surface_mesh = generate_infill_inside(mesh=original_shell, log_func=log_func, **current_lattice_params, **remesh_params)

        if not new_surface_mesh or new_surface_mesh.n_points == 0:
            log_func("Lattice generation failed. Applying penalty.", "error")
            optimization_log.append(log_entry)
            return 1e12

        try:
            lattice_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_lattice.stl")
            new_surface_mesh.save(lattice_filename)
            iteration_paths['lattice_path'] = lattice_filename
        except Exception as e:
            log_func(f"WARNING: Could not save lattice mesh for iteration {iteration}. Reason: {e}", "warning")
        
        try:
            success, new_vol_mesh = create_robust_volumetric_mesh(surface_mesh=new_surface_mesh, **meshing_params, log_func=log_func)
        except:
            log_func("ERROR: Volumetric Meshing failed")
            optimization_log.append(log_entry)
            return 1e12

        if not success:
            log_func("ERROR: The Generated Volumetric Mesh is Empty")
            optimization_log.append(log_entry)
            return 1e12
        else:
            new_fixed_ids = _remap_indices_by_proximity(new_vol_mesh, original_fixed_coords, search_radius)
            new_loaded_ids = _remap_indices_by_proximity(new_vol_mesh, original_loaded_coords, search_radius)
        
        if not new_fixed_ids and not original_disp_definitions:
            log_func("Failed to remap any boundary conditions. Applying penalty.", "error")
            optimization_log.append(log_entry)
            return 1e12
        
        # Remap displacement BCs to the new mesh
        new_disp_node_data_ui = {}
        if original_disp_definitions:
            new_points = new_vol_mesh.points
            if 'persistent_ids' not in new_vol_mesh.point_data:
                 new_vol_mesh.point_data['persistent_ids'] = np.arange(new_vol_mesh.n_points)
            new_pids = new_vol_mesh.point_data['persistent_ids']
            kdtree = KDTree(new_points)
            
            original_coords = np.array([item['coord'] for item in original_disp_definitions])
            distances, indices = kdtree.query(original_coords, distance_upper_bound=search_radius)
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist != np.inf:
                    new_pid = new_pids[idx]
                    new_disp_node_data_ui[new_pid] = original_disp_definitions[i]['vector_ui']
            
            if len(new_disp_node_data_ui) != len(original_disp_definitions):
                log_func(f"Warning: Could only remap {len(new_disp_node_data_ui)} of {len(original_disp_definitions)} displacement BCs.", "warning")

        # Convert all relevant params to solver units for this iteration's FEA
        solver_mesh_input = new_vol_mesh.copy()
        solver_mesh_input.points = unit_manager.convert_to_solver(solver_mesh_input.points, 'length')
        force_vector_solver = tuple(unit_manager.convert_to_solver(np.array(f), 'force') for f in fea_params["force"])
        
        disp_data_solver = {}
        if new_disp_node_data_ui:
            for pid, disp_vector_ui in new_disp_node_data_ui.items():
                disp_vector_solver = unit_manager.convert_to_solver(np.array(disp_vector_ui), 'length')
                disp_data_solver[pid] = disp_vector_solver
        
        # Correctly get material properties for this iteration
        material_properties = fea_params["material"]

        current_fea_params = {**fea_params,
            'mesh': solver_mesh_input, 
            'fixed_node_indices': new_fixed_ids, 
            'loaded_node_indices': new_loaded_ids, 
            'force': force_vector_solver,
            'disp_node_data': disp_data_solver,
            'material': material_properties
        }

        try:
            result_mesh_solver = run_native_fea(**current_fea_params)
        except Exception as e:
            log_func(f"ERROR: FEA failed during optimization loop: {e}")
            optimization_log.append(log_entry)
            return 1e12

        if result_mesh_solver is None:
            log_func("FEA solver failed to produce a result. Applying penalty.", "error")
            optimization_log.append(log_entry)
            return 1e12
        
        result_mesh_ui = result_mesh_solver.copy()
        result_mesh_ui.points = unit_manager.convert_from_solver(result_mesh_solver.points, 'length')
        if 'Displacements' in result_mesh_solver.point_data:
            disp_solver = result_mesh_solver.point_data['Displacements']
            result_mesh_ui.point_data['Displacements'] = unit_manager.convert_from_solver(disp_solver, 'length')
            result_mesh_ui.point_data['displacement'] = np.linalg.norm(result_mesh_ui.point_data['Displacements'], axis=1)

        if 'von_mises_stress' in result_mesh_solver.point_data:
            stress_solver = result_mesh_solver.point_data["von_mises_stress"]
            result_mesh_ui.point_data["von_mises_stress"] = unit_manager.convert_from_solver(stress_solver, 'pressure')
        
        try:
            fea_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_fea_result.vtk")
            result_mesh_ui.save(fea_filename)
            iteration_paths['fea_result_path'] = fea_filename
        except Exception as e:
            log_func(f"WARNING: Could not save FEA result for iteration {iteration}. Reason: {e}", "warning")

        max_stress_ui = get_robust_max_stress(result_mesh_ui)
        max_disp_ui = get_robust_max_displacement(result_mesh_ui)
        current_volume = new_surface_mesh.volume
        volume_fraction = current_volume / initial_volume if initial_volume > 1e-9 else 0.0
        
        current_total_strain_energy = result_mesh_solver.field_data.get("total_strain_energy_J", 0.0)
        energy_ui = unit_manager.convert_from_solver(current_total_strain_energy, 'energy')

        log_entry['volume_fraction'] = volume_fraction
        log_entry['max_stress'] = max_stress_ui
        log_entry['max_displacement'] = max_disp_ui
        log_entry['total_strain_energy'] = energy_ui

        max_stress_solver = get_robust_max_stress(result_mesh_solver)
        stress_limit_solver = unit_manager.convert_to_solver(optim_params.get('stress_limit', np.inf), 'pressure')
        if max_stress_solver > stress_limit_solver:
            log_func(f"CONSTRAINT VIOLATED: Stress limit exceeded. Applying high penalty.", "error")
            penalty = 1000.0 * ((max_stress_solver - stress_limit_solver)/ stress_limit_solver if stress_limit_solver > 1e-9 else max_stress_solver)
            log_entry['objective_score'] = penalty
            optimization_log.append(log_entry)
            iteration_data["results"][iteration] = iteration_paths
            return penalty

        selected_objective = optim_params.get('objective', "Minimize Max Stress")

        if selected_objective == "Minimize Max Stress":
            initial_max_stress_solver = unit_manager.convert_to_solver(initial_max_stress_ui, 'pressure')
            fea_score = (max_stress_solver - initial_max_stress_solver) / initial_max_stress_solver if initial_max_stress_solver > 1e-9 else max_stress_solver
        elif selected_objective == "Maximize Strain Energy":
            fea_score = - ((current_total_strain_energy - initial_total_strain_energy)/ initial_total_strain_energy)
        else:
             initial_max_stress_solver = unit_manager.convert_to_solver(initial_max_stress_ui, 'pressure')
             fea_score = (max_stress_solver - initial_max_stress_solver) / initial_max_stress_solver if initial_max_stress_solver > 1e-9 else max_stress_solver

        # NOTE: This mass_reduction_weight is the *target* reduction from optim_params
        mass_reduction_weight = optim_params.get('mass_reduction_ratio', 0.5)

        volume_score = np.abs((current_volume - ((1 - mass_reduction_weight) * initial_volume)) / initial_volume)
        
        # Weighted fitness function
        objective_score = float(10 * fea_score + volume_score)

        current_mass_reduction = (initial_volume - current_volume)/initial_volume
        log_func(f"Mass Reduction: {current_mass_reduction:.2f}%")
        log_func(f"Total Strain Energy: {float(energy_ui):.4f} {unit_manager.get_ui_label('energy')}")
        log_func(f"Scores: Performance={float(fea_score):.4f}, Volume={volume_score:.4f} -> Combined Objective={float(objective_score):.4f}")

        log_entry['objective_score'] = objective_score
        optimization_log.append(log_entry)
        
        if objective_score < iteration_data["best_score"]:
            iteration_data["best_score"] = objective_score
            iteration_data["best_params"] = {
                'max_thick': max_thick_suggested, 
                'min_thick': min_thick_calculated,
                'thickness_ratio': thickness_ratio, # Store the ratio that worked best
                'lattice_type': lattice_type_suggested, 
                'mass_reduction_achieved': current_mass_reduction
            }
            iteration_data["best_mesh"] = result_mesh_ui.copy()
            iteration_data['best_mesh_iteration'] = iteration
            log_func(f"*** New best design found with score: {float(objective_score):.4f} ***")

        iteration_data["last_successful_mesh"] = result_mesh_ui.copy()
        iteration_data["results"][iteration] = iteration_paths
        
        return objective_score

    # --- 4. Run the Optimization ---
    max_iterations = optim_params.get('max_iterations', 25)
    for i in range(max_iterations):
        progress_callback(int((i / max_iterations) * 100), f"Running optimization iteration {i+1}/{max_iterations}")
        suggested_params = optimizer.ask()
        objective_score = objective(suggested_params)

        if objective_score is not None:
            optimizer.tell(suggested_params, objective_score)
            
            if len(optimizer.yi) > 0:
                current_best_score = min(optimizer.yi)
                if (last_best_score - current_best_score) < tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0
                last_best_score = current_best_score
                if patience_counter >= patience:
                    log_func("\n--- Early stopping triggered due to lack of improvement. ---")
                    break

    # --- 5. Finalize and Report ---
    log_func(f"\n--- Lattice Optimization Finished ---\n"
             f"Best result: Score={iteration_data['best_score']:.4f}\n"
             f"Best Parameters Found: {iteration_data['best_params']}\n"
             f"Best lattice at iteration {iteration_data['best_mesh_iteration']}")
    progress_callback(100, "Optimization complete.")

    # Write the optimization log to a text file
    log_filepath = os.path.join(output_dir, "optimization_summary.txt")
    try:
        with open(log_filepath, 'w') as f:
            stress_unit = unit_manager.get_ui_label('pressure')
            disp_unit = unit_manager.get_ui_label('length')
            energy_unit = unit_manager.get_ui_label('energy')
            
            header = f"{'Iteration':<10}{'Lattice_Type':<12}{'Vol_Fraction':<15}{'Max_Stress':<18}{'Max_Disp':<18}{'Total_Energy':<20}{'Objective_Score':<20}\n"
            f.write(header)
            header_units = f"{'':<10}{'':<12}{'':<15}{f'({stress_unit})':<18}{f'({disp_unit})':<18}{f'({energy_unit})':<20}{'':<20}\n"
            f.write(header_units)
            f.write("-" * (len(header) + 10) + "\n")
            
            # --- START: NEW HELPER FUNCTION ---
            def _get_scalar(val_obj):
                """Helper to extract scalar from a potential 1-element array."""
                # Check if it's list-like (and not a string)
                if hasattr(val_obj, '__iter__') and not isinstance(val_obj, str):
                    try:
                        return val_obj[0] # Grab first element
                    except IndexError:
                        return "N/A" # Return "N/A" if it's an empty list
                return val_obj # It's already a scalar or a string like "N/A"
            # --- END: NEW HELPER FUNCTION ---

            for entry in optimization_log:
                
                # --- START: MODIFIED LOGIC ---
                # Extract and format all values
                vf_val = _get_scalar(entry['volume_fraction'])
                ms_val = _get_scalar(entry['max_stress'])
                md_val = _get_scalar(entry['max_displacement'])
                te_val = _get_scalar(entry['total_strain_energy'])
                os_val = _get_scalar(entry['objective_score'])

                # Format them, with a fallback to str() for "N/A"
                vf = f"{vf_val:.4f}" if isinstance(vf_val, (int, float, np.integer, np.floating)) else str(vf_val)
                ms = f"{ms_val:.4f}" if isinstance(ms_val, (int, float, np.integer, np.floating)) else str(ms_val)
                md = f"{md_val:.6f}" if isinstance(md_val, (int, float, np.integer, np.floating)) else str(md_val)
                te = f"{te_val:.4f}" if isinstance(te_val, (int, float, np.integer, np.floating)) else str(te_val)
                os_val = f"{os_val:.4f}" if isinstance(os_val, (int, float, np.integer, np.floating)) else str(os_val)
                
                f.write(f"{entry['iteration']:<10}{str(entry['lattice_type']):<12}{vf:<15}{ms:<18}{md:<18}{te:<20}{os_val:<20}\n")
                # --- END: MODIFIED LOGIC ---

        log_func(f"Optimization summary saved to '{log_filepath}'")
    except Exception as e:
        log_func(f"Could not write optimization summary file. Reason: {e}", "warning")

    if MATPLOTLIB_AVAILABLE:
        try:
            plot_convergence(optimizer.get_result()); plt.savefig("optimization_convergence.png"); plt.close()
            log_func("...convergence plot saved to 'optimization_convergence.png'")
        except Exception as e:
            log_func(f"Warning: Could not generate convergence plot. Reason: {e}", "warning")

    final_control_values = _map_fea_result_to_control_values(
        iteration_data["best_mesh"], control_points_coords, unit_manager, log_func, optim_params.get('scalar_field_source', 'von_mises_stress')
    )
    final_scalar_field_values = _create_normalized_scalar_field(control_points_coords, final_control_values, full_target_points)

    return (
        (full_target_points, final_scalar_field_values),
        iteration_data["best_mesh"],
        iteration_data["results"],
        output_dir
    )
