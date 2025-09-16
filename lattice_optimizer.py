import os
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# --- Mock imports for demonstration purposes ---
# In your actual project, replace these with your real utility modules.
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

# Add optional import for plotting
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
    if "von_mises_stress" in mesh.cell_data and mesh.n_cells > 0 and mesh.cell_data["von_mises_stress"].size > 0:
        return np.max(mesh.cell_data["von_mises_stress"])
    elif "von_mises_stress" in mesh.point_data and mesh.n_points > 0 and mesh.point_data["von_mises_stress"].size > 0:
        return np.max(mesh.point_data['von_mises_stress'])
    return 1e12 # Return a large number if stress data is not found

def _remap_indices_by_proximity(new_mesh: pv.UnstructuredGrid, original_coords: np.ndarray):
    """Finds the new node indices on a new mesh that correspond to original coordinates."""
    if original_coords.size == 0: return []
    if 'persistent_ids' not in new_mesh.point_data:
        new_mesh.point_data['persistent_ids'] = np.arange(new_mesh.n_points)
    
    kdtree = KDTree(new_mesh.points)
    _, indices = kdtree.query(original_coords)
    return list(new_mesh.point_data['persistent_ids'][indices])

def _create_normalized_scalar_field(control_points_coords, control_point_values, full_target_points):
    """Interpolates control point values to a full field and ensures values are [0, 1]."""
    full_scalar_values = griddata(
        control_points_coords, control_point_values, full_target_points,
        method='linear', fill_value=np.mean(control_point_values)
    )
    np.clip(full_scalar_values, 0.0, 1.0, out=full_scalar_values)
    return full_scalar_values

def _map_stress_to_control_values(fea_mesh, control_points_coords, unit_manager, log_func):
    """Takes a mesh with stress data and maps it to normalized [0,1] control point values."""
    if fea_mesh is None:
        log_func("No FEA mesh provided for stress mapping; using uniform 0.5.", "warning")
        return [0.5] * len(control_points_coords)

    stress_data = None
    source_points = None

    if 'von_mises_stress' in fea_mesh.cell_data and fea_mesh.cell_data['von_mises_stress'].size > 0:
        stress_data = fea_mesh.cell_data['von_mises_stress']
        source_points = fea_mesh.cell_centers().points
    elif 'von_mises_stress' in fea_mesh.point_data and fea_mesh.point_data['von_mises_stress'].size > 0:
        stress_data = fea_mesh.point_data['von_mises_stress']
        source_points = fea_mesh.points
    
    if stress_data is None or source_points is None:
        log_func("No valid stress data found; using uniform 0.5.", "warning")
        return [0.5] * len(control_points_coords)

    stresses_solver = unit_manager.convert_to_solver(stress_data, 'pressure')
    
    global_min_s, global_max_s = np.min(stresses_solver), np.max(stresses_solver)
    if (global_max_s - global_min_s) < 1e-9:
        normalized_stresses = np.full_like(stresses_solver, 0.5)
    else:
        normalized_stresses = (stresses_solver - global_min_s) / (global_max_s - global_min_s)

    # Use KDTree to find the average stress from the k-nearest neighbors for each control point
    k_neighbors = min(10, len(source_points))
    if k_neighbors > 0:
        kdtree = KDTree(source_points)
        _, nearest_indices = kdtree.query(control_points_coords, k=k_neighbors)
        if k_neighbors > 1 and nearest_indices.ndim > 1:
            return list(np.mean(normalized_stresses[nearest_indices], axis=1))
        else:
            return list(normalized_stresses[nearest_indices])
            
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
    log_func("--- Starting Advanced Bayesian Lattice Optimization ---")

    output_dir = "optimization_steps"
    os.makedirs(output_dir, exist_ok=True)
    log_func(f"Intermediate results will be saved in '{output_dir}/'")
    
    if 'persistent_ids' not in initial_fea_mesh.point_data:
        initial_fea_mesh.point_data['persistent_ids'] = np.arange(initial_fea_mesh.n_points)

    # --- 1. Initial Setup ---
    initial_ids = initial_fea_mesh.point_data['persistent_ids']
    fixed_mask = np.isin(initial_ids, fea_params.get("fixed_node_indices", []))
    loaded_mask = np.isin(initial_ids, fea_params.get("loaded_node_indices", []))
    original_fixed_coords = initial_fea_mesh.points[fixed_mask]
    original_loaded_coords = initial_fea_mesh.points[loaded_mask]
    
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
    try:
        initial_volume = original_shell.volume
        if initial_volume < 1e-9: initial_volume = 1.0
    except Exception:
        initial_volume = 1.0

    # --- 2. Define Optimization Search Space ---
    user_min_limit = optim_params.get('min_thickness', 0.5)
    user_max_limit = optim_params.get('max_thickness', 5.0)
    available_lattice_types = optim_params.get('available_lattice_types', ['gyroid', 'diamond', 'lidinoid'])
    
    search_space = [
        Real(user_min_limit, user_max_limit, name='max_thick'),
        Categorical(available_lattice_types, name='lattice_type')
    ]
    
    seed = optim_params.get('random_state', 42)
    optimizer = Optimizer(
        dimensions=search_space, 
        base_estimator="GP", 
        acq_func="EI",
        random_state=seed
    )

    # --- 3. Define the Objective Function to Minimize ---
    iteration_data = {
        "counter": 0, "best_score": np.inf, "best_params": {},
        "best_mesh": initial_fea_mesh.copy(), "last_successful_mesh": initial_fea_mesh.copy(),
        "results": {} # This will store the file paths for each iteration
    }
    
    def objective(params):
        iteration = iteration_data["counter"]
        iteration_data["counter"] += 1
        
        iteration_paths = {
            'scalar_path': '',
            'lattice_path': '',
            'fea_result_path': ''
        }
        
        # Unpack parameters and calculate thickness
        max_thick_suggested, lattice_type_suggested = params
        mass_reduction_ratio = optim_params.get('mass_reduction_ratio', 0.5)
        min_thick_bound = optim_params.get('min_thickness', 0.5)
        min_thick_calculated = max_thick_suggested - mass_reduction_ratio * (max_thick_suggested - min_thick_bound)
        min_thick_calculated = max(min_thick_calculated, min_thick_bound)

        log_func(f"\n--- Iteration {iteration+1} ---")
        log_func(f"Suggested: max_thick={max_thick_suggested:.3f}, type={lattice_type_suggested}")
        log_func(f"Calculated min_thick (ratio={mass_reduction_ratio}): {min_thick_calculated:.3f}")

        # A. Generate and Save Scalar Field
        control_values = _map_stress_to_control_values(iteration_data["last_successful_mesh"], control_points_coords, unit_manager, log_func)
        full_scalar_values = _create_normalized_scalar_field(control_points_coords, control_values, full_target_points)
        
        try:
            scalar_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_scalar.txt")
            scalar_data_to_save = np.hstack([full_target_points, full_scalar_values[:, np.newaxis]])
            np.savetxt(scalar_filename, scalar_data_to_save, header='X Y Z NormalizedValue', fmt='%.6f')
            iteration_paths['scalar_path'] = scalar_filename
        except Exception as e:
            log_func(f"WARNING: Could not save scalar field for iteration {iteration}. Reason: {e}", "warning")

        # B. Generate and Save New Lattice Surface
        current_lattice_params = {**lattice_params,
            'external_scalar': (full_target_points, full_scalar_values), 'use_scalar_for_thickness': True,
            'solidify': True, 'min_thickness_bound': min_thick_calculated,
            # DEBUG FIX 1: Corrected typo from 'resoluiton' to 'resolution'
            'resolution': 100, 
            'max_thickness_bound': max_thick_suggested, 'lattice_type': lattice_type_suggested
        }

        remesh_params = {
            "remesh_enabled": True,
            "smoothing": "Taubin",
            "smoothing_iterations": 500,
            "repair_methods": {
                "Simplification": {"reduction": 0.2},
                "Adaptive":{}
            }
        }
        new_surface_mesh = generate_infill_inside(mesh=original_shell, log_func=log_func, **current_lattice_params, **remesh_params)

        if not new_surface_mesh or new_surface_mesh.n_points == 0:
            log_func("Lattice generation failed. Applying penalty.", "error"); return 1e12

        try:
            lattice_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_lattice.stl")
            new_surface_mesh.save(lattice_filename)
            iteration_paths['lattice_path'] = lattice_filename
        except Exception as e:
            log_func(f"WARNING: Could not save lattice mesh for iteration {iteration}. Reason: {e}", "warning")

        # C. Create Volumetric Mesh and Run FEA
        meshing_params = {
            'detail_size': 1,
            'feature_angle': 30.0,
            'volume_g_size': 2,
            'mesh_order': 1,
            'optimize_ho': False,
            'algorithm': 'HEX',
            'skip_preprocessing': False,
            'lattice_model': True
        }
        success, new_vol_mesh = create_robust_volumetric_mesh(surface_mesh=new_surface_mesh, **meshing_params, log_func=log_func)
        if not success:
            log_func("Volumetric meshing failed. Applying penalty.", "error"); return 1e12

        new_fixed_ids = _remap_indices_by_proximity(new_vol_mesh, original_fixed_coords)
        new_loaded_ids = _remap_indices_by_proximity(new_vol_mesh, original_loaded_coords)
        if not new_fixed_ids or not new_loaded_ids:
            log_func("Failed to remap boundary conditions. Applying penalty.", "error"); return 1e12
        
        solver_mesh_input = new_vol_mesh.copy()
        solver_mesh_input.points = unit_manager.convert_to_solver(solver_mesh_input.points, 'length')
        force_vector_solver = tuple(unit_manager.convert_to_solver(f, 'force') for f in fea_params["force"])
        current_fea_params = {**fea_params,
            'mesh': solver_mesh_input, 'fixed_node_indices': new_fixed_ids, 
            'loaded_node_indices': new_loaded_ids, 'force': force_vector_solver
        }
        result_mesh_solver = run_native_fea(**current_fea_params)

        # DEBUG FIX 2: Added a check to handle FEA solver failure.
        if result_mesh_solver is None:
            log_func("FEA solver failed to produce a result. Applying penalty.", "error")
            return 1e12
        
        # D. Convert FEA results to UI units and Save
        result_mesh_ui = result_mesh_solver.copy()
        result_mesh_ui.points = unit_manager.convert_from_solver(result_mesh_solver.points, 'length')
        if 'Displacements' in result_mesh_solver.point_data:
            disp_solver = result_mesh_solver.point_data['Displacements']
            result_mesh_ui.point_data['Displacements'] = unit_manager.convert_from_solver(disp_solver, 'length')
            result_mesh_ui.point_data['displacement'] = np.linalg.norm(result_mesh_ui.point_data['Displacements'], axis=1)
        for field in ["von_mises_stress", "principal_s1", "principal_s2", "principal_s3"]:
            if field in result_mesh_solver.cell_data:
                stress_solver = result_mesh_solver.cell_data[field]
                result_mesh_ui.cell_data[field] = unit_manager.convert_from_solver(stress_solver, 'pressure')
        
        try:
            fea_filename = os.path.join(output_dir, f"iteration_{iteration:03d}_fea_result.vtk")
            result_mesh_ui.save(fea_filename)
            iteration_paths['fea_result_path'] = fea_filename
        except Exception as e:
            log_func(f"WARNING: Could not save FEA result for iteration {iteration}. Reason: {e}", "warning")

        # E. Calculate Objective Score
        max_stress_solver = get_robust_max_stress(result_mesh_solver)
        stress_limit_solver = unit_manager.convert_to_solver(optim_params.get('stress_limit', np.inf), 'pressure')
        if max_stress_solver > stress_limit_solver:
            log_func("CONSTRAINT VIOLATED: Stress limit exceeded. Applying high penalty.", "error"); return 1e12

        initial_max_stress_solver = unit_manager.convert_to_solver(initial_max_stress_ui, 'pressure')
        performance_score = max_stress_solver / initial_max_stress_solver if initial_max_stress_solver > 1e-9 else max_stress_solver
        
        current_volume = new_surface_mesh.volume
        volume_score = current_volume / initial_volume
        
        mass_reduction_weight = optim_params.get('mass_reduction_weight', 0.5)
        objective_score = performance_score + (mass_reduction_weight * volume_score)

        current_mass_reduction = (1.0 - volume_score) * 100.0
        log_func(f"Mass Reduction: {current_mass_reduction:.2f}%")
        log_func(f"Scores: Performance={performance_score:.4f}, Volume={volume_score:.4f} -> Combined Objective={objective_score:.4f}")
        
        # Store results if this is the best iteration so far
        if objective_score < iteration_data["best_score"]:
            iteration_data["best_score"] = objective_score
            iteration_data["best_params"] = {
                'max_thick': max_thick_suggested, 'min_thick': min_thick_calculated,
                'lattice_type': lattice_type_suggested, 'mass_reduction_achieved': current_mass_reduction
            }
            iteration_data["best_mesh"] = result_mesh_ui.copy()
            log_func(f"*** New best design found with score: {objective_score:.4f} ***")

        iteration_data["last_successful_mesh"] = result_mesh_ui.copy()
        # Store the paths for this iteration so the UI can load them
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

    # --- 5. Finalize and Report ---
    log_func(f"\n--- Lattice Optimization Finished ---\n"
             f"Best result: Score={iteration_data['best_score']:.4f}\n"
             f"Best Parameters Found: {iteration_data['best_params']}")
    progress_callback(100, "Optimization complete.")

    if MATPLOTLIB_AVAILABLE:
        try:
            plot_convergence(optimizer.get_result()); plt.savefig("optimization_convergence.png"); plt.close()
            log_func("...convergence plot saved to 'optimization_convergence.png'")
        except Exception as e:
            log_func(f"Warning: Could not generate convergence plot. Reason: {e}", "warning")

    final_control_values = _map_stress_to_control_values(iteration_data["best_mesh"], control_points_coords, unit_manager, log_func)
    best_scalar_field = _create_normalized_scalar_field(control_points_coords, final_control_values, full_target_points)

    return (
        (full_target_points, best_scalar_field),
        iteration_data["best_mesh"],
        iteration_data["results"], # Return the dictionary of file paths
    )
