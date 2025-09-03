import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from lattice_utils import generate_infill_inside
from fea_utils import create_robust_volumetric_mesh
from fea_solver_core import run_native_fea
from scipy.spatial import KDTree
import os
from scipy import ndimage
from unit_utils import UnitManager

# Bayesian Optimization requires the scikit-optimize library
try:
    from skopt import Optimizer
    from skopt.space import Real
    from skopt.plots import plot_convergence
except ImportError:
    raise ImportError("Bayesian optimization requires 'scikit-optimize'. Please install it using: pip install scikit-optimize")

# Add optional import for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def get_robust_max_displacement(mesh):
    """Calculates a robust maximum displacement value from a mesh."""
    if "displacement" in mesh.point_data and mesh.n_points > 0:
        displacements = mesh.point_data["displacement"]
        if displacements.size == 0: return 0.0
        return np.percentile(displacements, 99.5)
    return 1e12

def get_robust_max_stress(mesh):
    """Calculates a robust maximum stress value from a mesh."""
    if "von_mises_stress" in mesh.cell_data and mesh.n_cells > 0 and mesh.cell_data["von_mises_stress"].size > 0:
        return np.max(mesh.cell_data["von_mises_stress"])
    elif "von_mises_stress" in mesh.point_data and mesh.n_points > 0 and mesh.point_data["von_mises_stress"].size > 0:
        return np.max(mesh.point_data['von_mises_stress'])
    return 1e12


def _remap_indices_by_proximity(new_mesh: pv.UnstructuredGrid, original_coords: np.ndarray, search_radius: float):
    if original_coords.size == 0: return []
    if 'persistent_ids' not in new_mesh.point_data:
        new_mesh.point_data['persistent_ids'] = np.arange(new_mesh.n_points)
    kdtree = KDTree(new_mesh.points)
    _, indices = kdtree.query(original_coords)
    return list(new_mesh.point_data['persistent_ids'][indices])

def _create_normalized_scalar_field(control_points_coords, control_point_values, full_target_points):
    """
    Takes optimizer's suggested [0,1] control values and returns a clean,
    full-sized, and validated normalized scalar field.
    """
    full_scalar_values = griddata(
        control_points_coords, control_point_values, full_target_points,
        method='linear', fill_value=np.mean(control_point_values)
    )
    np.clip(full_scalar_values, 0.0, 1.0, out=full_scalar_values)
    return full_scalar_values

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
    log_func("--- Starting Advanced Bayesian Lattice Optimization ---")

    output_dir = "optimization_steps"
    os.makedirs(output_dir, exist_ok=True)
    
    if 'persistent_ids' not in initial_fea_mesh.point_data:
        raise ValueError("Initial mesh is missing 'persistent_ids' required for remapping.")

    search_radius = meshing_params.get('detail_size', 1.0) * 2.0
    initial_ids = initial_fea_mesh.point_data['persistent_ids']
    fixed_mask = np.isin(initial_ids, fea_params.get("fixed_node_indices", []))
    loaded_mask = np.isin(initial_ids, fea_params.get("loaded_node_indices", []))
    original_fixed_coords = initial_fea_mesh.points[fixed_mask]
    original_loaded_coords = initial_fea_mesh.points[loaded_mask]
    
    num_nodes = initial_fea_mesh.n_points
    target_num_control_points = max(100, int(num_nodes * 0.0001))
    bounds = initial_fea_mesh.bounds
    model_dims = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    model_volume = np.prod(model_dims)
    if model_volume < 1e-9: raise ValueError("Model has zero volume.")
    target_voxel_volume = model_volume / target_num_control_points
    unit_side_length = target_voxel_volume**(1/3.0)
    nx, ny, nz = (max(1, int(np.ceil(d / unit_side_length))) for d in model_dims)
    x_coords, y_coords, z_coords = (np.linspace(b[0], b[1], n) for b, n in zip([bounds[:2], bounds[2:4], bounds[4:]], [nx, ny, nz]))
    grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    _, nearest_node_indices = KDTree(initial_fea_mesh.points).query(grid_points)
    control_point_indices = np.unique(nearest_node_indices)
    control_points_coords = initial_fea_mesh.points[control_point_indices]
    num_control_points = len(control_points_coords)
    log_func(f"...selected {num_control_points} uniformly distributed control points.")
    
    full_target_points = initial_fea_mesh.points
    min_thick = optim_params.get('min_thickness', 0.5)
    max_thick = optim_params.get('max_thickness', 2.0)

    log_func("Optimizer will work with normalized [0, 1] scalar values.")
    search_space = [Real(0.0, 1.0, name=f'v_{i}') for i in range(num_control_points)]
    optimizer = Optimizer(dimensions=search_space, base_estimator="GP", acq_func="EI")

    log_func("...mapping elemental stresses to control points for warm start.")
    
    stress_values_ui = initial_fea_mesh.cell_data.get('von_mises_stress')
    element_stresses = None
    normalized_element_stresses = None
    if stress_values_ui is not None:
        element_stresses = unit_manager.convert_to_solver(stress_values_ui, 'pressure')

    if element_stresses is None or element_stresses.size == 0:
        log_func("No stress data for warm start. Using a uniform medium value (0.5).", "warning")
        initial_control_values = [0.5] * num_control_points
    else:
        global_min_s, global_max_s = np.min(element_stresses), np.max(element_stresses)
        if (global_max_s - global_min_s) < 1e-9:
            normalized_element_stresses = np.full_like(element_stresses, 0.5)
        else:
            normalized_element_stresses = (element_stresses - global_min_s) / (global_max_s - global_min_s)
        
        # --- NEWLY ADDED BLOCK: SAVE INITIAL STRESS FIELD ---
        try:
            log_func("...saving the initial normalized stress field as a baseline.")
            temp_mesh = initial_fea_mesh.copy()
            temp_mesh.cell_data['normalized_stress'] = normalized_element_stresses
            temp_mesh = temp_mesh.cell_data_to_point_data()
            
            points = temp_mesh.points
            values = temp_mesh.point_data['normalized_stress']
            scalar_data_to_save = np.hstack([points, values[:, np.newaxis]])
            
            filename = os.path.join(output_dir, "initial_stress_scalar.txt")
            np.savetxt(filename, scalar_data_to_save, header='X Y Z NormalizedStressValue', fmt='%.6f')
            log_func(f"--- Baseline stress field saved to {filename} ---")
        except Exception as e:
            log_func(f"WARNING: Could not save the initial baseline stress field. Reason: {str(e)}", "warning")
        # --- END OF NEWLY ADDED BLOCK ---

        k_neighbors = min(10, initial_fea_mesh.n_cells)
        if k_neighbors > 0:
            element_centers = initial_fea_mesh.cell_centers().points
            _, nearest_element_indices = KDTree(element_centers).query(control_points_coords, k=k_neighbors)
            if k_neighbors > 1 and nearest_element_indices.ndim > 1:
                initial_control_values = list(np.mean(normalized_element_stresses[nearest_element_indices], axis=1))
            else:
                initial_control_values = list(normalized_element_stresses[nearest_element_indices])
        else:
             initial_control_values = [0.5] * num_control_points

    initial_max_stress_ui = get_robust_max_stress(initial_fea_mesh)
    initial_max_disp_ui = get_robust_max_displacement(initial_fea_mesh)
    try:
        initial_volume = initial_fea_mesh.extract_surface().volume
    except Exception:
        initial_volume = 1.0

    iteration_counter = [0]
    iteration_mesh_paths = []

    def objective(control_point_values):
        current_iteration = iteration_counter[0]
        iteration_counter[0] += 1
        
        mass_reduction_priority = optim_params.get('mass_reduction_priority', 0.5)
        selected_objective = optim_params.get('objective', "Minimize Max Stress")
        
        stress_limit_solver = unit_manager.convert_to_solver(optim_params.get('stress_limit', np.inf), 'pressure')
        disp_limit_solver = unit_manager.convert_to_solver(optim_params.get('disp_limit', np.inf), 'length')
        initial_max_stress_solver = unit_manager.convert_to_solver(initial_max_stress_ui, 'pressure')
        initial_max_disp_solver = unit_manager.convert_to_solver(initial_max_disp_ui, 'length')
        
        log_func("Step 1/4: Creating and validating normalized scalar field...")
        full_scalar_values = _create_normalized_scalar_field(
            control_points_coords, control_point_values, full_target_points
        )
        current_external_scalar = (full_target_points, full_scalar_values)

        try:
            scalar_filename = os.path.join(output_dir, f"iteration_{current_iteration:03d}_scalar.txt")
            points, values = current_external_scalar
            scalar_data_to_save = np.hstack([points, values[:, np.newaxis]])
            np.savetxt(scalar_filename, scalar_data_to_save, header='X Y Z NormalizedValue', fmt='%.6f')
        except Exception as e:
            log_func(f"WARNING: Could not save intermediate scalar field. Reason: {str(e)}")
        
        current_lattice_params = lattice_params.copy()
        current_lattice_params.update({
            'external_scalar': current_external_scalar, 'use_scalar_for_thickness': True,
            'solidify': True,
            'min_thickness_bound': min_thick, 'max_thickness_bound': max_thick
        })
        new_surface_mesh = generate_infill_inside(
            mesh=original_shell, log_func=log_func, **current_lattice_params, **remesh_params
        )

        if not new_surface_mesh or new_surface_mesh.n_points == 0:
            log_func("Lattice generation failed or produced an empty mesh.", "error")
            return None, None, None, 1e12
        
        try:
            filename = os.path.join(output_dir, f"iteration_{current_iteration:03d}.stl")
            new_surface_mesh.save(filename)
            iteration_mesh_paths.append(filename)
        except Exception as e:
            log_func(f"WARNING: Could not save intermediate mesh. Reason: {str(e)}")

        try:
            current_volume = new_surface_mesh.volume
        except Exception:
            current_volume = initial_volume

        success, new_vol_mesh = create_robust_volumetric_mesh(surface_mesh=new_surface_mesh, **meshing_params, log_func=log_func)
        if not success:
            log_func(f"Volumetric meshing failed for iteration {current_iteration}.", "error")
            return None, None, None, 1e12

        new_fixed_ids = _remap_indices_by_proximity(new_vol_mesh, original_fixed_coords, search_radius)
        new_loaded_ids = _remap_indices_by_proximity(new_vol_mesh, original_loaded_coords, search_radius)
        if not new_fixed_ids or not new_loaded_ids:
            log_func("Failed to remap boundary conditions.", "error")
            return None, None, None, 1e12

        force_vector_solver = (
            unit_manager.convert_to_solver(fea_params["force"][0], 'force'),
            unit_manager.convert_to_solver(fea_params["force"][1], 'force'),
            unit_manager.convert_to_solver(fea_params["force"][2], 'force')
        )
        
        current_fea_params = fea_params.copy()
        current_fea_params.update({
            'mesh': new_vol_mesh,
            'fixed_node_indices': new_fixed_ids, 
            'loaded_node_indices': new_loaded_ids,
            'force': force_vector_solver
        })
        
        result_mesh_solver = run_native_fea(**current_fea_params)
        max_stress_solver = get_robust_max_stress(result_mesh_solver)
        max_displacement_solver = get_robust_max_displacement(result_mesh_solver)
        
        max_stress_ui = unit_manager.convert_from_solver(max_stress_solver, 'pressure')

        if max_stress_solver > stress_limit_solver or max_displacement_solver > disp_limit_solver:
            log_func(f"CONSTRAINT VIOLATED: Applying high penalty.", "error")
            log_func(f"Stress Limit Violated: {max_stress_ui:.2f} > {optim_params.get('stress_limit'):.2f} {unit_manager.get_ui_label('pressure')}")
            return new_vol_mesh, max_stress_ui, current_external_scalar, 1e12

        if selected_objective == "Minimize Max Stress":
            performance_score = max_stress_solver / initial_max_stress_solver if initial_max_stress_solver > 1e-9 else max_stress_solver
        else:
            performance_score = max_displacement_solver / initial_max_disp_solver if initial_max_disp_solver > 1e-9 else max_displacement_solver
            
        volume_score = current_volume / initial_volume if initial_volume > 1e-9 else 1.0
        objective_score = performance_score * (1 + mass_reduction_priority * volume_score)
        
        mass_reduction_percent = (1.0 - volume_score) * 100
        max_disp_ui = unit_manager.convert_from_solver(max_displacement_solver, 'length')
        log_func(f"Result: Stress={max_stress_ui:,.2f} {unit_manager.get_ui_label('pressure')}, Disp={max_disp_ui:.4f} {unit_manager.get_ui_label('length')}, Mass Reduction={mass_reduction_percent:.2f}%")
        log_func(f"Blended Objective Score: {objective_score:.4f}")

        result_mesh_ui = new_vol_mesh.copy()
        result_mesh_ui.point_data['Displacements'] = unit_manager.convert_from_solver(result_mesh_solver.point_data['Displacements'], 'length')
        result_mesh_ui.point_data['displacement'] = np.linalg.norm(result_mesh_ui.point_data['Displacements'], axis=1)
        for field in ["von_mises_stress", "principal_s1", "principal_s2", "principal_s3"]:
            if field in result_mesh_solver.cell_data:
                result_mesh_ui.cell_data[field] = unit_manager.convert_from_solver(result_mesh_solver.cell_data[field], 'pressure')

        return result_mesh_ui, max_stress_ui, current_external_scalar, objective_score

    log_func("\n--- Evaluating Initial Warm-Start Configuration ---")

    initial_mesh, initial_stress, initial_scalar_field, initial_objective_score = objective(initial_control_values)

    if initial_objective_score is None or initial_objective_score > 1e11:
        log_func("Initial warm-start evaluation failed. Proceeding with default values.", "error")
        best_mesh = initial_fea_mesh
        best_stress = initial_max_stress_ui
        baseline_scalar_field_values = _create_normalized_scalar_field(
            control_points_coords, initial_control_values, full_target_points
        )
        best_scalar_field = (full_target_points, baseline_scalar_field_values)
        best_objective_score = np.inf
        optimizer.tell([initial_control_values], [1e12])
    else:
        log_func(f"Initial configuration score: {initial_objective_score:.4f}")
        best_mesh = initial_mesh
        best_stress = initial_stress
        best_scalar_field = initial_scalar_field
        best_objective_score = initial_objective_score
        optimizer.tell([initial_control_values], [initial_objective_score])

    max_iterations = optim_params.get('max_iterations', 10)
    for i in range(max_iterations):
        progress_callback(int((i / max_iterations) * 100), f"Running optimization iteration {i+1}/{max_iterations}")
        log_func(f"\n--- Bayesian Optimization Iteration {i+1}/{max_iterations} ---")
        
        suggested_params = optimizer.ask()
        result_mesh, result_stress, current_scalar, objective_score = objective(suggested_params)

        if objective_score is not None:
            optimizer.tell(suggested_params, objective_score)

        if objective_score is not None and objective_score < best_objective_score and result_mesh is not None:
            best_objective_score = objective_score
            best_stress = result_stress
            best_mesh = result_mesh
            best_scalar_field = current_scalar
            log_func(f"*** New best design found with score: {best_objective_score:.4f} ***")

    log_func(f"\n--- Lattice Optimization Finished ---")
    log_func(f"Best result: Score={best_objective_score:.4f}, Stress={best_stress:,.2f} {unit_manager.get_ui_label('pressure')}")
    progress_callback(100, "Optimization complete.")

    if MATPLOTLIB_AVAILABLE:
        try:
            plot_convergence(optimizer.get_result())
            plt.savefig("optimization_convergence.png")
            plt.close()
            log_func("...convergence plot saved to 'optimization_convergence.png'")
        except Exception as e:
            log_func(f"WARNING: Could not generate diagnostic plots. Reason: {str(e)}")

    return best_scalar_field, best_mesh, iteration_mesh_paths
