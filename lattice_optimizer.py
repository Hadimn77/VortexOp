import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from lattice_utils import generate_infill_inside
from fea_utils import create_robust_volumetric_mesh
from fea_solver_core import run_native_fea

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


def run_optimization_loop(
    initial_fea_mesh: pv.UnstructuredGrid,
    original_shell: pv.PolyData,
    lattice_params: dict,
    remesh_params: dict,
    meshing_params: dict,
    fea_params: dict,
    optim_params: dict,
    log_func,
    progress_callback
):
    """
    Runs a Bayesian optimization loop using a high-dimensional design space.
    - Selects 10% of mesh nodes as control points.
    - Optimizes the scalar value at each control point.
    - Interpolates a full scalar field for each iteration.
    - Minimizes the average of the top 1% of elemental von Mises stresses.
    """
    log_func("--- Starting Advanced Bayesian Lattice Optimization ---")

    # --- 1. Define the High-Dimensional Design Space ---
    log_func("Defining a high-dimensional design space...")
    num_nodes = initial_fea_mesh.n_points
    # Use 10% of nodes as control points, with a minimum of 10
    num_control_points = max(10, int(num_nodes * 0.10))
    # Randomly select the indices for our control points
    control_point_indices = np.random.choice(num_nodes, size=num_control_points, replace=False)
    # Get the 3D coordinates of these control points
    control_points_coords = initial_fea_mesh.points[control_point_indices]
    # We will interpolate from the control points back to all original points
    full_target_points = initial_fea_mesh.points
    log_func(f"...selected {num_control_points} control points ({num_control_points} optimization dimensions).")

    # The search space is a scalar value for each control point.
    # A range of [0, 2] is a good starting point for normalized scalar values.
    search_space = [Real(0.0, 2.0, name=f'v_{i}') for i in range(num_control_points)]
    optimizer = Optimizer(dimensions=search_space, base_estimator="GP", acq_func="EI")

    # --- 2. Initial Analysis & Optimizer Warm Start ---
    if 'von_mises_stress' not in initial_fea_mesh.point_data:
        raise ValueError("The input mesh is missing 'von_mises_stress' results.")

    def get_robust_max_stress(mesh):
        if "element_von_mises" in mesh.cell_data and mesh.n_cells > 0:
            element_stresses = mesh.cell_data["element_von_mises"]
            num_elements = len(element_stresses)
            top_percent_count = max(1, int(np.ceil(num_elements * 0.01)))
            top_stresses = np.sort(element_stresses)[-top_percent_count:]
            return np.mean(top_stresses)
        else:
            log_func("Warning: Elemental stress data not found. Falling back to max nodal stress.")
            return np.max(mesh.point_data['von_mises_stress'])
            
    # Provide a "warm start" to the optimizer using the initial stress field
    initial_stresses = initial_fea_mesh.point_data['von_mises_stress']
    initial_norm_stress = (initial_stresses - np.min(initial_stresses)) / (np.max(initial_stresses) - np.min(initial_stresses))
    initial_control_values = list(initial_norm_stress[control_point_indices])
    initial_max_stress = get_robust_max_stress(initial_fea_mesh)
    
    log_func("Providing optimizer with a warm start based on initial stress field.")
    optimizer.tell([initial_control_values], [initial_max_stress])
    
    # --- 3. Define the Objective Function ---
    def objective(control_point_values):
        # The optimizer provides a list of values, one for each control point.
        log_func("Step 1/4: Interpolating full scalar field from control points...")
        
        # Use griddata to interpolate from our sparse control points to the dense full mesh
        full_scalar_values = griddata(
            control_points_coords, 
            control_point_values, 
            full_target_points, 
            method='linear', 
            fill_value=np.mean(control_point_values) # Use mean for points outside convex hull
        )
        # Ensure scalar values are non-negative
        full_scalar_values[full_scalar_values < 0] = 0
        current_external_scalar = (full_target_points, full_scalar_values)
        
        log_func("Step 2/4: Regenerating lattice with new thickness profile...")
        current_lattice_params = lattice_params.copy()
        current_lattice_params.update({
            'external_scalar': current_external_scalar,
            'use_scalar_for_thickness': True,
        })
        
        new_surface_mesh = generate_infill_inside(mesh=original_shell, **current_lattice_params, **remesh_params)
        if not new_surface_mesh or new_surface_mesh.n_points == 0:
            log_func("WARNING: Lattice generation failed. Returning high stress.")
            return None, 1e12, None

        log_func("Step 3/4: Creating new volumetric mesh...")
        success, new_vol_mesh = create_robust_volumetric_mesh(surface_mesh=new_surface_mesh, **meshing_params, log_func=log_func)
        if not success:
            log_func(f"WARNING: Volumetric meshing failed. Returning high stress. Reason: {new_vol_mesh}")
            return None, 1e12, None

        log_func("Step 4/4: Running FEA on the new design...")
        current_fea_params = fea_params.copy()
        current_fea_params['mesh'] = new_vol_mesh
        result_mesh = run_native_fea(**current_fea_params)

        max_stress = get_robust_max_stress(result_mesh)
        log_func(f"Result: Top 1% Avg von Mises Stress = {max_stress:,.2f} Pa")
        
        return result_mesh, max_stress, current_external_scalar

    # --- 4. Run the Optimization Loop ---
    max_iterations = optim_params.get('max_iterations', 20)
    convergence_tolerance = optim_params.get('convergence_tolerance', 0.01)
    
    best_mesh = initial_fea_mesh
    best_stress = initial_max_stress
    best_scalar_field = (full_target_points, initial_norm_stress)
    
    for i in range(max_iterations):
        progress_callback((i / max_iterations) * 100, f"Running optimization iteration {i+1}/{max_iterations}...")
        log_func(f"\n--- Bayesian Optimization Iteration {i+1}/{max_iterations} ---")

        suggested_params = optimizer.ask()
        
        result_mesh, result_stress, current_scalar = objective(suggested_params)
        
        optimizer.tell(suggested_params, result_stress)

        if result_stress < best_stress and result_mesh is not None:
            best_stress = result_stress
            best_mesh = result_mesh
            best_scalar_field = current_scalar
            log_func(f"*** New best design found with stress: {best_stress:,.2f} Pa ***")

    # --- 5. Finalize and Return the Best Result ---
    log_func("\n--- Lattice Optimization Finished ---")
    log_func(f"Best result achieved (Top 1% Avg Stress): {best_stress:,.2f} Pa")
    
    progress_callback(100, "Optimization complete.")

    # --- 6. Generate and save diagnostic plots ---
    if MATPLOTLIB_AVAILABLE:
        log_func("Generating optimization diagnostic plots...")
        try:
            result = optimizer.get_result()

            # Generate and save convergence plot
            plot_convergence(result)
            plt.savefig("optimization_convergence.png")
            plt.close()
            log_func("...convergence plot saved to 'optimization_convergence.png'")
            
            # NOTE: plot_objective is not suitable for high-dimensional spaces, so it is omitted.

        except Exception as e:
            log_func(f"WARNING: Could not generate diagnostic plots. Reason: {e}")
    else:
        log_func("Skipping plot generation: 'matplotlib' is not installed.")

    return best_scalar_field, best_mesh