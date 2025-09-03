# lattice_utils.py (v5.03 – Corrected)
import numpy as np
import pyvista as pv
import trimesh
from packaging.version import parse
from scipy.interpolate import griddata

MIN_PYVISTA_VERSION = "0.38.0"
assert parse(pv.__version__) >= parse(MIN_PYVISTA_VERSION), \
    f"CRITICAL ERROR: Your Python environment is using an old version of PyVista " \
    f"(found {pv.__version__}, but require >= {MIN_PYVISTA_VERSION}).\n" \
    f"Please configure your IDE to use the Python interpreter from your 'venv' folder."

# ----------------------------- Lattice Implicit Functions -----------------------------
def gyroid(x, y, z, wx, wy, wz):
    return (
        np.sin(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) +
        np.sin(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz) +
        np.sin(2 * np.pi * z / wz) * np.cos(2 * np.pi * x / wx)
    )

def diamond(x, y, z, wx, wy, wz):
    return (
        np.cos(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz) -
        np.sin(2 * np.pi * x / wx) * np.sin(2 * np.pi * y / wy) * np.sin(2 * np.pi * z / wz)
    )

def neovius(x, y, z, wx, wy, wz):
    return (
        3 * (np.cos(2 * np.pi * x / wx) +
             np.cos(2 * np.pi * y / wy) +
             np.cos(2 * np.pi * z / wz)) +
        4 * np.cos(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz)
    )

def generate_scalar_field(x, y, z, lattice_type='gyroid', wx=10, wy=10, wz=10):
    if lattice_type == 'gyroid':
        return gyroid(x, y, z, wx, wy, wz)
    elif lattice_type == 'diamond':
        return diamond(x, y, z, wx, wy, wz)
    elif lattice_type == 'neovius':
        return neovius(x, y, z, wx, wy, wz)
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")

# ----------------------------- Utilities -----------------------------
def convert_to_triangulated_polydata(grid: pv.UnstructuredGrid) -> pv.PolyData:
    return grid.extract_surface().triangulate()

def apply_remeshing(mesh: pv.PolyData, remesh_params: dict, log_func=print):
    if not remesh_params.get('remesh_enabled', False):
        return mesh

    log = log_func or print
    smoothing = remesh_params.get('smoothing', 'None')
    smoothing_iterations = remesh_params.get('smoothing_iterations', 100)
    repair_methods = remesh_params.get('repair_methods', {})

    log("Applying post-processing to lattice...")

    if "Fill Holes" in repair_methods:
        log("...filling holes.")
        mesh = mesh.fill_holes(repair_methods["Fill Holes"]['hole_size'])

    if "Simplification" in repair_methods:
        log("...simplifying mesh.")
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        mesh = mesh.decimate_pro(repair_methods["Simplification"]['reduction'], preserve_topology=True)

    if "Adaptive (Curvature)" in repair_methods:
        log("...applying adaptive remeshing based on curvature.")
        reduction = repair_methods.get("Adaptive (Curvature)", {}).get('reduction', 0.1)
        feature_angle = repair_methods.get("Adaptive (Curvature)", {}).get('feature_angle', 100.0)
        mesh = mesh.decimate_pro(reduction, preserve_topology=True, feature_angle=feature_angle)

    if smoothing == "Laplacian":
        log(f"...applying Laplacian smoothing ({smoothing_iterations} iterations).")
        mesh = mesh.smooth(n_iter=smoothing_iterations, pass_band=0.05)
    elif smoothing == "Taubin":
        log(f"...applying Taubin smoothing ({smoothing_iterations} iterations).")
        mesh = mesh.smooth_taubin(n_iter=smoothing_iterations, pass_band=0.05)

    log("Post-processing completed.")
    return mesh.triangulate().clean()

# ----------------------------- Robust SDF Sign Helper -----------------------------
def _sdf_negative_inside(
    grid: pv.StructuredGrid,
    mesh: pv.PolyData,
    sdf_1d: np.ndarray,
    log_func=None,
    hole_factor: float = 0.02,
    enclosed_tol: float = 0.0
) -> np.ndarray:
    """
    Force SDF to be NEGATIVE inside the mesh.
    """
    log = log_func or (lambda *a, **k: None)
    dims = tuple(int(x) for x in grid.dimensions)
    phi = np.asarray(sdf_1d, dtype=np.float32).reshape(dims, order='F')

    def _select(m: pv.PolyData, check_surface: bool, tol: float):
        return grid.select_enclosed_points(
            m, check_surface=check_surface, tolerance=tol, inside_out=False
        )

    try:
        enclosed = _select(mesh, check_surface=True, tol=enclosed_tol)
    except Exception:
        log("...surface not closed; attempting auto-repair (fill small holes).")
        try:
            hole_size = max(mesh.length * hole_factor, 1e-9)
            repaired = mesh.triangulate().clean().fill_holes(hole_size).triangulate().clean()
            enclosed = _select(repaired, check_surface=True, tol=enclosed_tol)
        except Exception:
            log("...falling back to tolerant inside test (approximate).")
            enclosed = _select(mesh, check_surface=False, tol=max(enclosed_tol, 1e-9))

    inside_mask = enclosed.point_data['SelectedPoints'].astype(bool).reshape(dims, order='F')

    abs_phi = np.abs(phi)
    phi_fixed = np.where(inside_mask, -abs_phi, abs_phi).astype(np.float32)
    return phi_fixed

# ----------------------------- Helpers -----------------------------
def _expand_bounds(bounds, pad):
    x0, x1, y0, y1, z0, z1 = bounds
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad, z0 - pad, z1 + pad)

def _filter_small_components(mesh: pv.PolyData, threshold_factor: float = 0.1, log_func=print):
    """Splits mesh and removes components with volume < threshold * max_volume."""
    if mesh.n_cells == 0:
        return mesh

    log = log_func or print
    log("...filtering small disconnected lattice components...")
    
    try:
        components = mesh.split_bodies()
        if len(components) <= 1:
            log("...no filtering needed (single component).")
            return mesh

        volumes = np.array([comp.volume for comp in components])
        max_volume = np.max(volumes)
        if max_volume == 0:
            log("...no filtering needed (all components have zero volume).")
            return mesh

        volume_threshold = threshold_factor * max_volume
        
        filtered_meshes = [comp for comp, vol in zip(components, volumes) if vol >= volume_threshold]
        num_removed = len(components) - len(filtered_meshes)
        
        if num_removed > 0:
            log(f"...removed {num_removed} of {len(components)} components below volume threshold.")
        else:
            log("...no components were smaller than the threshold.")

        if not filtered_meshes:
            log("WARNING: All components were filtered out. Returning original mesh to avoid errors.")
            return mesh
        
        return pv.MultiBlock(filtered_meshes).combine(merge_points=True)
        
    except Exception as e:
        log(f"WARNING: Could not perform volume filtering due to an error: {e}")
        return mesh

# ----------------------------- Main Infill Generation -----------------------------
def generate_infill_inside(mesh: pv.PolyData, **kwargs):
    """
    Unified implicit workflow using PyVista. Returns a single PyVista PolyData mesh.
    """
    log_func = kwargs.get('log_func', print)
    log = log_func or print
    log("--- Running Unified Implicit Workflow (PyVista Native) ---")

    shell_hole_factor = float(kwargs.get('shell_hole_factor', 0.02))
    bbox_pad_factor   = float(kwargs.get('bbox_pad_factor', 0.5))
    enclosed_tol = float(kwargs.get('enclosed_tolerance', 0.0))

    mesh = mesh.triangulate().clean()

    # --- Step 1: Grid (padded) ---
    log("Step 1/4: Creating implicit grid...", percent=10)
    resolution = int(kwargs['resolution'])
    base_bounds = np.array(mesh.bounds, dtype=float)

    if resolution > 1:
        dx = (base_bounds[1] - base_bounds[0]) / (resolution - 1)
        dy = (base_bounds[3] - base_bounds[2]) / (resolution - 1)
        dz = (base_bounds[5] - base_bounds[4]) / (resolution - 1)
        min_spacing = max(min(dx, dy, dz), 1e-9)
    else:
        min_spacing = 1e-3

    pad = bbox_pad_factor * min_spacing
    bounds = _expand_bounds(base_bounds, pad)

    xs = np.linspace(bounds[0], bounds[1], resolution)
    ys = np.linspace(bounds[2], bounds[3], resolution)
    zs = np.linspace(bounds[4], bounds[5], resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = pv.StructuredGrid(X, Y, Z)

    # --- Step 2: Base implicit fields ---
    log("Step 2/4: Generating base implicit fields...", percent=20)
    sdf_outer_1d = grid.compute_implicit_distance(mesh, inplace=False).point_data.active_scalars
    phi = _sdf_negative_inside(
        grid, mesh, sdf_outer_1d,
        log_func=log, hole_factor=shell_hole_factor, enclosed_tol=enclosed_tol
    )

    current_wx, current_wy, current_wz = kwargs['wx'], kwargs['wy'], kwargs['wz']
    if kwargs.get('use_scalar_for_cell_size') and kwargs.get('external_scalar'):
        log("...applying variable cell size from scalar field...")
        points, values = kwargs['external_scalar']
        interp_vals = griddata(points, values, grid.points, method='linear')
        interp_vals = np.nan_to_num(interp_vals, nan=np.mean(values))
        vmin, vmax = float(np.min(interp_vals)), float(np.max(interp_vals))
        norm = 0.5 + ((interp_vals - vmin) / (vmax - vmin)) if vmax - vmin > 1e-6 else np.ones_like(interp_vals)
        norm_grid = norm.reshape(grid.dimensions, order='F')
        current_wx *= norm_grid
        current_wy *= norm_grid
        current_wz *= norm_grid

    lattice_field = generate_scalar_field(grid.x, grid.y, grid.z, kwargs['lattice_type'], current_wx, current_wy, current_wz)

    if kwargs.get('solidify', False):
        thickness = kwargs.get('thickness', 1.0)
        if kwargs.get('use_scalar_for_thickness') and kwargs.get('external_scalar'):
            log("...applying variable thickness from scalar field...")
            points, values = kwargs['external_scalar']
            interp_vals = griddata(points, values, grid.points, method='linear')
            interp_vals = np.nan_to_num(interp_vals, nan=np.mean(values))
            vmin, vmax = float(np.min(interp_vals)), float(np.max(interp_vals))
            normalized = (interp_vals - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(interp_vals)
            
            # **MODIFICATION**: Correctly map the normalized scalar to user-defined bounds
            min_bound = float(kwargs.get('min_thickness_bound', 0.2))
            max_bound = float(kwargs.get('max_thickness_bound', thickness))
            thickness_field = (normalized * (max_bound - min_bound) + min_bound).reshape(grid.dimensions, order='F')

            lattice_body_field = np.abs(lattice_field) - (thickness_field / 2.0)
        else:
            lattice_body_field = np.abs(lattice_field) - (float(thickness) / 2.0)
    else:
        lattice_body_field = lattice_field

    # --- Step 3: CSG and Surface Extraction ---
    log("Step 3/4: Performing CSG and extracting surfaces...", percent=50)
    
    if kwargs.get('create_shell', False):
        log("...generating combined OPEN-CELL lattice and shell via TRIM-then-UNION.")
        t = float(kwargs['shell_thickness'])
        phi_inner = phi + t
        shell_field = np.maximum(phi, -phi_inner)
        trimmed_lattice_field = np.maximum(phi_inner - t / 2, lattice_body_field)
        final_field = np.minimum(shell_field, trimmed_lattice_field)
        grid.point_data['final'] = final_field.ravel(order='F')
        lattice_mesh_raw = grid.contour(isosurfaces=[0.0], scalars='final').triangulate().clean()
    else:
        log("...generating combined (closed-cell) lattice body via INTERSECTION.")
        final_field = np.maximum(phi, lattice_body_field)
        grid.point_data['final'] = final_field.ravel(order='F')
        lattice_mesh_raw = grid.contour(isosurfaces=[0.0], scalars='final').triangulate().clean()

    # --- Step 4: Filtering and Remeshing ---
    log("Step 4/4: Filtering and remeshing final structure...", percent=80)
    
    lattice_mesh_filtered = _filter_small_components(lattice_mesh_raw, log_func=log)
    
    if lattice_mesh_filtered.n_points == 0:
        raise RuntimeError(
            "Implicit generation and filtering resulted in an empty lattice mesh. "
            "Increase resolution, reduce shell thickness, or check geometry integrity."
        )

    # **MODIFICATION**: Correctly pass the remeshing parameters to the function.
    # The calling code (e.g., main_window.py) provides these settings.
    remesh_params = {
        'remesh_enabled': kwargs.get('remesh_enabled', False),
        'smoothing': kwargs.get('smoothing'),
        'smoothing_iterations': kwargs.get('smoothing_iterations'),
        'repair_methods': kwargs.get('repair_methods', {})
    }
    lattice_mesh_final = apply_remeshing(lattice_mesh_filtered, remesh_params, log_func)

    log("--- Workflow Complete ---", percent=100)
    return lattice_mesh_final
