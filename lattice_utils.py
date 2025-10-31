import numpy as np
import pyvista as pv
import trimesh
from packaging.version import parse
from scipy.interpolate import griddata
from numba import jit
import gmsh
import tempfile
import os
import shutil

MIN_PYVISTA_VERSION = "0.38.0"
assert parse(pv.__version__) >= parse(MIN_PYVISTA_VERSION), \
    f"CRITICAL ERROR: Your Python environment is using an old version of PyVista " \
    f"(found {pv.__version__}, but require >= {MIN_PYVISTA_VERSION}).\n" \
    f"Please configure your IDE to use the Python interpreter from your 'venv' folder."

# ----------------------------- Lattice Implicit Functions (Optimized with Numba) -----------------------------
@jit(nopython=True, fastmath=True, parallel=True)
def gyroid(x, y, z, wx, wy, wz):
    return (
        np.sin(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) +
        np.sin(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz) +
        np.sin(2 * np.pi * z / wz) * np.cos(2 * np.pi * x / wx)
    )

@jit(nopython=True, fastmath=True, parallel=True)
def diamond(x, y, z, wx, wy, wz):
    return (
        np.cos(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz) -
        np.sin(2 * np.pi * x / wx) * np.sin(2 * np.pi * y / wy) * np.sin(2 * np.pi * z / wz)
    )

@jit(nopython=True, fastmath=True, parallel=True)
def neovius(x, y, z, wx, wy, wz):
    return (
        3 * (np.cos(2 * np.pi * x / wx) +
             np.cos(2 * np.pi * y / wy) +
             np.cos(2 * np.pi * z / wz)) +
        4 * np.cos(2 * np.pi * x / wx) * np.cos(2 * np.pi * y / wy) * np.cos(2 * np.pi * z / wz)
    )

@jit(nopython=True, fastmath=True, parallel=True)
def schwarz_P(x, y, z, wx, wy, wz):
    return (
            (np.cos(2 * np.pi * x / wx) +
             np.cos(2 * np.pi * y / wy) +
             np.cos(2 * np.pi * z / wz))
    )

@jit(nopython=True, fastmath=True, parallel=True)
def lidinoid(x, y, z, wx, wy, wz):
    kx, ky, kz = 2 * np.pi / wx, 2 * np.pi / wy, 2 * np.pi / wz
    return (0.5 * (np.sin(2 * kx * x) * np.cos(ky * y) * np.sin(kz * z) +
                   np.sin(2 * ky * y) * np.cos(kz * z) * np.sin(kx * x) +
                   np.sin(2 * kz * z) * np.cos(kx * x) * np.sin(ky * y)) -
            0.5 * (np.cos(2 * kx * x) * np.cos(2 * ky * y) +
                   np.cos(2 * ky * y) * np.cos(2 * kz * z) +
                   np.cos(2 * kz * z) * np.cos(2 * kx * x)))

def generate_scalar_field(x, y, z, lattice_type='gyroid', wx=10, wy=10, wz=10):
    if lattice_type == 'gyroid':
        return gyroid(x, y, z, wx, wy, wz)
    elif lattice_type == 'diamond':
        return diamond(x, y, z, wx, wy, wz)
    elif lattice_type == 'neovius':
        return neovius(x, y, z, wx, wy, wz)
    elif lattice_type == 'lidinoid':
        return lidinoid(x, y, z, wx, wy, wz)
    elif lattice_type == 'schwarz_P':
        return schwarz_P(x, y, z, wx, wy, wz)
    else:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")

# ----------------------------- Utilities -----------------------------

def _remesh_surface_with_gmsh(mesh: pv.PolyData, edge_length: float, log_func=print):
    """Uses Gmsh for robust 2D surface remeshing."""
    temp_dir = tempfile.mkdtemp()
    temp_stl_path = os.path.join(temp_dir, "surface.stl")
    gmsh.initialize()
    try:
        mesh.save(temp_stl_path)
        gmsh.model.add("surface_remesh")
        gmsh.merge(temp_stl_path)
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", edge_length)
        gmsh.option.setNumber("Mesh.MeshSizeMax", edge_length)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
        
        gmsh.model.mesh.generate(2)
        
        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.model.mesh.optimize("Netgen")
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        node_map = {tag: i for i, tag in enumerate(node_tags)}
        
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        if not elem_node_tags or not any(tags.size for tags in elem_node_tags):
            raise RuntimeError("Gmsh remeshing failed to produce any 2D elements.")
        
        faces = elem_node_tags[0].reshape(-1, 3)
        faces_mapped = np.array([node_map[n] for n in faces.ravel()]).reshape(-1, 3)
        
        faces_padded = np.hstack([np.full((faces_mapped.shape[0], 1), 3), faces_mapped])
        return pv.PolyData(node_coords, faces_padded)

    except Exception as e:
        log_func(f"ERROR: Gmsh surface remeshing failed: {e}", "error")
        return mesh
    finally:
        gmsh.finalize()
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)

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
        log("...applying true adaptive remeshing via Trimesh.")
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        tri_mesh = trimesh.Trimesh(vertices=mesh.points, faces=faces)
        target_edge_length = tri_mesh.scale * 0.05 
        
        new_trimesh, _ = trimesh.remesh.remesh_isotropic(
            tri_mesh, face_length=target_edge_length, iterations=5
        )
        
        new_faces_padded = np.hstack([np.full((new_trimesh.faces.shape[0], 1), 3), new_trimesh.faces])
        mesh = pv.PolyData(new_trimesh.vertices, new_faces_padded)

    if "Loop Subdivision" in repair_methods:
        log("...applying Loop subdivision for smoothing.")
        # BUG FIX: Ensure mesh is triangulated before subdivision
        if not mesh.is_all_triangles:
            log("...triangulating mesh before subdivision.")
            mesh.triangulate(inplace=True)
        subdivisions = repair_methods["Loop Subdivision"].get('subdivisions', 1)
        mesh = mesh.subdivide(subdivisions, subfilter='loop')
    
    if "Advanced Repair" in repair_methods:
        log("...applying robust surface remeshing via Gmsh.")
        target_edge_length = np.cbrt(mesh.volume) / 10.0
        mesh = _remesh_surface_with_gmsh(mesh, target_edge_length, log_func=log)

    if smoothing == "Laplacian":
        log(f"...applying Laplacian smoothing ({smoothing_iterations} iterations).")
        mesh = mesh.smooth(n_iter=smoothing_iterations)
    elif smoothing == "Taubin":
        log(f"...applying Taubin smoothing ({smoothing_iterations} iterations).")
        mesh = mesh.smooth_taubin(n_iter=smoothing_iterations, pass_band=0.05)

    log("Post-processing completed.")
    return mesh.triangulate().clean()

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

def _expand_bounds(bounds, pad):
    x0, x1, y0, y1, z0, z1 = bounds
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad, z0 - pad, z1 + pad)

def _filter_small_components(mesh: pv.PolyData, threshold_factor: float = 0.1, log_func=print):
    """
    Splits mesh and removes small disconnected components.
    Falls back to filtering by cell count if volume calculation fails (e.g., for non-watertight meshes).
    """
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

        if max_volume > 0:
            log(f"...filtering based on volume (max volume = {max_volume:.2f}).")
            volume_threshold = threshold_factor * max_volume
            filtered_meshes = [comp for comp, vol in zip(components, volumes) if vol >= volume_threshold]
        
        else:
            log("...WARNING: Volume calculation failed (non-watertight mesh?). Falling back to filtering by cell count.")
            cell_counts = np.array([comp.n_cells for comp in components])
            max_cells = np.max(cell_counts)
            cell_threshold = threshold_factor * max_cells
            log(f"...filtering based on cell count (max cells = {max_cells}).")
            filtered_meshes = [comp for comp, count in zip(components, cell_counts) if count >= cell_threshold]
            
        num_removed = len(components) - len(filtered_meshes)
        
        if num_removed > 0:
            log(f"...removed {num_removed} of {len(components)} components below the threshold.")
        else:
            log("...no components were smaller than the threshold.")

        if not filtered_meshes:
            log("WARNING: All components were filtered out. Returning original mesh to avoid errors.")
            return mesh
        
        return pv.MultiBlock(filtered_meshes).combine(merge_points=True).extract_surface().clean()
        
    except Exception as e:
        log(f"WARNING: Could not perform component filtering due to an error: {e}")
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
    bbox_pad_factor   = float(kwargs.get('bbox_pad_factor', 1))
    enclosed_tol = float(kwargs.get('enclosed_tolerance', 0.0))

    mesh = mesh.triangulate().clean()

    # --- Step 1: Grid (padded) ---
    log("Step 1/4: Creating implicit grid...", percent=10)
    resolution = kwargs['resolution']
    base_bounds = np.array(mesh.bounds, dtype=float)

    pad = bbox_pad_factor * resolution
    bounds = _expand_bounds(base_bounds, pad)

    xs = np.arange(bounds[0], bounds[1], resolution)
    ys = np.arange(bounds[2], bounds[3], resolution)
    zs = np.arange(bounds[4], bounds[5], resolution)
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
        # Normalize the field amplitude, common for both thickness and VF methods
        lattice_type = kwargs['lattice_type']
        NORMALIZATION_FACTORS = {
            'gyroid': 1.5, 'diamond': 1.0, 'neovius': 13.0, 'lidinoid': 1.5, 'schwarz_P': 3.0
        }
        norm_factor = NORMALIZATION_FACTORS.get(lattice_type, 1.0)
        normalized_field = lattice_field / norm_factor

        isovalue = 0.0 # Default isovalue

        if kwargs.get('use_volume_fraction', False):
            target_vf = kwargs.get('target_volume_fraction', 0.2)
            log(f"...calculating isovalue for target volume fraction: {target_vf:.2f}")

            abs_field = np.abs(normalized_field)
            
            # Filter for values inside the original part boundary
            inside_mask = phi.ravel(order='F') <= 0
            field_values_inside_part = abs_field.ravel(order='F')[inside_mask]

            if field_values_inside_part.size == 0:
                raise RuntimeError("Could not determine lattice region inside the part. The part may be too thin for the chosen resolution.")

            # To get a volume fraction VF, we need to find the isovalue 'C' such that
            # the proportion of the field where |f| <= C is equal to VF.
            # This is by definition the (VF * 100)-th percentile of the |f| values.
            isovalue = np.percentile(field_values_inside_part, target_vf * 100)
            log(f"...determined isovalue {isovalue:.4f} for target volume fraction.")
        
        else: # Fallback to thickness-based logic
            thickness = kwargs.get('thickness', 1.0)
            log("...applying constant thickness correction.")
            w = np.array((current_wx + current_wy + current_wz) / 3.0)
            isovalue = (float(thickness) * np.pi) / w
            
            if kwargs.get('use_scalar_for_thickness') and kwargs.get('external_scalar'):
                log("...applying variable constant thickness from scalar field...")
                points, values = kwargs['external_scalar']
                interp_vals = griddata(points, values, grid.points, method='linear')
                interp_vals = np.nan_to_num(interp_vals, nan=np.mean(values))
                vmin, vmax = float(np.min(interp_vals)), float(np.max(interp_vals))
                normalized_thickness = (interp_vals - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(interp_vals)
                
                min_b = float(kwargs.get('min_thickness_bound', 0.5))
                max_b = float(kwargs.get('max_thickness_bound', thickness))
                thickness_field = (normalized_thickness * (max_b - min_b) + min_b)
                
                isovalue = (thickness_field * np.pi) / w.ravel(order='F')
                isovalue = isovalue.reshape(grid.dimensions, order='F')
        
        lattice_body_field = np.abs(normalized_field) - isovalue
            
    else: # Not solidified, use raw field
        lattice_body_field = lattice_field

    # --- Step 3: CSG and Surface Extraction ---
    log("Step 3/4: Performing CSG and extracting surfaces...", percent=50)
    
    if kwargs.get('create_shell', False):
        log("...generating combined OPEN-CELL lattice and shell via TRIM-then-UNION.")
        t = float(kwargs['shell_thickness'])
        phi_inner = phi + t
        shell_field = np.maximum(phi, -phi_inner)
        trimmed_lattice_field = np.maximum(phi_inner, lattice_body_field)
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
    
    remesh_params = {
        'remesh_enabled': kwargs.get('remesh_enabled', False),
        'smoothing': kwargs.get('smoothing'),
        'smoothing_iterations': kwargs.get('smoothing_iterations'),
        'repair_methods': kwargs.get('repair_methods', {})
    }
    lattice_mesh_remeshed = apply_remeshing(lattice_mesh_raw, remesh_params, log_func)

    if lattice_mesh_remeshed.n_points == 0:
        raise RuntimeError(
            "Implicit generation and filtering resulted in an empty lattice mesh. "
            "Increase resolution, reduce thickness/shell_thickness, or check geometry integrity."
        )

    lattice_mesh_final = _filter_small_components(lattice_mesh_remeshed, log_func=log)

    log("--- Workflow Complete ---", percent=100)
    return lattice_mesh_final
