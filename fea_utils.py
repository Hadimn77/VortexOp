# fea_utils.py
import numpy as np
import pyvista as pv
import trimesh
import os
import tempfile
import gmsh
import shutil
from datetime import datetime

try:
    import pymeshfix as mf
except ImportError:
    raise ImportError("The 'pymeshfix' library is required for mesh repair. Install with: pip install pymeshfix")

MATERIALS = {
    "Titanium (Ti-6Al-4V)": {"ex": 113.8e9, "prxy": 0.342},
    "Aluminum (7075-T6)": {"ex": 71.7e9, "prxy": 0.33},
    "Stainless Steel (316L)": {"ex": 193e9, "prxy": 0.30},
    "Structural Steel": {"ex": 200e9, "prxy": 0.30},
}

def create_robust_volumetric_mesh(surface_mesh: pv.PolyData,
                                 detail_size: float,
                                 feature_angle: float,
                                 volume_g_size: float = 0.0,
                                 refinement_region: list = None,
                                 log_func=print,
                                 skip_preprocessing=False,
                                 mesh_order: int = 1,
                                 optimize_ho: bool = False,
                                 algorithm: str = "HXT"):
    """
    Creates a robust tetrahedral volumetric mesh from a surface mesh using Gmsh.
    Includes an automatic curvature-based refinement process.
    """
    try:
        if skip_preprocessing:
            log_func("Skipping pre-processing steps as requested.")
            log_func("WARNING: The input mesh must be high-quality and watertight for this to succeed.")
            final_surface = surface_mesh.triangulate()
        else:
            log_func("Step 1/6: Validating and converting input mesh...", percent=5)
            if not isinstance(surface_mesh, pv.PolyData) or surface_mesh.n_points == 0 or surface_mesh.n_cells == 0:
                return False, "Input to meshing function must be a valid PyVista PolyData mesh with points and cells."
            faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
            trimesh_mesh = trimesh.Trimesh(vertices=surface_mesh.points, faces=faces)
            log_func("Step 2/6: Attempting to fill holes directly...", percent=20)
            trimesh_mesh.fill_holes()
            trimesh_mesh.process()
            faces_padded = np.hstack([np.full((trimesh_mesh.faces.shape[0], 1), 3), trimesh_mesh.faces])
            repaired_surface = pv.PolyData(trimesh_mesh.vertices, faces_padded)
            log_func("Step 3/6: Applying adaptive simplification...", percent=40)
            final_surface = repaired_surface.decimate_pro(
                reduction=0.1, feature_angle=feature_angle, preserve_topology=True
            )
            final_surface = final_surface.smooth_taubin(n_iter=500, pass_band=0.05)
            log_func("Step 4/6: Performing final repair with PyMeshFix...", percent=60)
            meshfix = mf.MeshFix(final_surface.points, final_surface.faces.reshape(-1, 4)[:, 1:])
            meshfix.repair()
            final_surface = pv.PolyData(meshfix.v, np.hstack((np.full((meshfix.f.shape[0], 1), 3), meshfix.f)))
            final_surface.clean(inplace=True).triangulate(inplace=True)
        log_func("Step 5/6: Isolating the largest mesh component...", percent=75)
        faces = final_surface.faces.reshape(-1, 4)[:, 1:]
        trimesh_mesh_for_split = trimesh.Trimesh(vertices=final_surface.points, faces=faces)
        components = trimesh_mesh_for_split.split(only_watertight=False)
        if len(components) > 1:
            log_func(f"...found {len(components)} disconnected parts. Selecting the largest by volume.")
            component_volumes = [comp.volume for comp in components]
            largest_component_trimesh = components[np.argmax(component_volumes)]
            faces_padded = np.hstack([np.full((largest_component_trimesh.faces.shape[0], 1), 3), largest_component_trimesh.faces])
            final_surface = pv.PolyData(largest_component_trimesh.vertices, faces_padded)
        else:
            log_func("...mesh is a single connected body.")
        if not final_surface.is_manifold:
            error_message = (
                "The surface mesh is not watertight after adaptive repairs. "
                "This will likely cause the volumetric meshing to fail."
            )
            return False, error_message
    except Exception as e:
        return False, f"A failure occurred during pre-processing: {e}"

    temp_dir = tempfile.mkdtemp()
    temp_stl_path = os.path.join(temp_dir, "temp_surface.stl")
    gmsh.initialize()
    
    log_func(f"...enabling parallel processing on {os.cpu_count()} threads.")
    gmsh.option.setNumber("General.NumThreads", os.cpu_count())

    try:
        final_surface.save(temp_stl_path)
        gmsh.model.add("RobustMeshModel")
        gmsh.merge(temp_stl_path)
        log_func("Step 6/6: Generating 3D tetrahedral mesh with Gmsh...", percent=85)
        angle_rad = feature_angle * np.pi / 180
        gmsh.model.mesh.classifySurfaces(angle_rad, boundary=True, forReparametrization=True)
        gmsh.model.geo.synchronize()
        s = gmsh.model.getEntities(2)
        if not s:
            raise RuntimeError("Gmsh failed to create geometric surfaces from the mesh.")
        surface_tags = [tag for dim, tag in s]
        sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        log_func(f"...setting 3D algorithm to '{algorithm}'.")
        if algorithm == "Delaunay": gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        elif algorithm == "Netgen (Frontal)": gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        else: gmsh.option.setNumber("Mesh.Algorithm3D", 10)

        active_fields = []
        log_func("...configuring automatic mesh refinement.")
        
        curvature_field = gmsh.model.mesh.field.add("Curvature")
        curvature_size_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(curvature_size_field, "InField", curvature_field)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "SizeMin", detail_size / 3.0)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "SizeMax", detail_size)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "DistMax", 2.0)
        active_fields.append(curvature_size_field)

        dist_field_base = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field_base, "SurfacesList", surface_tags)
        base_size_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(base_size_field, "InField", dist_field_base)
        gmsh.model.mesh.field.setNumber(base_size_field, "SizeMin", detail_size)
        max_size = volume_g_size if volume_g_size > detail_size else detail_size
        gmsh.model.mesh.field.setNumber(base_size_field, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(base_size_field, "DistMin", detail_size * 2.0)
        gmsh.model.mesh.field.setNumber(base_size_field, "DistMax", detail_size * 5.0)
        active_fields.append(base_size_field)
        
        if refinement_region:
            box_field = gmsh.model.mesh.field.add("Box")
            box_vin = detail_size / 3.0
            box_vout = volume_g_size if volume_g_size > detail_size else detail_size * 2
            gmsh.model.mesh.field.setNumber(box_field, "VIn", box_vin); gmsh.model.mesh.field.setNumber(box_field, "VOut", box_vout)
            gmsh.model.mesh.field.setNumber(box_field, "XMin", refinement_region[0]); gmsh.model.mesh.field.setNumber(box_field, "YMin", refinement_region[1]); gmsh.model.mesh.field.setNumber(box_field, "ZMin", refinement_region[2])
            gmsh.model.mesh.field.setNumber(box_field, "XMax", refinement_region[3]); gmsh.model.mesh.field.setNumber(box_field, "YMax", refinement_region[4]); gmsh.model.mesh.field.setNumber(box_field, "ZMax", refinement_region[5])
            active_fields.append(box_field)
            
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", active_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
            
        gmsh.model.mesh.generate(3)
        
        log_func("...applying advanced quality optimization routines.")
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.model.mesh.optimize("Netgen")
        
        if mesh_order > 1:
            log_func(f"Setting mesh order to {mesh_order}.", percent=95)
            gmsh.model.mesh.setOrder(mesh_order)
            if optimize_ho:
                log_func("Optimizing high-order mesh...", percent=97)
                gmsh.model.mesh.optimize("HighOrder")
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        node_map = {tag: i for i, tag in enumerate(node_tags)}
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
        if not elem_tags or not any(tags.size for tags in elem_node_tags):
            raise RuntimeError("Gmsh generated a mesh but failed to produce any 3D elements.")
        gmsh_to_vtk = {4: pv.CellType.TETRA, 11: pv.CellType.QUADRATIC_TETRA}
        vtk_type_to_n_nodes = {pv.CellType.TETRA: 4, pv.CellType.QUADRATIC_TETRA: 10}
        cells, cell_types = [], []
        for gmsh_type, _, nodes in zip(elem_types, elem_tags, elem_node_tags):
            vtk_type = gmsh_to_vtk.get(gmsh_type)
            if vtk_type:
                num_nodes_per_elem = vtk_type_to_n_nodes.get(vtk_type)
                nodes_reshaped = nodes.reshape(-1, num_nodes_per_elem)
                for element_nodes in nodes_reshaped:
                    cells.extend([num_nodes_per_elem] + [node_map[n] for n in element_nodes])
                    cell_types.append(vtk_type)
        vol_mesh = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), node_coords)
        
    except Exception as e:
        gmsh_error = gmsh.logger.getLastError()
        saved_location_msg = ""
        if os.path.exists(temp_stl_path):
            save_dir = os.getcwd()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"latticemaker_debug_mesh_{timestamp}.stl"
            debug_filepath = os.path.join(save_dir, debug_filename)
            try:
                shutil.copy(temp_stl_path, debug_filepath)
                saved_location_msg = f"A copy of the problematic mesh has been saved: '{debug_filename}'"
            except Exception as copy_e:
                saved_location_msg = f"Could not save debug file due to: {copy_e}"
        error_details = f"GMSH ERROR: {gmsh_error}." if gmsh_error else f"An unexpected error occurred: {e}."
        return False, f"{error_details} The surface mesh may have defects. {saved_location_msg}"
    finally:
        gmsh.finalize()
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)

    log_func(f"Robust volumetric mesh created with {vol_mesh.n_cells} cells.", percent=100)
    return True, vol_mesh

def create_hexahedral_mesh(surface_mesh: pv.PolyData, voxel_size: float, log_func=print):
    log_func(f"Voxelizing mesh with target voxel size: {voxel_size}")
    density = [
        (surface_mesh.bounds[1] - surface_mesh.bounds[0]) / voxel_size,
        (surface_mesh.bounds[3] - surface_mesh.bounds[2]) / voxel_size,
        (surface_mesh.bounds[5] - surface_mesh.bounds[4]) / voxel_size,
    ]
    voxel_grid = pv.voxelize(surface_mesh, density=density)
    log_func(f"Successfully created a hexahedral mesh with {voxel_grid.n_cells} cells.")
    return voxel_grid

def check_mesh_quality(mesh: pv.UnstructuredGrid, log_func=print):
    """
    Analyzes the quality of a volumetric mesh and returns a report.

    Args:
        mesh (pv.UnstructuredGrid): The volumetric mesh to check.
        log_func (callable, optional): A function for logging progress.

    Returns:
        tuple: A tuple containing (is_pass, report_string).
    """
    if not isinstance(mesh, pv.UnstructuredGrid) or mesh.n_cells == 0:
        return False, "Mesh is empty or not a valid UnstructuredGrid."

    # --- THIS SECTION IS CORRECTED FOR ROBUSTNESS ---
    # The cell_quality() method returns a new mesh with the quality data
    # stored in it as the active scalar array.
    mesh_with_quality = mesh.cell_quality(quality_measure='scaled_jacobian')
    
    # Access the active scalar array directly without relying on its name.
    # This is the most robust method.
    quality_values = mesh_with_quality.active_scalars
    
    # Now perform calculations on the NumPy array of quality values.
    min_quality = quality_values.min()
    avg_quality = quality_values.mean()
    # --- END OF CORRECTION ---
    
    # Set a reasonable quality threshold.
    quality_threshold = 0.1

    report = (
        f"--- Mesh Quality Report ---\n"
        f"  Average Quality: {avg_quality:.3f}\n"
        f"  Minimum Quality: {min_quality:.3f}\n"
        f"  Acceptable Threshold: > {quality_threshold}\n"
        f"--------------------------"
    )
    
    if min_quality < quality_threshold:
        log_func("WARNING: Mesh quality is poor. Solver results may be inaccurate.")
        return False, report
    else:
        log_func("Mesh quality is acceptable.")
        return True, report