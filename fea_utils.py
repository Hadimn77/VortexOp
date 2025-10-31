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
                                  lattice_model: bool = False,
                                  algorithm: str = "Delaunay"):
    """
    Creates a robust tetrahedral volumetric mesh from a surface mesh using Gmsh.
    Includes an automatic curvature-based refinement process.
    """
    # --- The Pre-processing section remains the same ---
    final_surface = None
    try:
        if skip_preprocessing:
            log_func("Skipping pre-processing steps as requested.")
            log_func("WARNING: The input mesh must be high-quality and watertight for this to succeed.")
            final_surface = surface_mesh.triangulate()
        else:
            log_func("Step 1/6: Validating input mesh...", percent=5)
            if not isinstance(surface_mesh, pv.PolyData) or surface_mesh.n_points == 0 or surface_mesh.n_cells == 0:
                return False, "Input must be a valid PyVista PolyData mesh with points and cells."
            
            repaired_mesh = surface_mesh.copy()
            
            if lattice_model:
                log_func("...using AGGRESSIVE mesh preprocssing pipeline for lattice model.")
                log_func("Step 2/6: Performing initial repairs with PyMeshFix...", percent=20)
                meshfix = mf.MeshFix(repaired_mesh.points, repaired_mesh.faces.reshape(-1, 4)[:, 1:])
                meshfix.repair()
                repaired_mesh = pv.PolyData(meshfix.v, np.hstack((np.full((meshfix.f.shape[0], 1), 3), meshfix.f)))
                repaired_mesh.clean(inplace=True).triangulate(inplace=True)
                log_func("Step 3/6: Applying adaptive simplification...", percent=50)
                simplified_surface = repaired_mesh.decimate_pro(reduction=0, feature_angle=feature_angle, preserve_topology=True)
                log_func("Step 4/6: Applying smoothing...", percent=60)
                final_surface = simplified_surface.smooth(n_iter=10)
                log_func("Step 5/6: Re-checking watertight status with PyMeshFix...", percent=80)
                meshfix = mf.MeshFix(final_surface.points, final_surface.faces.reshape(-1, 4)[:, 1:])
                meshfix.repair()
                final_surface = pv.PolyData(meshfix.v, np.hstack((np.full((meshfix.f.shape[0], 1), 3), meshfix.f)))
                final_surface.clean(inplace=True).triangulate(inplace=True)
            else:
                log_func("...using CONSERVATIVE mesh preprocssing pipeline for solid model.")
                log_func("Step 2/6: Analyzing mesh components...", percent=20)
                faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
                trimesh_mesh = trimesh.Trimesh(vertices=surface_mesh.points, faces=faces)
                components = trimesh_mesh.split(only_watertight=False)
                if len(components) > 1:
                    log_func(f"...found {len(components)} disconnected parts. Selecting the largest by volume.")
                    component_volumes = [comp.volume for comp in components]
                    trimesh_mesh = components[np.argmax(component_volumes)]
                remeshed_trimesh = trimesh_mesh.subdivide_to_size(max_edge=2*detail_size)
                faces_padded = np.hstack([np.full((remeshed_trimesh.faces.shape[0], 1), 3), remeshed_trimesh.faces])
                repaired_mesh = pv.PolyData(remeshed_trimesh.vertices, faces_padded)
                log_func("Step 3/6: Performing watertight repair with PyMeshFix...", percent=60)
                meshfix = mf.MeshFix(repaired_mesh.points, repaired_mesh.faces.reshape(-1, 4)[:, 1:])
                meshfix.repair()
                final_surface = pv.PolyData(meshfix.v, np.hstack((np.full((meshfix.f.shape[0], 1), 3), meshfix.f)))
                final_surface.clean(inplace=True).triangulate(inplace=True)
                log_func("Step 4/6: Mesh Repair complete complete.", percent=100)              

        if not final_surface.is_manifold:
            return False, "The surface mesh is not watertight after repairs. This will likely cause the volumetric meshing to fail."
            
    except Exception as e:
        return False, f"A failure occurred during pre-processing: {e}"

    # --- Gmsh Meshing section with improvements ---
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
        else: gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT

        active_fields = []
        log_func("...configuring automatic mesh refinement.")
        
        curvature_field = gmsh.model.mesh.field.add("Curvature")
        curvature_size_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(curvature_size_field, "InField", curvature_field)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "SizeMin", detail_size / 3.0)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "SizeMax", detail_size)
        # --- 3. ADAPTIVE SIZING ---
        gmsh.model.mesh.field.setNumber(curvature_size_field, "DistMin", 0.5 * detail_size)
        gmsh.model.mesh.field.setNumber(curvature_size_field, "DistMax", 3 * detail_size)
        active_fields.append(curvature_size_field)

        dist_field_base = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field_base, "SurfacesList", surface_tags)
        base_size_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(base_size_field, "InField", dist_field_base)
        gmsh.model.mesh.field.setNumber(base_size_field, "SizeMin", detail_size)
        max_size = volume_g_size if volume_g_size > detail_size else detail_size
        gmsh.model.mesh.field.setNumber(base_size_field, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(base_size_field, "DistMin", 0)
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
        gmsh.option.setNumber("Mesh.Optimize", 1) # Global optimizer
        # gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize("Relocate3D") # 3D-specific node relocation
        
        if mesh_order > 1:
            log_func(f"Setting mesh order to {mesh_order}.", percent=95)
            gmsh.model.mesh.setOrder(mesh_order)
            if optimize_ho:
                log_func("Optimizing high-order mesh...", percent=97)
                gmsh.model.mesh.optimize("HighOrder")
        
        # --- The Post-processing/Conversion section remains the same ---
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
