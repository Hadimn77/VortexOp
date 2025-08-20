# export_utils.py
import pyvista as pv
import trimesh

def export_model(mesh, filename: str):
    """
    Exports a PyVista mesh to VTK, STL, or STEP format.

    For STEP export, the 'python-occ-core' library is required.
    Install with: pip install python-occ-core

    Args:
        mesh (pv.DataSet): The PyVista mesh to export.
        filename (str): The path to save the file to.
    """
    try:
        # Handle VTK export for volumetric grids directly
        if filename.endswith('.vtk'):
            if isinstance(mesh, pv.UnstructuredGrid):
                # Changed binary=True to binary=False for better compatibility
                mesh.save(filename, binary=False)
                return
            else:
                raise TypeError("Only volumetric meshes (UnstructuredGrid) can be saved as .vtk files.")

        # For other formats, ensure we have a surface mesh
        if isinstance(mesh, pv.UnstructuredGrid):
            surface_mesh = mesh.extract_surface().triangulate()
        else:
            surface_mesh = mesh

        # Trimesh handles the export based on file extension (for STL, STEP, etc.)
        # process=True automatically runs repair and validation routines.
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
        tri_mesh = trimesh.Trimesh(vertices=surface_mesh.points, faces=faces, process=True)
        tri_mesh.export(filename)

    except Exception as e:
        # Re-raise the exception to be caught by the main window's logger
        raise RuntimeError(f"Export to '{filename}' failed: {e}")