# direct_importer.py (v2.0 - STEP/IGES support via OpenCascade, with robust triangulation)
import os
import traceback
import numpy as np
import trimesh

# ----------------------------- Optional OpenCascade (pythonOCC / OCP) -----------------------------
# We support both package names: "OCP" (newer) and "OCC.Core" (older).
_Have_OCC = False
try:
    # OCP (preferred)
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader, STEPControl_AsIs
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import topods_Face
    _Have_OCC = True
    _OCC_VARIANT = "OCP"
except Exception:
    try:
        _Have_OCC = True
        _OCC_VARIANT = "OCC.Core"
    except Exception:
        _Have_OCC = False
        _OCC_VARIANT = None


class DirectCADImporter:
    """
    A direct importer for mesh and CAD files.
    - Mesh formats (STL/OBJ/PLY/etc.) are loaded with trimesh.
    - STEP/IGES are tessellated via OpenCascade (pythonOCC / OCP) if available.

    Triangulation produces a single Trimesh with welded vertices and no processing
    beyond basic cleanup (no repairs or decimation).
    """

    def __init__(
        self,
        linear_deflection: float = 0.2,
        angular_deflection: float = 0.5,  # radians
        parallel: bool = True,
        merge_duplicate_vertices: bool = True,
    ):
        """
        Args:
            linear_deflection: Chordal deviation (same units as the file; smaller = finer).
            angular_deflection: Angular deviation in radians (smaller = finer).
            parallel: Use OCC meshing in parallel if supported.
            merge_duplicate_vertices: Weld identical vertices after tessellation.
        """
        self.linear_deflection = float(linear_deflection)
        self.angular_deflection = float(angular_deflection)
        self.parallel = bool(parallel)
        self.merge_duplicate_vertices = bool(merge_duplicate_vertices)

    # ----------------------------- Public API -----------------------------

    def load(self, file_path: str) -> trimesh.Trimesh:
        """
        Load a geometric file and return a single Trimesh object.

        - For mesh files: returns geometry as-is (no processing).
        - For STEP/IGES: tessellates the B-Rep using OpenCascade.

        Raises:
            ValueError if loading fails or if the resulting mesh is empty.
        """
        print(f"--- Direct Import for '{os.path.basename(file_path)}' ---")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".stp", ".step", ".igs", ".iges"}:
            if not _Have_OCC:
                raise ValueError(
                    "STEP/IGES import requires OpenCascade Python bindings.\n"
                    "Please install either 'OCP' (preferred) or 'pythonocc-core'."
                )
            return self._load_step_iges_with_occ(file_path)

        # Fall back to trimesh for standard mesh formats
        try:
            scene_or_mesh = trimesh.load(file_path, process=False)
            if isinstance(scene_or_mesh, trimesh.Scene):
                print("Scene detected. Merging bodies into a single mesh.")
                model = scene_or_mesh.dump().sum()
            else:
                model = scene_or_mesh

            if not isinstance(model, trimesh.Trimesh) or model.faces.size == 0:
                raise ValueError("Loaded file is empty or not a valid mesh.")

            print("✅ Success: Model loaded without modifications.")
            return model
        except Exception as e:
            print(f"❌ Fatal import error (mesh loader): {e}")
            print(f"\nTraceback:\n{traceback.format_exc()}")
            raise ValueError(f"Could not load file: {file_path}") from e

    # ----------------------------- STEP/IGES via OCC -----------------------------

    def _load_step_iges_with_occ(self, file_path: str) -> trimesh.Trimesh:
        """
        Read a STEP/IGES file with OpenCascade and tessellate it into a single Trimesh.
        """
        print(f"[OCC] Using {_OCC_VARIANT} to read and tessellate STEP/IGES...")
        try:
            # Read the CAD file
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(file_path))
            if status != IFSelect_RetDone:
                raise ValueError(f"OCC failed to read file (status={status}).")

            # Transfer roots to a single shape
            ok = reader.TransferRoots()
            if not ok:
                raise ValueError("OCC failed to transfer STEP roots.")
            shape = reader.Shape()

            # Tessellate the B-Rep
            self._mesh_shape_with_occ(shape)

            # Extract triangles from all faces
            V, F = self._extract_mesh_from_shape(shape)

            if len(V) == 0 or len(F) == 0:
                raise ValueError("No triangles produced during OCC tessellation.")

            V = np.asarray(V, dtype=np.float64)
            F = np.asarray(F, dtype=np.int64)

            mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

            # Basic cleanup / welding (no heavy repairs)
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            if self.merge_duplicate_vertices:
                mesh.merge_vertices()

            if mesh.faces.size == 0:
                raise ValueError("Resulting mesh is empty after cleanup.")

            print("✅ Success: STEP/IGES tessellated via OpenCascade.")
            return mesh

        except Exception as e:
            print(f"❌ Fatal import error (OCC): {e}")
            print(f"\nTraceback:\n{traceback.format_exc()}")
            raise ValueError(f"Could not load file: {file_path}") from e

    def _mesh_shape_with_occ(self, shape):
        """
        Run OCC's incremental mesher on the shape with configured deflections.
        """
        # OCC API differs slightly by version; these args cover both OCP and OCC.Core
        try:
            # BRepMesh_IncrementalMesh(shape, lin_defl, isRelative, ang_defl, parallel)
            BRepMesh_IncrementalMesh(
                shape,
                self.linear_deflection,
                False,
                self.angular_deflection,
                self.parallel,
            )
        except TypeError:
            # Some builds don't have the 'parallel' argument
            BRepMesh_IncrementalMesh(
                shape,
                self.linear_deflection,
                False,
                self.angular_deflection,
            )

    def _extract_mesh_from_shape(self, shape):
        """
        Iterate faces, collect triangulations with location transforms applied.
        Returns:
            vertices (list of [x,y,z]), faces (list of [i,j,k])
        """
        vertices = []
        faces = []
        v_offset = 0

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods_Face(exp.Current())
            loc = TopLoc_Location()
            tri_handle = BRep_Tool.Triangulation(face, loc)

            # If a face wasn't meshed for some reason, skip it
            if tri_handle is None:
                exp.Next()
                continue

            # Nodes (1-based) with location transform
            tri = tri_handle  # Handle_Poly_Triangulation
            trsf = loc.Transformation()

            nodes = tri.Nodes()
            n_lower = nodes.Lower()
            n_upper = nodes.Upper()

            # Build local vertex buffer and apply transform
            local_vertices = []
            local_vertices_extend = local_vertices.extend
            for i in range(n_lower, n_upper + 1):
                p = nodes.Value(i)  # gp_Pnt
                # Apply TopLoc_Location transform
                # Copy point, then transform in-place
                p_t = p.Transformed(trsf)
                local_vertices_extend([(p_t.X(), p_t.Y(), p_t.Z())])

            # Triangles (1-based)
            tris = tri.Triangles()
            t_lower = tris.Lower()
            t_upper = tris.Upper()

            # Append to global arrays with index offset
            vertices.extend(local_vertices)
            for i in range(t_lower, t_upper + 1):
                tri_i = tris.Value(i)
                try:
                    # Newer wrappers
                    i1, i2, i3 = tri_i.Get()
                except TypeError:
                    # Older wrappers: Get returns via references (simulate)
                    refs = [0, 0, 0]
                    tri_i.Get(refs)  # type: ignore
                    i1, i2, i3 = refs
                faces.append([v_offset + (i1 - 1), v_offset + (i2 - 1), v_offset + (i3 - 1)])

            v_offset += len(local_vertices)
            exp.Next()

        return vertices, faces


# ----------------------------- Convenience function -----------------------------

def load_cad(file_path: str, **kwargs) -> trimesh.Trimesh:
    """
    Convenience wrapper:
      importer = DirectCADImporter(**kwargs)
      return importer.load(file_path)
    """
    return DirectCADImporter(**kwargs).load(file_path)
