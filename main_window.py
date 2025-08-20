import sys
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget,
    QApplication, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QStatusBar,
    QTextEdit, QToolBar, QAction, QMenuBar, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QStyle, QHBoxLayout, QFrame, QMessageBox, QScrollArea, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# Import your custom utility modules
from cad_importer import DirectCADImporter
from lattice_utils import generate_infill_inside
from fea_utils import MATERIALS, create_robust_volumetric_mesh, create_hexahedral_mesh, check_mesh_quality
from fea_solver_core import run_native_fea
from export_utils import export_model
from lattice_optimizer import run_optimization_loop

import numpy as np
import pyvista as pv
import os
from pyvistaqt import QtInteractor
import vtk
import traceback

# Dependency checks
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
try:
    import pymeshfix
    PYMESHFIX_AVAILABLE = True
except ImportError:
    PYMESHFIX_AVAILABLE = False


class RemeshOptionsDialog(QDialog):
    # This dialog class remains unchanged
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Post-Processing Options")
        layout = QFormLayout(self)
        
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QFormLayout()
        self.smoothing_combo = QComboBox(); self.smoothing_combo.addItems(["None", "Laplacian", "Taubin"])
        smoothing_layout.addRow("Method:", self.smoothing_combo)
        self.smoothing_iterations_spin = QSpinBox(); self.smoothing_iterations_spin.setRange(10, 500); self.smoothing_iterations_spin.setValue(100)
        smoothing_layout.addRow("Iterations:", self.smoothing_iterations_spin)
        smoothing_group.setLayout(smoothing_layout)
        layout.addRow(smoothing_group)

        repair_group = QGroupBox("Repair, Simplification & Remeshing")
        repair_layout = QFormLayout()
        self.fill_holes_check = QCheckBox("Fill Holes")
        self.hole_size_spin = QDoubleSpinBox(); self.hole_size_spin.setRange(0.1, 1000.0); self.hole_size_spin.setValue(10.0)
        repair_layout.addRow(self.fill_holes_check, self.hole_size_spin)
        self.simplification_check = QCheckBox("Simplification")
        self.reduction_percentage_spin = QDoubleSpinBox(); self.reduction_percentage_spin.setRange(0.0, 0.99); self.reduction_percentage_spin.setValue(0.2)
        repair_layout.addRow(self.simplification_check, self.reduction_percentage_spin)
        self.adaptive_check = QCheckBox("Adaptive (Curvature)")
        repair_layout.addRow(self.adaptive_check)
        self.delaunay_check = QCheckBox("Delaunay")
        repair_layout.addRow(self.delaunay_check)
        repair_group.setLayout(repair_layout)
        layout.addRow(repair_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.fill_holes_check.toggled.connect(self.hole_size_spin.setVisible)
        self.simplification_check.toggled.connect(self.reduction_percentage_spin.setVisible)
        self.hole_size_spin.setVisible(False)
        self.reduction_percentage_spin.setVisible(False)

        if current_settings:
            self.smoothing_combo.setCurrentText(current_settings["smoothing"])
            self.smoothing_iterations_spin.setValue(current_settings["smoothing_iterations"])
            repair_methods = current_settings.get("repair_methods", {})
            if "Fill Holes" in repair_methods:
                self.fill_holes_check.setChecked(True); self.hole_size_spin.setValue(repair_methods["Fill Holes"]["hole_size"])
            if "Simplification" in repair_methods:
                self.simplification_check.setChecked(True); self.reduction_percentage_spin.setValue(repair_methods["Simplification"]["reduction"])
            if "Adaptive" in repair_methods: self.adaptive_check.setChecked(True)
            if "Delaunay" in repair_methods: self.delaunay_check.setChecked(True)

    def get_settings(self):
        settings = {"remesh_enabled": True, "smoothing": self.smoothing_combo.currentText(), "smoothing_iterations": self.smoothing_iterations_spin.value(), "repair_methods": {}}
        if self.fill_holes_check.isChecked(): settings["repair_methods"]["Fill Holes"] = {"hole_size": self.hole_size_spin.value()}
        if self.simplification_check.isChecked(): settings["repair_methods"]["Simplification"] = {"reduction": self.reduction_percentage_spin.value()}
        if self.adaptive_check.isChecked(): settings["repair_methods"]["Adaptive"] = {}
        if self.delaunay_check.isChecked(): settings["repair_methods"]["Delaunay"] = {}
        return settings

class ClippingOptionsDialog(QDialog):
    # This dialog class remains unchanged
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Clipping Tool Settings")
        layout = QFormLayout(self)
        self.enable_clipping_check = QCheckBox("Enable Clipping")
        layout.addRow(self.enable_clipping_check)
        self.invert_cut_check = QCheckBox("Invert Cut")
        layout.addRow(self.invert_cut_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if current_settings:
            self.enable_clipping_check.setChecked(current_settings.get("enabled", False))
            self.invert_cut_check.setChecked(current_settings.get("invert", False))
        self.invert_cut_check.setEnabled(self.enable_clipping_check.isChecked())
        self.enable_clipping_check.toggled.connect(self.invert_cut_check.setEnabled)

    def get_settings(self):
        return {"enabled": self.enable_clipping_check.isChecked(), "invert": self.invert_cut_check.isChecked()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LatticeMaker")
        self.setGeometry(100, 100, 1200, 800)
        self.importer = DirectCADImporter()
        
        # --- KEY CHANGE: UPDATED MESH ATTRIBUTES ---
        # Replaced self.surface_mesh with two distinct attributes for clarity and correctness
        self.original_pv_shell = None
        self.surface_mesh = None
        self.volumetric_mesh = None
        self.fea_result_model = None
        self.external_scalar = None
        # --- END OF KEY CHANGE ---
        
        self.fixed_node_indices = set()
        self.load_node_indices = set()
        self.fixed_node_actor = None
        self.load_node_actor = None
        self.main_mesh_actor = None
        self.selection_surface = None
        self.remesh_settings = {
            "remesh_enabled": True, 
            "smoothing": "Taubin", 
            "smoothing_iterations": 10, 
            "repair_methods": {}
        }
        self.clipping_settings = {"enabled": False, "invert": False}
        self._is_box_selection_mode = False
        self._create_widgets()
        pv.set_plot_theme('dark')
        self.plotter.set_background('#2d2d2d')
        self._create_layouts()
        self._create_menu_bar()
        self._create_tool_bar()
        self._connect_signals()
        self.setCentralWidget(self.main_container)
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(100)
        self.fea_group.setEnabled(False)
        self.optim_group.setEnabled(False)
        self._check_dependencies()
        self._on_shell_toggle(False)
        self.use_scalar_for_cell_size_checkbox.setEnabled(False)
        self.use_scalar_checkbox.setEnabled(False)
        self.show_voxel_preview_check.setEnabled(False)
        # --- NEW WIDGET INITIAL STATE ---
        self.show_scalar_field_check.setEnabled(False)
        self._on_element_type_change()
        self._update_thickness_limit()

    def _check_dependencies(self):
        if not GMSH_AVAILABLE or not PYMESHFIX_AVAILABLE:
            self.log("WARNING: 'gmsh' and/or 'pymeshfix' not found.")
            self.generate_tet_mesh_button.setEnabled(False)
            self.generate_tet_mesh_button.setToolTip("Install with: pip install gmsh pymeshfix")

    def _create_widgets(self):
        # This method is unchanged
        self.plotter = QtInteractor(self)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.status_bar = QStatusBar(); self.log_output = QTextEdit()
        self.lattice_type_box = QComboBox(); self.lattice_type_box.addItems(['gyroid', 'diamond', 'neovius', 'schwarz_p'])
        self.resolution_spin = QSpinBox(); self.resolution_spin.setRange(20, 500); self.resolution_spin.setValue(100)
        self.suggest_resolution_button = QPushButton("Auto-resolution")
        self.unit_x_spin = QDoubleSpinBox(); self.unit_x_spin.setRange(0.01, 100.0); self.unit_x_spin.setValue(10.0)
        self.unit_y_spin = QDoubleSpinBox(); self.unit_y_spin.setRange(0.01, 100.0); self.unit_y_spin.setValue(10.0)
        self.unit_z_spin = QDoubleSpinBox(); self.unit_z_spin.setRange(0.01, 100.0); self.unit_z_spin.setValue(10.0)
        self.lattice_thickness_spin = QDoubleSpinBox(); self.lattice_thickness_spin.setRange(0.01, 10.0); self.lattice_thickness_spin.setValue(1.0)
        self.solidify_checkbox = QCheckBox("Solidify Lattice"); self.shell_checkbox = QCheckBox("Create Shell")
        self.shell_thickness_spin = QDoubleSpinBox(); self.shell_thickness_spin.setRange(0.1, 50.0); self.shell_thickness_spin.setValue(1.0)
        self.remesh_button = QPushButton("Post-Processing Settings")
        self.infill_button = QPushButton("Generate Lattice"); self.infill_button.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.use_scalar_for_cell_size_checkbox = QCheckBox("Use Scalar for Cell Size")
        self.use_scalar_checkbox = QCheckBox("Use Scalar for Thickness")
        self.scalar_button = QPushButton("Load Scalar Field"); self.scalar_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.element_type_group = QButtonGroup(self)
        self.tet_radio = QRadioButton("Tetrahedral"); self.tet_radio.setChecked(True)
        self.hex_radio = QRadioButton("Hexahedral (Voxel)"); self.element_type_group.addButton(self.tet_radio); self.element_type_group.addButton(self.hex_radio)
        self.detail_size_label = QLabel("Detail Size (Surface/Voxel):")
        self.detail_size_spin = QDoubleSpinBox(); self.detail_size_spin.setRange(0.001, 100.0); self.detail_size_spin.setValue(1.0); self.detail_size_spin.setSingleStep(0.01); self.detail_size_spin.setDecimals(4)
        self.volume_g_size_label = QLabel("Volume Growth Size:")
        self.volume_g_size_spin = QDoubleSpinBox(); self.volume_g_size_spin.setRange(0.0, 500.0); self.volume_g_size_spin.setValue(0.0); self.volume_g_size_spin.setSingleStep(0.1); self.volume_g_size_spin.setDecimals(4)
        self.volume_g_size_spin.setToolTip("Set a value larger than Detail Size to create a coarser mesh in the volume.\nSet to 0 for a uniform mesh.")
        self.mesh_algo_combo = QComboBox(); self.mesh_algo_combo.addItems(["HXT", "Delaunay", "Netgen (Frontal)"])
        self.mesh_order_spin = QSpinBox(); self.mesh_order_spin.setRange(1, 2); self.mesh_order_spin.setValue(1)
        self.ho_optimize_check = QCheckBox("Optimize High-Order Mesh"); self.ho_optimize_check.setChecked(True); self.ho_optimize_check.setEnabled(False)
        self.generate_tet_mesh_button = QPushButton("Generate Tetrahedral Mesh"); self.generate_tet_mesh_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.feature_angle_label = QLabel("Feature Angle:")
        self.feature_angle_spin = QDoubleSpinBox(); self.feature_angle_spin.setRange(10.0, 120.0); self.feature_angle_spin.setValue(30.0); self.feature_angle_spin.setSingleStep(5.0)
        self.skip_preprocessing_check = QCheckBox("Skip Pre-processing & Repair"); self.skip_preprocessing_check.setToolTip("WARNING: Only use this for high-quality, watertight models.\nSkipping this step on a flawed model will likely cause meshing to fail.")
        self.generate_hex_mesh_button = QPushButton("Generate Hexahedral Mesh")
        self.material_combo = QComboBox(); self.material_combo.addItems(MATERIALS.keys())
        self.fx_spin = QDoubleSpinBox(); self.fx_spin.setRange(-1e9, 1e9)
        self.fy_spin = QDoubleSpinBox(); self.fy_spin.setRange(-1e9, 1e9)
        self.fz_spin = QDoubleSpinBox(); self.fz_spin.setRange(-1e9, 1e9); self.fz_spin.setValue(-1000)
        self.select_toggle_button = QPushButton("Activate Node Selection"); self.select_toggle_button.setCheckable(True); self.select_toggle_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowRight))
        self.selection_target_group = QButtonGroup(self)
        self.fixed_bc_radio = QRadioButton("Select Fixed"); self.fixed_bc_radio.setChecked(True)
        self.load_bc_radio = QRadioButton("Select Load"); self.selection_target_group.addButton(self.fixed_bc_radio); self.selection_target_group.addButton(self.load_bc_radio)
        self.fixed_node_label = QLabel("0 nodes"); self.load_node_label = QLabel("0 nodes")
        self.clear_fea_button = QPushButton("Clear Selections"); self.clear_fea_button.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogResetButton))
        self.run_fea_button = QPushButton("Run Simulation"); self.run_fea_button.setIcon(QApplication.style().standardIcon(QStyle.SP_ComputerIcon))
        self.refinement_group = QGroupBox("Regional Refinement"); self.refinement_group.setCheckable(True); self.refinement_group.setChecked(False)
        self.ref_xmin_spin = QDoubleSpinBox(); self.ref_xmin_spin.setRange(-1e6, 1e6); self.ref_xmin_spin.setDecimals(2)
        self.ref_xmax_spin = QDoubleSpinBox(); self.ref_xmax_spin.setRange(-1e6, 1e6); self.ref_xmax_spin.setDecimals(2)
        self.ref_ymin_spin = QDoubleSpinBox(); self.ref_ymin_spin.setRange(-1e6, 1e6); self.ref_ymin_spin.setDecimals(2)
        self.ref_ymax_spin = QDoubleSpinBox(); self.ref_ymax_spin.setRange(-1e6, 1e6); self.ref_ymax_spin.setDecimals(2)
        self.ref_zmin_spin = QDoubleSpinBox(); self.ref_zmin_spin.setRange(-1e6, 1e6); self.ref_zmin_spin.setDecimals(2)
        self.ref_zmax_spin = QDoubleSpinBox(); self.ref_zmax_spin.setRange(-1e6, 1e6); self.ref_zmax_spin.setDecimals(2)
        self.get_bounds_button = QPushButton("Get from Model Bounds")
        self.optim_max_iter_spin = QSpinBox(); self.optim_max_iter_spin.setRange(1, 50); self.optim_max_iter_spin.setValue(5)
        self.optim_stress_reduc_spin = QDoubleSpinBox(); self.optim_stress_reduc_spin.setRange(1.0, 99.0); self.optim_stress_reduc_spin.setValue(30.0); self.optim_stress_reduc_spin.setSuffix(" %")
        self.run_optimization_button = QPushButton("Run Optimization"); self.run_optimization_button.setIcon(QApplication.style().standardIcon(QStyle.SP_CommandLink))
        self.view_selector = QComboBox(); self.view_selector.addItems(["CAD", "Result", "Volumetric Mesh", "FEA Result", "Optimized Result"])
        self.fea_result_selector = QComboBox(); self.fea_result_selector.addItems(["von_mises_stress", "displacement", "principal_s1", "principal_s2", "principal_s3"]); self.fea_result_selector.setVisible(False)
        self.clipping_plane_button = QPushButton("Clipping Tool Settings")
        self.show_voxel_preview_check = QCheckBox("Show Voxel Preview")
        # --- NEW WIDGET INSTANTIATION ---
        self.show_scalar_field_check = QCheckBox("Show Scalar Field")
        self.show_deformation_check = QCheckBox("Show Deformed Shape")
        self.deformation_scale_spin = QDoubleSpinBox(); self.deformation_scale_spin.setRange(0, 1e6); self.deformation_scale_spin.setValue(100.0); self.deformation_scale_spin.setSingleStep(10); self.deformation_scale_spin.setDecimals(1)

    def _create_tool_bar(self):
        # This method is unchanged
        self.toolbar = QToolBar("Controls"); self.addToolBar(Qt.LeftToolBarArea, self.toolbar); self.toolbar.setMovable(False)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        toolbar_container = QWidget(); toolbar_layout = QVBoxLayout(toolbar_container); toolbar_layout.setAlignment(Qt.AlignTop)
        lattice_group = QGroupBox("Lattice Generation"); lattice_layout = QFormLayout(); lattice_group.setLayout(lattice_layout)
        lattice_layout.addRow("Type:", self.lattice_type_box)
        resolution_layout = QHBoxLayout(); resolution_layout.addWidget(self.resolution_spin); resolution_layout.addWidget(self.suggest_resolution_button)
        lattice_layout.addRow("Resolution:", resolution_layout)
        lattice_layout.addRow("Cell X:", self.unit_x_spin); lattice_layout.addRow("Cell Y:", self.unit_y_spin); lattice_layout.addRow("Cell Z:", self.unit_z_spin)
        lattice_layout.addRow(self.solidify_checkbox); lattice_layout.addRow("Lattice Thickness:", self.lattice_thickness_spin)
        lattice_layout.addRow(self.shell_checkbox); self.shell_thickness_label = QLabel("Shell Thickness:"); lattice_layout.addRow(self.shell_thickness_label, self.shell_thickness_spin)
        lattice_layout.addRow(self.use_scalar_for_cell_size_checkbox); lattice_layout.addRow(self.use_scalar_checkbox); lattice_layout.addRow(self.scalar_button)
        lattice_layout.addRow(self.remesh_button); lattice_layout.addRow(self.infill_button)
        toolbar_layout.addWidget(lattice_group)
        self.fea_group = QGroupBox("FEA Toolbox"); fea_layout = QFormLayout(); self.fea_group.setLayout(fea_layout)
        fea_layout.addRow(QLabel("<b>Volumetric Meshing</b>"))
        element_type_layout = QHBoxLayout(); element_type_layout.addWidget(self.tet_radio); element_type_layout.addWidget(self.hex_radio)
        fea_layout.addRow("Element Type:", element_type_layout)
        fea_layout.addRow("3D Algorithm:", self.mesh_algo_combo)
        fea_layout.addRow("Mesh Order:", self.mesh_order_spin)
        fea_layout.addRow(self.ho_optimize_check)
        line0 = QFrame(); line0.setFrameShape(QFrame.HLine); line0.setFrameShadow(QFrame.Sunken); fea_layout.addRow(line0)
        self.tet_meshing_widgets = [self.feature_angle_label, self.feature_angle_spin, self.volume_g_size_label, self.volume_g_size_spin, self.generate_tet_mesh_button, self.mesh_algo_combo, self.mesh_order_spin, self.ho_optimize_check, self.skip_preprocessing_check]
        self.hex_meshing_widgets = [self.generate_hex_mesh_button]
        fea_layout.addRow(self.detail_size_label, self.detail_size_spin)
        fea_layout.addRow(self.volume_g_size_label, self.volume_g_size_spin)
        fea_layout.addRow(self.feature_angle_label, self.feature_angle_spin)
        fea_layout.addRow(self.skip_preprocessing_check)
        fea_layout.addRow(self.generate_tet_mesh_button)
        fea_layout.addRow(self.generate_hex_mesh_button)
        line_refine = QFrame(); line_refine.setFrameShape(QFrame.HLine); line_refine.setFrameShadow(QFrame.Sunken); fea_layout.addRow(line_refine)
        refinement_layout = QFormLayout()
        refinement_layout.addRow("X Range:", self._create_hbox(self.ref_xmin_spin, self.ref_xmax_spin))
        refinement_layout.addRow("Y Range:", self._create_hbox(self.ref_ymin_spin, self.ref_ymax_spin))
        refinement_layout.addRow("Z Range:", self._create_hbox(self.ref_zmin_spin, self.ref_zmax_spin))
        refinement_layout.addRow(self.get_bounds_button)
        self.refinement_group.setLayout(refinement_layout)
        fea_layout.addRow(self.refinement_group)
        line1 = QFrame(); line1.setFrameShape(QFrame.HLine); line1.setFrameShadow(QFrame.Sunken); fea_layout.addRow(line1)
        fea_layout.addRow(QLabel("<b>Simulation Setup</b>"))
        fea_layout.addRow("Material:", self.material_combo)
        force_layout = QHBoxLayout(); force_layout.addWidget(QLabel("Fx")); force_layout.addWidget(self.fx_spin); force_layout.addWidget(QLabel("Fy")); force_layout.addWidget(self.fy_spin); force_layout.addWidget(QLabel("Fz")); force_layout.addWidget(self.fz_spin)
        fea_layout.addRow("Force (N):", force_layout)
        fea_layout.addRow(self.select_toggle_button)
        selection_type_layout = QHBoxLayout(); selection_type_layout.addWidget(self.fixed_bc_radio); selection_type_layout.addWidget(self.load_bc_radio)
        fea_layout.addRow(selection_type_layout)
        fea_layout.addRow(QLabel("Fixed Nodes:"), self.fixed_node_label)
        fea_layout.addRow(QLabel("Load Nodes:"), self.load_node_label)
        fea_layout.addRow(self.clear_fea_button); fea_layout.addRow(self.run_fea_button)
        toolbar_layout.addWidget(self.fea_group)
        self.optim_group = QGroupBox("Lattice Optimization")
        optim_layout = QFormLayout()
        self.optim_group.setLayout(optim_layout)
        optim_layout.addRow("Max Iterations:", self.optim_max_iter_spin)
        optim_layout.addRow("Stress Reduction Target:", self.optim_stress_reduc_spin)
        optim_layout.addRow(self.run_optimization_button)
        toolbar_layout.addWidget(self.optim_group)
        view_group = QGroupBox("View Controls"); view_layout = QFormLayout(); view_group.setLayout(view_layout)
        self.deformation_scale_label = QLabel("Deformation Scale:")
        view_layout.addRow(self.show_deformation_check)
        view_layout.addRow(self.deformation_scale_label, self.deformation_scale_spin)
        view_layout.addRow("View Mode:", self.view_selector); view_layout.addRow("FEA Scalar:", self.fea_result_selector)
        # --- ADD NEW WIDGET TO LAYOUT ---
        view_layout.addRow(self.show_scalar_field_check)
        view_layout.addRow(self.show_voxel_preview_check)
        view_layout.addRow(self.clipping_plane_button)
        toolbar_layout.addWidget(view_group)
        scroll_area.setWidget(toolbar_container)
        self.toolbar.addWidget(scroll_area)
    
    def _create_layouts(self):
        main_layout = QVBoxLayout(); main_layout.addWidget(self.plotter.interactor); main_layout.addWidget(self.log_output)
        self.main_container = QWidget(); self.main_container.setLayout(main_layout)

    def _create_menu_bar(self):
        menubar = self.menuBar(); file_menu = menubar.addMenu("File")
        self.open_stl_action = QAction("Open Model", self); file_menu.addAction(self.open_stl_action)
        self.load_vol_action = QAction("Load Volumetric Mesh", self); file_menu.addAction(self.load_vol_action)
        self.save_vol_action = QAction("Save Volumetric Mesh", self); file_menu.addAction(self.save_vol_action)
        self.export_action = QAction("Export Model", self); file_menu.addAction(self.export_action)
        
    def _connect_signals(self):
        self.open_stl_action.triggered.connect(self.import_file)
        self.export_action.triggered.connect(self.export_current_model)
        self.save_vol_action.triggered.connect(self.save_volumetric_mesh)
        self.load_vol_action.triggered.connect(self.load_volumetric_mesh)
        self.remesh_button.clicked.connect(self.open_remesh_dialog)
        self.shell_checkbox.toggled.connect(self._on_shell_toggle)
        self.infill_button.clicked.connect(self.run_generation)
        self.view_selector.currentIndexChanged.connect(self.update_view)
        self.fea_result_selector.currentIndexChanged.connect(self.update_view)
        self.scalar_button.clicked.connect(self.load_scalar_field)
        self.select_toggle_button.toggled.connect(self._set_selection_mode)
        self.clear_fea_button.clicked.connect(self._clear_fea_selections)
        
        # --- THIS IS THE CORRECTED LINE ---
        self.generate_tet_mesh_button.clicked.connect(self.run_robust_tet_meshing)
        
        self.generate_hex_mesh_button.clicked.connect(self.run_hex_meshing)
        self.run_fea_button.clicked.connect(self.run_simulation)
        self.show_voxel_preview_check.toggled.connect(self.update_view)
        # --- CONNECT NEW WIDGET'S SIGNAL ---
        self.show_scalar_field_check.toggled.connect(self.update_view)
        self.clipping_plane_button.clicked.connect(self.open_clipping_dialog)
        self.tet_radio.toggled.connect(self._on_element_type_change)
        self.mesh_order_spin.valueChanged.connect(self._on_order_change)
        self.get_bounds_button.clicked.connect(self._get_model_bounds_for_refinement)
        self.show_deformation_check.toggled.connect(self.update_view)
        self.deformation_scale_spin.valueChanged.connect(self.update_view)
        self.run_optimization_button.clicked.connect(self.run_optimization)
        self.suggest_resolution_button.clicked.connect(self._suggest_resolution)
        for spin_box in [self.unit_x_spin, self.unit_y_spin, self.unit_z_spin]:
            spin_box.valueChanged.connect(self._update_thickness_limit)

    def _update_thickness_limit(self):
        min_cell_size = min(self.unit_x_spin.value(), self.unit_y_spin.value(), self.unit_z_spin.value())
        max_thickness = min_cell_size / 2.0
        max_thickness = max(self.lattice_thickness_spin.minimum(), max_thickness)
        self.lattice_thickness_spin.setRange(self.lattice_thickness_spin.minimum(), max_thickness)

    def _suggest_resolution(self):
        if not self.original_pv_shell:
            self.log("Load a model first to get its bounds.")
            return
        bounds = self.original_pv_shell.bounds
        model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        min_cell_size = min(self.unit_x_spin.value(), self.unit_y_spin.value(), self.unit_z_spin.value())
        if min_cell_size < 1e-6:
            self.log("Cannot suggest resolution for a near-zero cell size.")
            return
        suggested_res = (model_size / min_cell_size) * 5.0
        min_res, max_res = self.resolution_spin.minimum(), self.resolution_spin.maximum()
        final_res = int(np.clip(suggested_res, min_res, max_res))
        self.resolution_spin.setValue(final_res)
        self.log(f"Suggested Resolution: {final_res} (based on model/cell size ratio)")

    def _on_order_change(self, value):
        self.ho_optimize_check.setEnabled(value > 1)

    def _on_element_type_change(self):
        is_tet = self.tet_radio.isChecked()
        for widget in self.tet_meshing_widgets: widget.setVisible(is_tet)
        for widget in self.hex_meshing_widgets: widget.setVisible(not is_tet)
        self.detail_size_label.setVisible(True); self.detail_size_spin.setVisible(True)

    def _on_shell_toggle(self, checked):
        self.shell_thickness_label.setVisible(checked); self.shell_thickness_spin.setVisible(checked)
        if checked:
            self.solidify_checkbox.setChecked(True); self.solidify_checkbox.setEnabled(False)
        else:
            self.solidify_checkbox.setEnabled(True)
            
    def open_clipping_dialog(self):
        dialog = ClippingOptionsDialog(self, self.clipping_settings)
        if dialog.exec_() == QDialog.Accepted: self.clipping_settings = dialog.get_settings(); self.log("Clipping settings updated."); self.update_view()

    def open_remesh_dialog(self):
        dialog = RemeshOptionsDialog(self, self.remesh_settings)
        if dialog.exec_() == QDialog.Accepted: self.remesh_settings = dialog.get_settings(); self.log("Post-processing settings updated.")

    def run_robust_tet_meshing(self):
        target_mesh = self.surface_mesh if self.surface_mesh else self.original_pv_shell
        
        if not target_mesh:
            self.log("A model must be generated or imported first.", level="error")
            return
            
        self.set_busy(True)
        self.log("Starting robust tetrahedral meshing pipeline...")
        self._clear_fea_selections()
        try:
            params = {
                'surface_mesh': target_mesh,
                'detail_size': self.detail_size_spin.value(),
                'feature_angle': self.feature_angle_spin.value(),
                'volume_g_size': self.volume_g_size_spin.value(),
                'skip_preprocessing': self.skip_preprocessing_check.isChecked(),
                'mesh_order': self.mesh_order_spin.value(),
                'optimize_ho': self.ho_optimize_check.isChecked(),
                'algorithm': self.mesh_algo_combo.currentText(),
                'log_func': self.log
            }
            if self.refinement_group.isChecked():
                params['refinement_region'] = [self.ref_xmin_spin.value(), self.ref_ymin_spin.value(), self.ref_zmin_spin.value(), self.ref_xmax_spin.value(), self.ref_ymax_spin.value(), self.ref_zmax_spin.value()]
            
            success, result = create_robust_volumetric_mesh(**params)
            
            if success:
                self.volumetric_mesh = result
                self.volumetric_mesh.point_data['persistent_ids'] = np.arange(self.volumetric_mesh.n_points)
                self.log("Successfully created robust volumetric mesh.", level="success")
                is_pass, report = check_mesh_quality(self.volumetric_mesh, self.log)
                self.log(report)
                if not is_pass:
                    QMessageBox.warning(self, "Poor Mesh Quality", f"The generated mesh has low-quality elements, which can lead to inaccurate simulation results.\n\nConsider adjusting meshing parameters.\n\n{report}")
                self.update_view("Volumetric Mesh")
            else:
                error_message = result
                QMessageBox.critical(self, "Meshing Failed", error_message)
                self.log(f"Volumetric meshing failed: {error_message}", level="error")
        except Exception as e:
            self.log(f"An unexpected error occurred in the meshing process: {e}", level="error")
            traceback.print_exc()
        finally:
            self.set_busy(False)

    def run_hex_meshing(self):
        target_mesh = self.surface_mesh if self.surface_mesh else self.original_pv_shell
        
        if not target_mesh:
            self.log("A model must be generated or imported first.", level="error")
            return
            
        self.set_busy(True)
        self.log("Generating hexahedral (voxel) mesh...")
        self._clear_fea_selections()
        try:
            voxel_size = self.detail_size_spin.value()
            self.volumetric_mesh = create_hexahedral_mesh(target_mesh, voxel_size, self.log)
            if self.volumetric_mesh:
                self.volumetric_mesh.point_data['persistent_ids'] = np.arange(self.volumetric_mesh.n_points)
                self.log("Successfully created hexahedral mesh.", level="success")
                self.update_view("Volumetric Mesh")
        except Exception as e:
            self.log(f"Hex mesh creation failed: {e}", level="error")
            traceback.print_exc()
        finally:
            self.set_busy(False)

    def run_generation(self):
        if not self.original_pv_shell:
            self.log("Import a model first.", "error")
            return
            
        self.set_busy(True)
        self.log("Starting lattice generation...")
        try:
            params = {
                'mesh': self.original_pv_shell,
                'resolution': self.resolution_spin.value(),
                'wx': self.unit_x_spin.value(),
                'wy': self.unit_y_spin.value(),
                'wz': self.unit_z_spin.value(),
                'lattice_type': self.lattice_type_box.currentText(),
                'thickness': self.lattice_thickness_spin.value(),
                'log_func': self.log,
                **self.remesh_settings,
                'solidify': self.solidify_checkbox.isChecked(),
                'create_shell': self.shell_checkbox.isChecked(),
                'shell_thickness': self.shell_thickness_spin.value(),
                'external_scalar': self.external_scalar,
                'use_scalar_for_cell_size': self.use_scalar_for_cell_size_checkbox.isChecked() and self.external_scalar is not None,
                'use_scalar_for_thickness': self.solidify_checkbox.isChecked() and self.use_scalar_checkbox.isChecked() and self.external_scalar is not None
            }
            
            # The function returns a single mesh.
            lattice_infilled_surface_mesh = generate_infill_inside(**params)

            # Store the mesh in its dedicated attribute
            self.surface_mesh = lattice_infilled_surface_mesh
            
            self.log("Generation completed.", "success")
            self.update_view("Result")
            self.fea_group.setEnabled(True)
            self.optim_group.setEnabled(False)
            
        except Exception as e:
            self.log(f"Generation error: {e}", "error")
            traceback.print_exc()
        finally:
            self.set_busy(False)
            
    def load_scalar_field(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Scalar Field", "", "Text Files (*.txt)")
        if file_path:
            try:
                data = np.loadtxt(file_path, skiprows=1)
                if data.shape[1] != 4:
                    raise ValueError("Expected Nx4 matrix (X,Y,Z,Value)")
                self.external_scalar = (data[:, :3], data[:, 3])
                self.update_view()
                self.log(f"Loaded scalar field from: {os.path.basename(file_path)}")
                self.use_scalar_for_cell_size_checkbox.setEnabled(True)
                self.use_scalar_checkbox.setEnabled(True)
                # --- ENABLE THE NEW CHECKBOX ---
                self.show_scalar_field_check.setEnabled(True)
            except Exception as e:
                self.log(f"Scalar load error: {e}", "error")

    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "Mesh Files (*.stl *.obj *.ply *.step *.stp *.iges *.igs)")
        if file_path:
            self.log("--- Starting Advanced Import ---")
            model = self.importer.load(file_path)
            if model:
                # --- CLEAR OLD MESH DATA ---
                self.surface_mesh = None
                self.volumetric_mesh = None
                self.fea_result_model = None
                self.external_scalar = None
                
                self.clipping_settings['enabled'] = False
                self._clear_fea_selections()
                self.fea_group.setEnabled(False)
                self.optim_group.setEnabled(False)
                faces_padded = np.hstack([np.full((model.faces.shape[0], 1), 3), model.faces])
                self.original_pv_shell = pv.PolyData(model.vertices, faces_padded)
                self.update_view("CAD")
                self.log("--- Import Finished ---")
                self.use_scalar_for_cell_size_checkbox.setEnabled(False)
                self.use_scalar_checkbox.setEnabled(False)
                self.show_voxel_preview_check.setEnabled(True)
                # --- DISABLE SCALAR CHECKBOX ON NEW IMPORT ---
                self.show_scalar_field_check.setEnabled(False)
                self.fea_group.setEnabled(True)
            else:
                QMessageBox.critical(self, "Import Error", f"Failed to import and repair the model.\n\nDetails:\n")
                self.log("--- Import Failed ---", "error")

    def _set_selection_mode(self, enabled):
        if enabled:
            if self.clipping_settings.get("enabled", False): self.log("Selection mode cannot be used while Clipping Tool is active.", "error"); self.select_toggle_button.setChecked(False); return
            if not self.volumetric_mesh: self.log("A volumetric mesh must be created first.", "error"); self.select_toggle_button.setChecked(False); return
            if not self.main_mesh_actor: self.log("Error: No valid mesh actor in the scene to select from.", "error"); self.select_toggle_button.setChecked(False); return
            self.log("Activating surface selection mode...")
            self.main_mesh_actor.SetVisibility(False)
            self.selection_surface = self.volumetric_mesh.extract_surface(pass_pointid=True)
            self.selection_actor = self.plotter.add_mesh(self.selection_surface, style='surface', color='orange', show_edges=True)
            self.plotter.enable_cell_picking(callback=self._handle_cell_selection, show_message=False)
            self.log("Selection mode is ON. Please select faces on the model.")
        else:
            self.plotter.disable_picking()
            if hasattr(self, 'selection_actor') and self.selection_actor: self.plotter.remove_actor(self.selection_actor, render=False); self.selection_actor = None
            self.selection_surface = None
            if self.main_mesh_actor: self.main_mesh_actor.SetVisibility(True)
            self.plotter.render()
            self.log("Selection mode is OFF.")

    def _handle_cell_selection(self, picked_object):
        if not hasattr(self, 'selection_surface') or self.selection_surface is None: return
        meshes_to_process = []
        if isinstance(picked_object, pv.MultiBlock):
            for block in picked_object:
                if block and block.n_points > 0: meshes_to_process.append(block)
        elif picked_object and picked_object.n_points > 0:
            meshes_to_process.append(picked_object)
        if not meshes_to_process: return
        all_persistent_ids = set()
        for mesh_part in meshes_to_process:
            if 'persistent_ids' not in mesh_part.point_data: self.log("Selection Warning: A portion of the selection could not be mapped to persistent IDs."); continue
            all_persistent_ids.update(mesh_part.point_data['persistent_ids'])
        if not all_persistent_ids: self.log("Selection did not yield any nodes."); return
        target_set = self.fixed_node_indices if self.fixed_bc_radio.isChecked() else self.load_node_indices
        added_count, removed_count = 0, 0
        for persistent_id in all_persistent_ids:
            if persistent_id in target_set: target_set.remove(persistent_id); removed_count += 1
            else: target_set.add(persistent_id); added_count += 1
        if added_count > 0 or removed_count > 0:
            target_name = "Fixed" if self.fixed_bc_radio.isChecked() else "Load"
            self.log(f"Selection updated: Added {added_count}, Removed {removed_count}. Total {target_name} Nodes: {len(target_set)}")
        self._update_selection_highlight()

    def _update_selection_highlight(self, render=True):
        for actor_name in ['fixed_node_actor', 'load_node_actor']:
            if hasattr(self, actor_name) and getattr(self, actor_name): self.plotter.remove_actor(getattr(self, actor_name), render=False)
            setattr(self, actor_name, None)
        highlight_mesh = self.selection_surface if self.selection_surface else (self.volumetric_mesh.extract_surface(pass_pointid=True) if self.volumetric_mesh else None)
        if highlight_mesh is None:
            self.fixed_node_label.setText(f"{len(self.fixed_node_indices)} nodes")
            self.load_node_label.setText(f"{len(self.load_node_indices)} nodes")
            if render: self.plotter.render()
            return
        id_map = highlight_mesh.point_data.get('persistent_ids')
        if id_map is None: self.log("Warning: Cannot draw highlights. Mesh is missing 'persistent_ids' data."); return
        if self.fixed_node_indices:
            mask = np.isin(id_map, list(self.fixed_node_indices))
            if np.any(mask): self.fixed_node_actor = self.plotter.add_points(highlight_mesh.points[mask], color='blue', point_size=10.0, render_points_as_spheres=True, pickable=False)
        if self.load_node_indices:
            mask = np.isin(id_map, list(self.load_node_indices))
            if np.any(mask): self.load_node_actor = self.plotter.add_points(highlight_mesh.points[mask], color='red', point_size=10.0, render_points_as_spheres=True, pickable=False)
        self.fixed_node_label.setText(f"{len(self.fixed_node_indices)} nodes")
        self.load_node_label.setText(f"{len(self.load_node_indices)} nodes")
        if render: self.plotter.render()

    def _clear_fea_selections(self):
        self.fixed_node_indices.clear(); self.load_node_indices.clear(); self._update_selection_highlight(); self.log("FEA selections cleared.")

    def run_simulation(self):
        if not self.volumetric_mesh: self.log("Create a volumetric mesh first.", "error"); return
        if not (self.fixed_node_indices or self.load_node_indices): self.log("Select boundaries (fixed or loaded nodes) first.", "error"); return
        if pv.CellType.QUADRATIC_TETRA in self.volumetric_mesh.celltypes: QMessageBox.warning(self, "Unsupported Element Type", "The native FEA solver currently supports linear tetrahedral elements only.\nPlease re-mesh with 'Mesh Order' set to 1."); self.log("Solver aborted: Native solver does not support quadratic elements.", "error"); return
        self.set_busy(True)
        try:
            if 'persistent_ids' not in self.volumetric_mesh.point_data: self.log("FATAL: Mesh is missing 'persistent_ids'. Cannot run simulation. Please regenerate the mesh.", "error"); self.set_busy(False); return
            params = {
                "mesh": self.volumetric_mesh.copy(), "material": MATERIALS[self.material_combo.currentText()], 
                "fixed_node_indices": list(self.fixed_node_indices), "loaded_node_indices": list(self.load_node_indices), 
                "force": (self.fx_spin.value(), self.fy_spin.value(), self.fz_spin.value()), "log_func": self.log, 
                "stress_percentile_threshold": 99.5, "progress_callback": self.update_progress_bar
            }
            self.fea_result_model = run_native_fea(**params)
            self.log("FEA simulation completed.", "success"); self.optim_group.setEnabled(True); self.update_view("FEA Result")
        except Exception as e: self.log(f"FEA Error: {e}", "error"); traceback.print_exc()
        finally: self.set_busy(False)

    def run_optimization(self):
        if not self.fea_result_model: QMessageBox.warning(self, "Prerequisite Missing", "You must run a standard FEA simulation first to provide an initial state for the optimization."); self.log("Optimization aborted: No initial FEA result found.", "error"); return
        if not self.original_pv_shell: QMessageBox.warning(self, "Prerequisite Missing", "The original CAD shell is required for the optimization loop. Please re-import your model."); self.log("Optimization aborted: Original CAD shell not in memory.", "error"); return
        self.set_busy(True)
        try:
            lattice_params = {'resolution': self.resolution_spin.value(), 'wx': self.unit_x_spin.value(), 'wy': self.unit_y_spin.value(), 'wz': self.unit_z_spin.value(),'lattice_type': self.lattice_type_box.currentText(), 'thickness': self.lattice_thickness_spin.value(),'solidify': self.solidify_checkbox.isChecked(), 'create_shell': self.shell_checkbox.isChecked(), 'shell_thickness': self.shell_thickness_spin.value()}
            meshing_params = {'detail_size': self.detail_size_spin.value(),'feature_angle': self.feature_angle_spin.value(),'volume_g_size': self.volume_g_size_spin.value(),'mesh_order': self.mesh_order_spin.value(),'optimize_ho': self.ho_optimize_check.isChecked(),'algorithm': self.mesh_algo_combo.currentText(),'skip_preprocessing': self.skip_preprocessing_check.isChecked()}
            fea_params = {"material": MATERIALS[self.material_combo.currentText()],"fixed_node_indices": list(self.fixed_node_indices),"loaded_node_indices": list(self.load_node_indices),"force": (self.fx_spin.value(), self.fy_spin.value(), self.fz_spin.value()),"log_func": self.log,"stress_percentile_threshold": 99.5,"progress_callback": self.update_progress_bar}
            optim_params = {'max_iterations': self.optim_max_iter_spin.value(),'stress_reduction_target': self.optim_stress_reduc_spin.value()}
            
            # --- MODIFIED: Unpack both the scalar field and the resulting mesh ---
            optimized_scalar, optimized_mesh = run_optimization_loop(
                initial_fea_mesh=self.fea_result_model, 
                original_shell=self.original_pv_shell, 
                lattice_params=lattice_params, 
                remesh_params=self.remesh_settings, 
                meshing_params=meshing_params, 
                fea_params=fea_params, 
                optim_params=optim_params, 
                log_func=self.log, 
                progress_callback=self.update_progress_bar
            )
            
            # Store the final mesh for viewing
            self.fea_result_model = optimized_mesh
            
            # --- MODIFIED: Store the optimal scalar field and update the UI ---
            self.external_scalar = optimized_scalar
            if optimized_mesh:
                self.optimized_surface_mesh = optimized_mesh.extract_surface().triangulate().clean()
                self.log("Extracted surface from optimized model for viewing.", "success")
            self.log("Optimization complete. Optimal scalar field is now active.", "success")
            self.use_scalar_for_cell_size_checkbox.setEnabled(True)
            self.use_scalar_checkbox.setEnabled(True)
            self.show_scalar_field_check.setEnabled(True)

            self.view_selector.setCurrentText("Optimized Result")
            self.update_view("Optimized Result")

        except Exception as e:
            self.log(f"Optimization Error: {e}", "error"); traceback.print_exc()
            QMessageBox.critical(self, "Optimization Failed", f"An error occurred during optimization:\n\n{e}")
        finally:
            self.set_busy(False)

    def _get_current_model_for_export(self):
        view_text = self.view_selector.currentText()
        if view_text == "Result": return self.surface_mesh
        if view_text == "Volumetric Mesh": return self.volumetric_mesh
        if view_text == "FEA Result": return self.fea_result_model
        if view_text == "Optimized Result": return self.optimized_surface_mesh
        return self.original_pv_shell

    def update_view(self, _=None):
        self.plotter.clear(); self.plotter.show_axes(); self.main_mesh_actor = None
        view_text = self.view_selector.currentText()
        self.plotter.clear_plane_widgets()

        mesh_to_display = self._get_current_model_for_export()
            
        is_fea_view = (view_text == "FEA Result")
        self.fea_result_selector.setVisible(is_fea_view)
        is_warped_view_active = is_fea_view and self.show_deformation_check.isChecked()
        self.deformation_scale_label.setVisible(is_warped_view_active)
        self.deformation_scale_spin.setVisible(is_warped_view_active)
        self.show_voxel_preview_check.setVisible(view_text == "CAD" and self.original_pv_shell is not None)
        # --- MANAGE VISIBILITY OF NEW WIDGET ---
        self.show_scalar_field_check.setVisible(view_text == "CAD" and self.external_scalar is not None)
        self.select_toggle_button.setEnabled(self.volumetric_mesh is not None)
        
        if not mesh_to_display: self.plotter.reset_camera(); return

        if is_warped_view_active and 'Displacements' in mesh_to_display.point_data:
            if np.linalg.norm(mesh_to_display.point_data['Displacements']) > 1e-9:
                scale_factor = self.deformation_scale_spin.value()
                scalar_to_show = self.fea_result_selector.currentText()
                mesh_kwargs = {'scalars': scalar_to_show, 'cmap': "jet", 'scalar_bar_args': {'title': scalar_to_show.replace("_", " ").title()}}
                self.plotter.add_mesh(mesh_to_display, style='wireframe', color='grey', opacity=0.5)
                warped_mesh = mesh_to_display.warp_by_vector('Displacements', factor=scale_factor)
                self.main_mesh_actor = self.plotter.add_mesh(warped_mesh, **mesh_kwargs)
            else:
                self.log("Deformation is zero or negligible. Showing undeformed result."); is_warped_view_active = False
        if not is_warped_view_active:
            mesh_kwargs = {}
            if view_text == "CAD": mesh_kwargs.update({'color': 'skyblue', 'style': 'wireframe', 'opacity': 0.5})
            elif is_fea_view: scalar_to_show = self.fea_result_selector.currentText(); mesh_kwargs.update({'scalars': scalar_to_show, 'cmap': "jet", 'scalar_bar_args': {'title': scalar_to_show.replace("_", " ").title()}})
            else: mesh_kwargs.update({'color': 'orange', 'show_edges': True})
            if self.clipping_settings.get("enabled", False): self.main_mesh_actor = self.plotter.add_mesh_clip_plane(mesh_to_display, invert=self.clipping_settings.get("invert", False), **mesh_kwargs)
            else: self.main_mesh_actor = self.plotter.add_mesh(mesh_to_display, **mesh_kwargs)

        if view_text == "CAD" and self.show_voxel_preview_check.isChecked():
            try:
                density = [(b - a) / self.detail_size_spin.value() for a, b in zip(mesh_to_display.bounds[::2], mesh_to_display.bounds[1::2])]
                self.plotter.add_mesh(pv.voxelize(mesh_to_display, density=density), style='surface', color='tan', opacity=0.7)
            except Exception as e: self.log(f"Could not generate voxel preview: {e}", "error")
        
        # --- NEW LOGIC TO DISPLAY SCALAR FIELD ---
        if view_text == "CAD" and self.show_scalar_field_check.isChecked() and self.external_scalar is not None:
            points, values = self.external_scalar
            scalar_cloud = pv.PolyData(points)
            scalar_cloud['values'] = values
            self.plotter.add_points(
                scalar_cloud,
                render_points_as_spheres=True,
                point_size=10,
                scalars='values',
                cmap='viridis',
                scalar_bar_args={'title': 'Scalar Field'}
            )

        if self.fixed_node_actor: self.plotter.add_actor(self.fixed_node_actor)
        if self.load_node_actor: self.plotter.add_actor(self.load_node_actor)
        self.plotter.reset_camera()
        
    def log(self, message, level="info", percent=None):
        prefix = f"[{level.upper()}]"
        self.log_output.append(f"{prefix}: {message}")
        if percent is not None:
            self.progress_bar.setValue(percent)
        QApplication.processEvents()

    def update_progress_bar(self, value, message=None):
        self.progress_bar.setValue(value)
        if message: self.status_bar.showMessage(message)

    def set_busy(self, is_busy):
        self.progress_bar.setRange(0, 100 if is_busy else 0)
        self.progress_bar.setVisible(is_busy)
        QApplication.setOverrideCursor(Qt.WaitCursor if is_busy else Qt.ArrowCursor)

    def export_current_model(self):
        filters = "STL Files (*.stl);;STEP Files (*.step *.stp);;OBJ Files (*.obj);;ANSYS VTK File (*.vtk)"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Model", "", filters)
        self.export_dialog_path = file_path
        if not file_path: return
        try:
            model_to_export = self.volumetric_mesh if file_path.endswith('.vtk') else self._get_current_model_for_export()
            if not model_to_export: self.log("No model to export for the current view.", "error"); return
            self.log(f"Exporting model to {file_path}...")
            QApplication.processEvents()
            export_model(model_to_export, file_path)
            self.log(f"Successfully exported model to {file_path}", "success")
        except Exception as e: self.log(f"Export error: {e}", "error")

    def save_volumetric_mesh(self):
        if not self.volumetric_mesh: self.log("No volumetric mesh exists to save.", "error"); QMessageBox.warning(self, "Save Error", "Please generate or load a volumetric mesh first."); return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Volumetric Mesh", "", "VTK Unstructured Grid (*.vtk)")
        if not file_path: return
        try:
            self.log(f"Saving volumetric mesh to {file_path}...")
            QApplication.processEvents()
            export_model(self.volumetric_mesh, file_path)
            self.log(f"Successfully saved volumetric mesh to {file_path}", "success")
        except Exception as e: self.log(f"Save error: {e}", "error"); QMessageBox.critical(self, "Save Error", f"Failed to save the volumetric mesh.\n\nDetails:\n{e}")

    def load_volumetric_mesh(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Volumetric Mesh", "", "VTK Unstructured Grid (*.vtk)")
        if not file_path: return
        try:
            self.log(f"Loading volumetric mesh from {file_path}...")
            QApplication.processEvents()
            mesh = pv.read(file_path)
            if not isinstance(mesh, pv.UnstructuredGrid): self.log("Error: Loaded file is not a volumetric mesh (UnstructuredGrid).", "error"); QMessageBox.critical(self, "Load Error", "The selected file is not a valid volumetric mesh."); return
            self.original_pv_shell, self.surface_mesh, self.fea_result_model = None, None, None
            self._clear_fea_selections()
            self.volumetric_mesh = mesh
            self.log("Successfully loaded volumetric mesh.", "success")
            if 'persistent_ids' not in self.volumetric_mesh.point_data:
                self.log("Warning: Loaded mesh has no persistent IDs. Creating them now.")
                self.volumetric_mesh.point_data['persistent_ids'] = np.arange(self.volumetric_mesh.n_points)
            self.fea_group.setEnabled(True)
            self.optim_group.setEnabled(False)
            self.update_view("Volumetric Mesh")
        except Exception as e:
            self.log(f"Load error: {e}", "error")
            QMessageBox.critical(self, "Load Error", f"Failed to load the volumetric mesh.\n\nDetails:\n{e}")

    def _create_hbox(self, *widgets):
        layout = QHBoxLayout()
        for w in widgets: layout.addWidget(w)
        return layout
    
    def _get_model_bounds_for_refinement(self):
        model_for_bounds = self.surface_mesh if self.surface_mesh is not None else self.original_pv_shell
        if model_for_bounds is None: self.log("Error: A model must be loaded or generated to get its bounds.", "error"); QMessageBox.warning(self, "No Model", "A model must be loaded to get its bounds."); return
        bounds = model_for_bounds.bounds
        self.ref_xmin_spin.setValue(bounds[0]); self.ref_xmax_spin.setValue(bounds[1])
        self.ref_ymin_spin.setValue(bounds[2]); self.ref_ymax_spin.setValue(bounds[3])
        self.ref_zmin_spin.setValue(bounds[4]); self.ref_zmax_spin.setValue(bounds[5])
        self.log("Set refinement region to current model bounds.")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- ADD THIS SECTION TO LOAD THE STYLESHEET ---
    try:
        with open('stylesheet.qss', 'r') as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: stylesheet.qss not found. Using default style.")
    # --- END OF ADDED SECTION ---

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())