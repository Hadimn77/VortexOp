# main_window.py
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget,
    QApplication, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QStatusBar,
    QTextEdit, QToolBar, QAction, QMenuBar, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QStyle, QHBoxLayout, QFrame, QMessageBox, QScrollArea, QRadioButton, QButtonGroup,
    QDockWidget, QSizePolicy, QSlider, QToolButton, QMenu, QActionGroup
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
import numpy as np
import pyvista as pv
import os
from pyvistaqt import QtInteractor
import vtk
import json
import zipfile
import shutil
import tempfile
import traceback

from cad_importer import DirectCADImporter
from lattice_utils import generate_infill_inside
from fea_utils import MATERIALS, create_robust_volumetric_mesh, create_hexahedral_mesh, check_mesh_quality
# Ensure your optimizer file is named 'lattice_optimizer.py'
from lattice_optimizer import run_optimization_loop
from export_utils import export_model
from unit_utils import UnitManager, SYSTEMS
from fea_solver_core import run_native_fea

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
    # This class remains unchanged
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
        self.hole_size_spin = QDoubleSpinBox(); self.hole_size_spin.setRange(0.1, 1000.0); self.hole_size_spin.setValue(0.1)
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
    # This class remains unchanged
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

class NumpyAwareJSONEncoder(json.JSONEncoder):
    # This class remains unchanged
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyAwareJSONEncoder, self).default(obj)
    
class LatticeMakerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LatticeMaker")
        self.setWindowIcon(QIcon('latticemaker_logo.ico'))
        self.setGeometry(100, 100, 1400, 900)
        self.importer = DirectCADImporter()
        self.unit_manager = UnitManager() 
        self.original_pv_shell = None
        self.surface_mesh = None
        self.volumetric_mesh = None
        self.fea_result_model = None
        self.external_scalar = None
        self.lattice_flag = False
        self.fixed_node_indices = set()
        self.load_node_indices = set()
        self.fixed_node_actor = None
        self.load_node_actor = None
        self.main_mesh_actor = None
        self.selection_surface = None
        # MODIFICATION: This dictionary will store the loaded mesh objects for each iteration
        self.optimization_results = {}
        self.remesh_settings = {
            "remesh_enabled": True, 
            "smoothing": "Taubin", 
            "smoothing_iterations": 10, 
            "repair_methods": {}
        }
        self.clipping_settings = {"enabled": False, "invert": False}
        self._is_box_selection_mode = False
        self.use_fea_stress_button = None
        self._create_widgets()
        self._unit_widgets = {
            'length': [
                self.unit_x_spin, self.unit_y_spin, self.unit_z_spin,
                self.lattice_thickness_spin, self.shell_thickness_spin,
                self.detail_size_spin, self.volume_g_size_spin,
                self.optim_min_thickness_spin, self.optim_max_thickness_spin,
                self.optim_disp_limit_spin
            ],
            'pressure': [
                self.optim_stress_limit_spin
            ],
            'force': [
                self.fx_spin, self.fy_spin, self.fz_spin
            ]
        }
        pv.set_plot_theme('dark')
        self.plotter.set_background('#2d2d2d')
        self._create_layouts()
        self._create_menu_bar()
        self._setup_ui() 
        self._connect_signals()
        
        self.setCentralWidget(self.main_container)
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.log_output.setReadOnly(True)
        
        self.status_bar.addPermanentWidget(self.unit_button)
        
        # Initial UI State
        self.fea_group.setEnabled(False)
        self.optim_group.setEnabled(False)
        self._check_dependencies()
        self._on_shell_toggle(False)
        self.use_scalar_for_cell_size_checkbox.setEnabled(False)
        self.use_scalar_checkbox.setEnabled(False)
        self.show_voxel_preview_check.setEnabled(False)
        self.show_scalar_field_check.setEnabled(False)
        self._on_element_type_change()
        self._update_thickness_limit()
        self.optim_iteration_selector.setVisible(False)
        self.optim_iteration_selector_label.setVisible(False)
        self._show_toolbox('lattice')
        self._update_ui_for_units() 

    def _check_dependencies(self):
        # This method remains unchanged
        if not GMSH_AVAILABLE or not PYMESHFIX_AVAILABLE:
            self.log("WARNING: 'gmsh' and/or 'pymeshfix' not found.")
            self.generate_tet_mesh_button.setEnabled(False)
            self.generate_tet_mesh_button.setToolTip("Install with: pip install gmsh pymeshfix")

    def _create_widgets(self):
        # --- Central Plotter and Logs ---
        self.plotter = QtInteractor(self)
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.status_bar = QStatusBar(); self.log_output = QTextEdit()

        # --- Toolboxes (as GroupBoxes) ---
        self.lattice_group = QGroupBox("Lattice Generation")
        self.fea_group = QGroupBox("FEA Toolbox")
        self.optim_group = QGroupBox("Lattice Optimization")
        self.view_group = QGroupBox("View Controls")

        # --- Lattice Group Widgets ---
        self.lattice_type_box = QComboBox(); self.lattice_type_box.addItems(['gyroid', 'diamond', 'neovius'])
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
        self.infill_button.setObjectName("infill_button")

        # --- FEA Group Widgets ---
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
        self.run_fea_button.setObjectName("run_fea_button")

        # --- Optimization Group Widgets ---
        self.optim_max_iter_spin = QSpinBox(); self.optim_max_iter_spin.setRange(1, 50); self.optim_max_iter_spin.setValue(5)
        self.optim_min_thickness_spin = QDoubleSpinBox(); self.optim_min_thickness_spin.setRange(0.01, 10.0); self.optim_min_thickness_spin.setValue(0.5); self.optim_min_thickness_spin.setDecimals(2)
        self.optim_max_thickness_spin = QDoubleSpinBox(); self.optim_max_thickness_spin.setRange(0.1, 20.0); self.optim_max_thickness_spin.setValue(2.0); self.optim_max_thickness_spin.setDecimals(2)
        self.run_optimization_button = QPushButton("Run Optimization"); self.run_optimization_button.setIcon(QApplication.style().standardIcon(QStyle.SP_CommandLink))
        self.run_optimization_button.setObjectName("run_optimization_button")
        self.optim_objective_combo = QComboBox(); self.optim_objective_combo.addItems(["Minimize Max Stress", "Minimize Max Displacement"])
        self.mass_reduction_slider = QSlider(Qt.Horizontal)
        self.mass_reduction_slider.setRange(0, 100)
        self.mass_reduction_slider.setValue(50)
        self.mass_reduction_label = QLabel("50%")
        self.optim_stress_limit_spin = QDoubleSpinBox(); self.optim_stress_limit_spin.setRange(1, 1e6); self.optim_stress_limit_spin.setDecimals(0)
        self.optim_disp_limit_spin = QDoubleSpinBox(); self.optim_disp_limit_spin.setRange(0.01, 1000.0); self.optim_disp_limit_spin.setDecimals(2)
        self.use_fea_stress_button = QPushButton("Use FEA Stress as Scalar Field")

        # --- View Group Widgets ---
        self.view_selector = QComboBox(); self.view_selector.addItems(["CAD", "Result", "Volumetric Mesh", "FEA Result", "Optimized Result"])
        self.fea_result_selector = QComboBox(); self.fea_result_selector.addItems(["von_mises_stress", "displacement", "principal_s1", "principal_s2", "principal_s3"]); self.fea_result_selector.setVisible(False)
        self.clipping_plane_button = QPushButton("Clipping Tool Settings")
        self.show_voxel_preview_check = QCheckBox("Show Voxel Preview")
        self.show_scalar_field_check = QCheckBox("Show Scalar Field")
        self.show_deformation_check = QCheckBox("Show Deformed Shape")
        self.deformation_scale_spin = QDoubleSpinBox(); self.deformation_scale_spin.setRange(0, 1e6); self.deformation_scale_spin.setValue(100.0); self.deformation_scale_spin.setSingleStep(10); self.deformation_scale_spin.setDecimals(1)
        self.deformation_scale_label = QLabel("Deformation Scale:")
        # MODIFICATION: Add checkbox for viewing FEA results in optimization view
        self.show_optim_fea_check = QCheckBox("Show FEA Result")
        
        self.optim_iteration_selector_label = QLabel("Iteration:")
        self.optim_iteration_selector = QComboBox()

        self.unit_button = QToolButton()
        self.unit_button.setText(self.unit_manager.system_name)
        self.unit_button.setPopupMode(QToolButton.InstantPopup)
        self.unit_button.setIcon(QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView)) 
        unit_menu = QMenu(self)
        self.unit_action_group = QActionGroup(self)
        self.unit_action_group.setExclusive(True)
        for system_name in SYSTEMS.keys():
            action = QAction(system_name, self, checkable=True)
            action.setData(system_name)
            unit_menu.addAction(action)
            self.unit_action_group.addAction(action)
        self.unit_button.setMenu(unit_menu)
        default_action = next((act for act in unit_menu.actions() if act.data() == self.unit_manager.system_name), None)
        if default_action:
            default_action.setChecked(True)

    def _setup_ui(self):
        # This method remains unchanged
        self.ribbon = QToolBar("Ribbon")
        self.ribbon.setIconSize(QSize(48, 48))
        self.ribbon.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.ribbon)

        style = QApplication.style()
        self.ribbon.addAction(self._create_ribbon_action("Lattice", style.standardIcon(QStyle.SP_FileDialogDetailedView), 'lattice'))
        self.ribbon.addAction(self._create_ribbon_action("FEA", style.standardIcon(QStyle.SP_ComputerIcon), 'fea'))
        self.ribbon.addAction(self._create_ribbon_action("Optimize", style.standardIcon(QStyle.SP_CommandLink), 'optim'))
        self.ribbon.addAction(self._create_ribbon_action("View", style.standardIcon(QStyle.SP_DesktopIcon), 'view'))

        self.settings_dock = QDockWidget("Toolbox Settings", self)
        self.settings_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.settings_dock)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.settings_container = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.settings_container)
        self.settings_dock.setWidget(scroll_area)
        
        self._populate_lattice_group()
        self._populate_fea_group()
        self._populate_optim_group()
        self._populate_view_group()
        
        self.toolboxes = {
            'lattice': self.lattice_group,
            'fea': self.fea_group,
            'optim': self.optim_group,
            'view': self.view_group
        }
        for box in self.toolboxes.values():
            self.settings_layout.addWidget(box)

    def _create_ribbon_action(self, text, icon, key):
        # This method remains unchanged
        action = QAction(icon, text, self)
        action.triggered.connect(lambda: self._show_toolbox(key))
        return action

    def _show_toolbox(self, key):
        # This method remains unchanged
        if not hasattr(self, 'toolboxes'):
            return
            
        for toolbox_key, widget in self.toolboxes.items():
            widget.setVisible(toolbox_key == key)

        self.settings_dock.setWindowTitle(f"{key.capitalize()} Toolbox Settings")
        self.settings_dock.show()
        
        QApplication.processEvents()

        ideal_width = self.settings_container.sizeHint().width() + 25 
        self.settings_dock.setFixedWidth(ideal_width)

    def _populate_lattice_group(self):
        # This method remains unchanged
        lattice_layout = QFormLayout(self.lattice_group)
        lattice_layout.addRow("Type:", self.lattice_type_box)
        resolution_layout = QHBoxLayout(); resolution_layout.addWidget(self.resolution_spin); resolution_layout.addWidget(self.suggest_resolution_button)
        lattice_layout.addRow("Resolution:", resolution_layout)
        lattice_layout.addRow("Cell X:", self.unit_x_spin); lattice_layout.addRow("Cell Y:", self.unit_y_spin); lattice_layout.addRow("Cell Z:", self.unit_z_spin)
        lattice_layout.addRow(self.solidify_checkbox); lattice_layout.addRow("Lattice Thickness:", self.lattice_thickness_spin)
        lattice_layout.addRow(self.shell_checkbox); self.shell_thickness_label = QLabel("Shell Thickness:"); lattice_layout.addRow(self.shell_thickness_label, self.shell_thickness_spin)
        lattice_layout.addRow(self.use_scalar_for_cell_size_checkbox); lattice_layout.addRow(self.use_scalar_checkbox); lattice_layout.addRow(self.scalar_button)
        lattice_layout.addRow(self.remesh_button); lattice_layout.addRow(self.infill_button)

    def _populate_fea_group(self):
        # This method remains unchanged
        fea_layout = QFormLayout(self.fea_group)
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
        self.force_label = QLabel("Force (N):") 
        self.force_label.setObjectName("force_label")
        force_layout = QHBoxLayout(); force_layout.addWidget(QLabel("Fx")); force_layout.addWidget(self.fx_spin); force_layout.addWidget(QLabel("Fy")); force_layout.addWidget(self.fy_spin); force_layout.addWidget(QLabel("Fz")); force_layout.addWidget(self.fz_spin)
        fea_layout.addRow(self.force_label, force_layout)
        fea_layout.addRow(self.select_toggle_button)
        selection_type_layout = QHBoxLayout(); selection_type_layout.addWidget(self.fixed_bc_radio); selection_type_layout.addWidget(self.load_bc_radio)
        fea_layout.addRow(selection_type_layout)
        fea_layout.addRow(QLabel("Fixed Nodes:"), self.fixed_node_label)
        fea_layout.addRow(QLabel("Load Nodes:"), self.load_node_label)
        fea_layout.addRow(self.clear_fea_button); fea_layout.addRow(self.run_fea_button)

    def _populate_optim_group(self):
        # This method remains unchanged
        optim_layout = QFormLayout(self.optim_group)
        optim_layout.addRow("Objective:", self.optim_objective_combo)
        
        mass_reduction_layout = QHBoxLayout()
        mass_reduction_layout.addWidget(self.mass_reduction_slider)
        mass_reduction_layout.addWidget(self.mass_reduction_label)
        optim_layout.addRow("Mass Reduction Priority:", mass_reduction_layout)

        optim_layout.addRow("Max Allowable Stress:", self.optim_stress_limit_spin)
        optim_layout.addRow("Max Allowable Displacement:", self.optim_disp_limit_spin)
        optim_layout.addRow("Min Thickness:", self.optim_min_thickness_spin)
        optim_layout.addRow("Max Thickness:", self.optim_max_thickness_spin)
        optim_layout.addRow("Max Iterations:", self.optim_max_iter_spin)
        optim_layout.addRow(self.use_fea_stress_button)
        optim_layout.addRow(self.run_optimization_button)

    def _populate_view_group(self):
        # MODIFICATION: Add the new checkbox
        view_layout = QFormLayout(self.view_group)
        view_layout.addRow(self.show_deformation_check)
        view_layout.addRow(self.deformation_scale_label, self.deformation_scale_spin)
        view_layout.addRow("View Mode:", self.view_selector)
        view_layout.addRow(self.optim_iteration_selector_label, self.optim_iteration_selector)
        view_layout.addRow(self.show_optim_fea_check)
        view_layout.addRow("FEA Scalar:", self.fea_result_selector)
        view_layout.addRow(self.show_scalar_field_check)
        view_layout.addRow(self.show_voxel_preview_check)
        view_layout.addRow(self.clipping_plane_button)

    def _create_layouts(self):
        # This method remains unchanged
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.plotter.interactor, 5) 
        main_layout.addWidget(self.log_output, 1)    
        self.plotter.interactor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_container = QWidget()
        self.main_container.setLayout(main_layout)

    def _create_menu_bar(self):
        # This method remains unchanged
        menubar = self.menuBar(); file_menu = menubar.addMenu("File")
        self.open_stl_action = QAction("Open Model", self); file_menu.addAction(self.open_stl_action); 
        self.load_vol_action = QAction("Load Volumetric Mesh", self); file_menu.addAction(self.load_vol_action)
        self.save_vol_action = QAction("Save Volumetric Mesh", self); file_menu.addAction(self.save_vol_action)
        self.export_action = QAction("Export Model", self); file_menu.addAction(self.export_action)
        self.save_project_action = QAction("Save Project", self); file_menu.addAction(self.save_project_action)
        self.load_project_action = QAction("Load Project", self); file_menu.addAction(self.load_project_action)
        file_menu.addSeparator()
        
    def _connect_signals(self):
        # This method remains unchanged
        self.open_stl_action.triggered.connect(self.import_file); self.save_project_action.triggered.connect(self._save_project)
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
        self.load_project_action.triggered.connect(self._load_project)
        self.generate_tet_mesh_button.clicked.connect(self.run_robust_tet_meshing)
        self.generate_hex_mesh_button.clicked.connect(self.run_hex_meshing)
        self.run_fea_button.clicked.connect(self.run_simulation)
        self.show_voxel_preview_check.toggled.connect(self.update_view)
        self.show_scalar_field_check.toggled.connect(self.update_view)
        self.clipping_plane_button.clicked.connect(self.open_clipping_dialog)
        self.tet_radio.toggled.connect(self._on_element_type_change)
        self.mesh_order_spin.valueChanged.connect(self._on_order_change)
        self.get_bounds_button.clicked.connect(self._get_model_bounds_for_refinement)
        self.show_deformation_check.toggled.connect(self.update_view)
        self.deformation_scale_spin.valueChanged.connect(self.update_view)
        self.run_optimization_button.clicked.connect(self.run_optimization)
        self.suggest_resolution_button.clicked.connect(self._suggest_resolution)
        self.optim_iteration_selector.currentIndexChanged.connect(self.update_view)
        for spin_box in [self.unit_x_spin, self.unit_y_spin, self.unit_z_spin]:
            spin_box.valueChanged.connect(self._update_thickness_limit)
        self.use_fea_stress_button.clicked.connect(self._set_fea_stress_as_scalar)
        
        self.mass_reduction_slider.valueChanged.connect(self._update_mass_reduction_label)
        self.show_optim_fea_check.toggled.connect(self.update_view)
        self.unit_action_group.triggered.connect(self._set_unit_system)

    def _set_unit_system(self, action):
        # This method remains unchanged
        old_system_name = self.unit_manager.system_name
        new_system_name = action.data()

        if old_system_name == new_system_name:
            return 

        old_values = {}
        for unit_type, widgets in self._unit_widgets.items():
            for widget in widgets:
                old_values[widget] = (widget.value(), unit_type)

        self.unit_manager.set_system(new_system_name)
        self.unit_button.setText(new_system_name)
        self.log(f"Unit system changed from '{old_system_name}' to '{new_system_name}'")

        old_unit_manager = UnitManager(old_system_name)

        for widget, (value, unit_type) in old_values.items():
            value_si = old_unit_manager.convert_to_solver(value, unit_type)
            new_value_ui = self.unit_manager.convert_from_solver(value_si, unit_type)
            
            widget.blockSignals(True)
            widget.setValue(new_value_ui)
            widget.blockSignals(False)

        self._update_ui_for_units()

    def _update_ui_for_units(self):
        # This method remains unchanged
        len_unit = self.unit_manager.get_ui_label('length')
        force_unit = self.unit_manager.get_ui_label('force')
        pressure_unit = self.unit_manager.get_ui_label('pressure')

        for spin in [self.unit_x_spin, self.unit_y_spin, self.unit_z_spin,
                     self.lattice_thickness_spin, self.shell_thickness_spin,
                     self.detail_size_spin, self.volume_g_size_spin,
                     self.optim_min_thickness_spin, self.optim_max_thickness_spin]:
            spin.setSuffix(f" {len_unit}")
        
        if hasattr(self, 'force_label'):
            self.force_label.setText(f"Force ({force_unit}):")
        
        self.optim_disp_limit_spin.setSuffix(f" {len_unit}")
        self.optim_stress_limit_spin.setSuffix(f" {pressure_unit}")
        
        self.update_view()

    def _update_mass_reduction_label(self, value):
        # This method remains unchanged
        self.mass_reduction_label.setText(f"{value}%")

    def run_optimization(self):
        if not self.fea_result_model: QMessageBox.warning(self, "Prerequisite Missing", "You must run a standard FEA simulation first to provide an initial state for the optimization."); self.log("Optimization aborted: No initial FEA result found.", "error"); return
        if not self.original_pv_shell: QMessageBox.warning(self, "Prerequisite Missing", "The original CAD shell is required for the optimization loop. Please re-import your model."); self.log("Optimization aborted: Original CAD shell not in memory.", "error"); return
        self.set_busy(True)
        try:
            self.optimization_results = {}
            
            lattice_params = {'resolution': self.resolution_spin.value(), 'wx': self.unit_x_spin.value(), 'wy': self.unit_y_spin.value(), 'wz': self.unit_z_spin.value(),'lattice_type': self.lattice_type_box.currentText(), 'thickness': self.lattice_thickness_spin.value(),'solidify': self.solidify_checkbox.isChecked(), 'create_shell': self.shell_checkbox.isChecked(), 'shell_thickness': self.shell_thickness_spin.value()}
            meshing_params = {'detail_size': self.detail_size_spin.value(),'feature_angle': self.feature_angle_spin.value(),'volume_g_size': self.volume_g_size_spin.value(),'mesh_order': self.mesh_order_spin.value(),'optimize_ho': self.ho_optimize_check.isChecked(),'algorithm': self.mesh_algo_combo.currentText(),'skip_preprocessing': self.skip_preprocessing_check.isChecked(), 'lattice_model': True}
            
            fea_params = {"material": MATERIALS[self.material_combo.currentText()],"fixed_node_indices": list(self.fixed_node_indices),"loaded_node_indices": list(self.load_node_indices),"force": (self.fx_spin.value(), self.fy_spin.value(), self.fz_spin.value()),"log_func": self.log,"stress_percentile_threshold": 99.5,"progress_callback": self.update_progress_bar}
            
            optim_params = {
                'max_iterations': self.optim_max_iter_spin.value(),
                'objective': self.optim_objective_combo.currentText(),
                'mass_reduction_ratio': self.mass_reduction_slider.value() / 100.0,
                'stress_limit': self.optim_stress_limit_spin.value(),
                'disp_limit': self.optim_disp_limit_spin.value(),
                'min_thickness': self.optim_min_thickness_spin.value(),
                'max_thickness': self.optim_max_thickness_spin.value()
            }
            
            # The optimizer returns a dictionary of file paths for each iteration
            optimized_scalar, optimized_mesh, iteration_data = run_optimization_loop(
                initial_fea_mesh=self.fea_result_model, 
                original_shell=self.original_pv_shell, 
                lattice_params=lattice_params, 
                remesh_params=self.remesh_settings, 
                meshing_params=meshing_params, 
                fea_params=fea_params, 
                optim_params=optim_params,
                unit_manager=self.unit_manager,
                log_func=self.log, 
                progress_callback=self.update_progress_bar
            )
            
            # The best result mesh is stored directly
            self.fea_result_model = optimized_mesh
            self.external_scalar = optimized_scalar
            
            self.optimization_results = {}
            if iteration_data:
                self.log("Loading optimization iteration results into memory...")
                for i, data in iteration_data.items():
                    try:
                        # Load the lattice and FEA result meshes from the saved files
                        surface_mesh = pv.read(data['lattice_path'])
                        fea_result_mesh = pv.read(data['fea_result_path'])
                        self.optimization_results[i] = {
                            'surface_mesh': surface_mesh,
                            'fea_result': fea_result_mesh
                        }
                    except Exception as e:
                        self.log(f"Could not load result files for iteration {i}. Paths: {data}. Error: {e}", "warning")

            self.log("Optimization complete. Optimal scalar field is now active.", "success")
            self.use_scalar_for_cell_size_checkbox.setEnabled(True)
            self.use_scalar_checkbox.setEnabled(True)
            self.show_scalar_field_check.setEnabled(True)

            self._populate_iteration_selector() 
            self.view_selector.setCurrentText("Optimized Result")

        except Exception as e:
            self.log(f"Optimization Error: {e}", "error"); traceback.print_exc()
            QMessageBox.critical(self, "Optimization Failed", f"An error occurred during optimization:\n\n{e}")
        finally:
            self.set_busy(False)

    def _save_project(self):
        if not self.original_pv_shell or not hasattr(self, 'original_model_path'):
            self.log("Cannot save project: No model is loaded.", "error")
            QMessageBox.warning(self, "Save Error", "Please load a model before saving a project.")
            return
        base_name = os.path.splitext(os.path.basename(self.original_model_path))[0]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project File", f"{base_name}.lmproj", "LatticeMaker Projects (*.lmproj)"
        )
        if not file_path:
            return
        self.log(f"Saving project to {file_path}...")
        self.set_busy(True)
        temp_dir = tempfile.mkdtemp()
        
        try:
            original_model_filename = os.path.basename(self.original_model_path)
            project_data = {
                "version": "2.3",
                "unit_system": self.unit_manager.system_name,
                "original_model_filename": original_model_filename,
                "active_view": self.view_selector.currentText(),
                "has_volumetric_mesh": self.volumetric_mesh is not None,
                "has_fea_result": self.fea_result_model is not None,
                "has_external_scalar": self.external_scalar is not None,
                "has_optimization_results": bool(self.optimization_results),
                "lattice_params": {
                    "type": self.lattice_type_box.currentText(), "resolution": self.resolution_spin.value(),
                    "cell_x": self.unit_x_spin.value(), "cell_y": self.unit_y_spin.value(), "cell_z": self.unit_z_spin.value(),
                    "solidify": self.solidify_checkbox.isChecked(), "thickness": self.lattice_thickness_spin.value(),
                    "create_shell": self.shell_checkbox.isChecked(), "shell_thickness": self.shell_thickness_spin.value(),
                },
                "fea_params": {
                    "material": self.material_combo.currentText(), "force_x": self.fx_spin.value(), "force_y": self.fy_spin.value(),
                    "force_z": self.fz_spin.value(), "fixed_node_indices": list(self.fixed_node_indices), "load_node_indices": list(self.load_node_indices)
                },
                "meshing_params": { "element_type": "Tetrahedral" if self.tet_radio.isChecked() else "Hexahedral", "detail_size": self.detail_size_spin.value(),
                    "volume_g_size": self.volume_g_size_spin.value(), "feature_angle": self.feature_angle_spin.value(),
                    "algorithm": self.mesh_algo_combo.currentText(), "mesh_order": self.mesh_order_spin.value(),
                },
                "optim_params": {
                    "max_iterations": self.optim_max_iter_spin.value(),
                    "objective": self.optim_objective_combo.currentText(),
                    "mass_reduction_ratio": self.mass_reduction_slider.value(),
                    "stress_limit": self.optim_stress_limit_spin.value(),
                    "disp_limit": self.optim_disp_limit_spin.value(),
                    "min_thickness": self.optim_min_thickness_spin.value(),
                    "max_thickness": self.optim_max_thickness_spin.value()
                }
            }
            with open(os.path.join(temp_dir, 'project.json'), 'w') as f:
                json.dump(project_data, f, indent=4, cls=NumpyAwareJSONEncoder)

            shutil.copy(self.original_model_path, os.path.join(temp_dir, original_model_filename))
            if self.volumetric_mesh:
                export_model(self.volumetric_mesh, os.path.join(temp_dir, 'volumetric_mesh.vtk'))
            if self.fea_result_model:
                export_model(self.fea_result_model, os.path.join(temp_dir, 'fea_result.vtk'))
            if self.external_scalar:
                points, values = self.external_scalar
                scalar_data = np.hstack([points, values[:, np.newaxis]])
                np.savetxt(os.path.join(temp_dir, 'scalar_field.txt'), scalar_data, header='X Y Z Value')
            
            # MODIFICATION: Save all optimization iteration data from memory
            if self.optimization_results:
                opt_dir = os.path.join(temp_dir, 'opt_results')
                os.makedirs(opt_dir)
                for i, data in self.optimization_results.items():
                    data['surface_mesh'].save(os.path.join(opt_dir, f'iter_{i}_lattice.stl'))
                    data['fea_result'].save(os.path.join(opt_dir, f'iter_{i}_fea.vtk'))

            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
            self.log(f"Project successfully saved.", "success")

        except Exception as e:
            self.log(f"Failed to save project: {e}", "error"); traceback.print_exc()
            QMessageBox.critical(self, "Save Failed", f"An error occurred while saving the project:\n\n{e}")
        finally:
            shutil.rmtree(temp_dir) 
            self.set_busy(False)

    def _load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Project File", "", "LatticeMaker Projects (*.lmproj)"
        )
        if not file_path: return
        self.log(f"Loading project from {file_path}...")
        self.set_busy(True)
        temp_dir = tempfile.mkdtemp()

        try:
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            with open(os.path.join(temp_dir, 'project.json'), 'r') as f:
                project_data = json.load(f)

            loaded_unit_system = project_data.get("unit_system", "Metric (mm, N, MPa)")
            action_to_set = next((act for act in self.unit_action_group.actions() if act.data() == loaded_unit_system), None)
            if action_to_set:
                action_to_set.setChecked(True)
                self._set_unit_system(action_to_set)

            model_filename = project_data.get("original_model_filename")
            extracted_model_path = os.path.join(temp_dir, model_filename)
            if not os.path.exists(extracted_model_path): raise FileNotFoundError(f"Original model '{model_filename}' not found inside the project file.")

            # --- MODIFICATION START ---
            # Create a new, persistent temporary file to store the model for the session.
            # This file will not be deleted when the extraction directory is cleaned up.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_filename)[1]) as temp_f:
                persistent_model_path = temp_f.name
            shutil.copy(extracted_model_path, persistent_model_path)

            # Use the new persistent path to load the model and set the instance variable
            self.import_file(file_path=persistent_model_path)
            self.original_model_path = persistent_model_path
            # --- MODIFICATION END ---

            lp = project_data.get("lattice_params", {}); self.lattice_type_box.setCurrentText(lp.get("type", "gyroid")); self.resolution_spin.setValue(lp.get("resolution", 100)); self.unit_x_spin.setValue(lp.get("cell_x", 10.0)); self.unit_y_spin.setValue(lp.get("cell_y", 10.0)); self.unit_z_spin.setValue(lp.get("cell_z", 10.0)); self.solidify_checkbox.setChecked(lp.get("solidify", False)); self.lattice_thickness_spin.setValue(lp.get("thickness", 1.0)); self.shell_checkbox.setChecked(lp.get("create_shell", False)); self.shell_thickness_spin.setValue(lp.get("shell_thickness", 1.0))
            fp = project_data.get("fea_params", {}); self.material_combo.setCurrentText(fp.get("material", "Titanium (Ti-6Al-4V)")); self.fx_spin.setValue(fp.get("force_x", 0.0)); self.fy_spin.setValue(fp.get("force_y", 0.0)); self.fz_spin.setValue(fp.get("force_z", -1000.0)); self.fixed_node_indices = set(fp.get("fixed_node_indices", [])); self.load_node_indices = set(fp.get("load_node_indices", []))
            mp = project_data.get("meshing_params", {}); self.hex_radio.setChecked(True) if mp.get("element_type") == "Hexahedral" else self.tet_radio.setChecked(True); self.detail_size_spin.setValue(mp.get("detail_size", 1.0)); self.volume_g_size_spin.setValue(mp.get("volume_g_size", 0.0)); self.feature_angle_spin.setValue(mp.get("feature_angle", 30.0)); self.mesh_algo_combo.setCurrentText(mp.get("algorithm", "HXT")); self.mesh_order_spin.setValue(mp.get("mesh_order", 1))
            op = project_data.get("optim_params", {}); self.optim_max_iter_spin.setValue(op.get("max_iterations", 5)); self.optim_objective_combo.setCurrentText(op.get("objective", "Minimize Max Stress")); self.optim_stress_limit_spin.setValue(op.get("stress_limit", 1e12)); self.optim_disp_limit_spin.setValue(op.get("disp_limit", 1000.0)); self.optim_min_thickness_spin.setValue(op.get("min_thickness", 0.2)); self.optim_max_thickness_spin.setValue(op.get("max_thickness", 2.0))
            
            priority_value = op.get("mass_reduction_priority", int(op.get("weight_penalty_factor", 0.5) * 100))
            self.mass_reduction_slider.setValue(priority_value)

            if project_data.get("has_volumetric_mesh"): self.volumetric_mesh = pv.read(os.path.join(temp_dir, 'volumetric_mesh.vtk'))
            if project_data.get("has_fea_result"):
                self.fea_result_model = pv.read(os.path.join(temp_dir, 'fea_result.vtk')); self.optim_group.setEnabled(True)
            if project_data.get("has_external_scalar"):
                data = np.loadtxt(os.path.join(temp_dir, 'scalar_field.txt'), skiprows=1)
                self.external_scalar = (data[:, :3], data[:, 3])
                self.use_scalar_checkbox.setEnabled(True); self.use_scalar_for_cell_size_checkbox.setEnabled(True); self.show_scalar_field_check.setEnabled(True)
            
            if project_data.get("has_optimization_results"):
                self.optimization_results = {}
                opt_dir = os.path.join(temp_dir, 'opt_results')
                if os.path.isdir(opt_dir):
                    i = 0
                    while True:
                        stl_path = os.path.join(opt_dir, f'iter_{i}_lattice.stl')
                        fea_path = os.path.join(opt_dir, f'iter_{i}_fea.vtk')
                        if os.path.exists(stl_path) and os.path.exists(fea_path):
                            self.optimization_results[i] = {
                                'surface_mesh': pv.read(stl_path),
                                'fea_result': pv.read(fea_path)
                            }
                            i += 1
                        else:
                            break
                    self._populate_iteration_selector()

            self.log("Project successfully loaded.", "success")
            self.view_selector.setCurrentText(project_data.get("active_view", "CAD"))
            self.update_view()

        except Exception as e:
            self.log(f"Failed to load project: {e}", "error"); traceback.print_exc()
            QMessageBox.critical(self, "Load Failed", f"An error occurred while loading the project:\n\n{e}")
        finally:
            shutil.rmtree(temp_dir) 
            self.set_busy(False)
            
    def _set_fea_stress_as_scalar(self):
        # This method remains unchanged
        if self.fea_result_model and "von_mises_stress" in self.fea_result_model.cell_data:
            self.log("Mapping FEA cell stress data to point data for visualization...")
            
            stress_values_ui = self.fea_result_model.cell_data["von_mises_stress"]
            stress_values_solver = self.unit_manager.convert_to_solver(stress_values_ui, 'pressure')
            
            temp_mesh = self.fea_result_model.copy()
            temp_mesh.cell_data["von_mises_stress"] = stress_values_solver
            mesh_with_point_stress = temp_mesh.cell_data_to_point_data()

            points = mesh_with_point_stress.points
            values = mesh_with_point_stress.point_data["von_mises_stress"]
            
            self.external_scalar = (points, values)
            self.use_scalar_checkbox.setEnabled(True)
            self.use_scalar_for_cell_size_checkbox.setEnabled(True)
            self.show_scalar_field_check.setEnabled(True)
            self.show_scalar_field_check.setChecked(True)
            
            self.log("FEA stress field is now active as the external scalar field.", "success")
            self.update_view("CAD")
        else:
            self.log("No FEA result with stress data available. Run a simulation first.", "error")
            QMessageBox.warning(self, "No Data", "Please run a simulation to generate a stress field first.")

    def _update_thickness_limit(self):
        # This method and subsequent helper methods remain unchanged
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
                'surface_mesh': target_mesh, 'detail_size': self.detail_size_spin.value(),
                'feature_angle': self.feature_angle_spin.value(), 'volume_g_size': self.volume_g_size_spin.value(),
                'skip_preprocessing': self.skip_preprocessing_check.isChecked(), 'mesh_order': self.mesh_order_spin.value(),
                'optimize_ho': self.ho_optimize_check.isChecked(), 'algorithm': self.mesh_algo_combo.currentText(),
                'log_func': self.log, 'lattice_model' : self.lattice_flag
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
            self.log(f"An unexpected error occurred in the meshing process: {e}", level="error"); traceback.print_exc()
        finally:
            self.set_busy(False)

    def run_hex_meshing(self):
        target_mesh = self.surface_mesh if self.surface_mesh else self.original_pv_shell
        if not target_mesh: self.log("A model must be generated or imported first.", level="error"); return
        self.set_busy(True); self.log("Generating hexahedral (voxel) mesh...")
        self._clear_fea_selections()
        try:
            voxel_size = self.detail_size_spin.value()
            self.volumetric_mesh = create_hexahedral_mesh(target_mesh, voxel_size, self.log)
            if self.volumetric_mesh:
                self.volumetric_mesh.point_data['persistent_ids'] = np.arange(self.volumetric_mesh.n_points)
                self.log("Successfully created hexahedral mesh.", level="success")
                self.update_view("Volumetric Mesh")
        except Exception as e:
            self.log(f"Hex mesh creation failed: {e}", level="error"); traceback.print_exc()
        finally:
            self.set_busy(False)

    def run_generation(self):
        if not self.original_pv_shell: self.log("Import a model first.", "error"); return
        self.set_busy(True); self.log("Starting lattice generation...")
        try:
            params = {
                'mesh': self.original_pv_shell, 'resolution': self.resolution_spin.value(),
                'wx': self.unit_x_spin.value(), 'wy': self.unit_y_spin.value(), 'wz': self.unit_z_spin.value(),
                'lattice_type': self.lattice_type_box.currentText(), 'thickness': self.lattice_thickness_spin.value(),
                'log_func': self.log, **self.remesh_settings, 'solidify': self.solidify_checkbox.isChecked(),
                'create_shell': self.shell_checkbox.isChecked(), 'shell_thickness': self.shell_thickness_spin.value(),
                'external_scalar': self.external_scalar,
                'use_scalar_for_cell_size': self.use_scalar_for_cell_size_checkbox.isChecked() and self.external_scalar is not None,
                'use_scalar_for_thickness': self.solidify_checkbox.isChecked() and self.use_scalar_checkbox.isChecked() and self.external_scalar is not None
            }
            
            lattice_infilled_surface_mesh = generate_infill_inside(**params)
            self.lattice_flag = True
            self.surface_mesh = lattice_infilled_surface_mesh
            
            self.log("Generation completed.", "success")
            self.update_view("Result")
            self.fea_group.setEnabled(True)
            self.optim_group.setEnabled(False)
            
        except Exception as e:
            self.log(f"Generation error: {e}", "error"); traceback.print_exc()
        finally:
            self.set_busy(False)
            
    def load_scalar_field(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Scalar Field", "", "Text Files (*.txt)")
        if file_path:
            try:
                data = np.loadtxt(file_path, skiprows=1)
                if data.shape[1] != 4: raise ValueError("Expected Nx4 matrix (X,Y,Z,Value)")
                self.external_scalar = (data[:, :3], data[:, 3])
                self.update_view()
                self.log(f"Loaded scalar field from: {os.path.basename(file_path)}")
                self.use_scalar_for_cell_size_checkbox.setEnabled(True)
                self.use_scalar_checkbox.setEnabled(True)
                self.show_scalar_field_check.setEnabled(True)
            except Exception as e:
                self.log(f"Scalar load error: {e}", "error")

    def import_file(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "Mesh Files (*.stl *.obj *.ply *.step *.stp *.iges *.igs)")
        if file_path:
            self.log("--- Starting Advanced Import ---")
            self.original_model_path = file_path
            model = self.importer.load(file_path)
            if model:
                # MODIFICATION: Clear optimization results on new import
                self.surface_mesh, self.volumetric_mesh, self.fea_result_model, self.external_scalar, self.optimization_results = None, None, None, None, {}
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
            
            solver_mesh = self.volumetric_mesh.copy()
            solver_mesh.points = self.unit_manager.convert_to_solver(solver_mesh.points, 'length')
            
            force_vector_solver = (
                self.unit_manager.convert_to_solver(self.fx_spin.value(), 'force'),
                self.unit_manager.convert_to_solver(self.fy_spin.value(), 'force'),
                self.unit_manager.convert_to_solver(self.fz_spin.value(), 'force')
            )

            params = {
                "mesh": solver_mesh, "material": MATERIALS[self.material_combo.currentText()], 
                "fixed_node_indices": list(self.fixed_node_indices), 
                "loaded_node_indices": list(self.load_node_indices), 
                "force": force_vector_solver, "log_func": self.log, 
                "stress_percentile_threshold": 99.5, 
                "progress_callback": self.update_progress_bar
            }
            self.fea_result_model = run_native_fea(**params)
            
            if self.fea_result_model:
                self.fea_result_model_solver = self.fea_result_model.copy()
                disp_solver = self.fea_result_model.point_data['Displacements']
                self.fea_result_model.points = self.unit_manager.convert_from_solver(self.fea_result_model.points, 'length')
                self.fea_result_model.point_data['Displacements'] = self.unit_manager.convert_from_solver(disp_solver, 'length')
                self.fea_result_model.point_data['displacement'] = np.linalg.norm(self.fea_result_model.point_data['Displacements'], axis=1)
                
                for field in ["von_mises_stress", "principal_s1", "principal_s2", "principal_s3"]:
                    if field in self.fea_result_model.cell_data:
                        stress_solver = self.fea_result_model.cell_data[field]
                        self.fea_result_model.cell_data[field] = self.unit_manager.convert_from_solver(stress_solver, 'pressure')
            
            self.log("FEA simulation completed.", "success")
            self.optim_group.setEnabled(True)
            self.update_view("FEA Result")
        except Exception as e: 
            self.log(f"FEA Error: {e}", "error"); traceback.print_exc()
        finally: 
            self.set_busy(False)
        
    def _populate_iteration_selector(self):
        self.optim_iteration_selector.blockSignals(True)
        self.optim_iteration_selector.clear()
        if not self.optimization_results:
            self.optim_iteration_selector.blockSignals(False)
            return
        
        # Populate from the results dictionary
        for i in sorted(self.optimization_results.keys()):
            self.optim_iteration_selector.addItem(f"Iteration {i}", userData=i)
        
        last_item_index = self.optim_iteration_selector.count() - 1
        if last_item_index >= 0:
            self.optim_iteration_selector.setCurrentIndex(last_item_index)
        self.optim_iteration_selector.blockSignals(False)

    def _get_current_model_for_export(self):
        view_text = self.view_selector.currentText()
        if view_text == "Result":
            return self.surface_mesh
        if view_text == "Volumetric Mesh": return self.volumetric_mesh
        if view_text == "FEA Result": return self.fea_result_model
        # --- MODIFICATION: Handle the new "Optimized Result" view logic ---
        if view_text == "Optimized Result":
            if self.optimization_results:
                iter_index = self.optim_iteration_selector.currentData()
                if iter_index is not None and iter_index in self.optimization_results:
                    iter_data = self.optimization_results[iter_index]
                    try:
                        if self.show_optim_fea_check.isChecked():
                            # Return the loaded FEA result mesh object
                            return iter_data['fea_result']
                        else:
                            # Return the loaded surface lattice mesh object
                            return iter_data['surface_mesh']
                    except KeyError as e:
                        self.log(f"Display error: Missing data for iteration {iter_index}: {e}", "error")
                        return None # Return None if data is missing
            # If no results or selection, fallback to the final best mesh from the optimizer
            return self.fea_result_model 
        return self.original_pv_shell
        
    def update_view(self, _=None):
        self.plotter.clear(); self.plotter.show_axes(); self.main_mesh_actor = None
        view_text = self.view_selector.currentText()
        self.plotter.clear_plane_widgets()

        is_optim_view = view_text == "Optimized Result"
        is_fea_view = view_text == "FEA Result"
        has_optim_results = bool(self.optimization_results)
        
        self.optim_iteration_selector.setVisible(is_optim_view)
        self.optim_iteration_selector_label.setVisible(is_optim_view)
        self.show_optim_fea_check.setVisible(is_optim_view)
        self.optim_iteration_selector.setEnabled(has_optim_results)
        self.show_optim_fea_check.setEnabled(has_optim_results)

        show_fea_controls = is_fea_view or (is_optim_view and has_optim_results and self.show_optim_fea_check.isChecked())
        
        self.fea_result_selector.setVisible(show_fea_controls)
        is_warped_view_active = show_fea_controls and self.show_deformation_check.isChecked()
        self.deformation_scale_label.setVisible(is_warped_view_active)
        self.deformation_scale_spin.setVisible(is_warped_view_active)
        self.show_deformation_check.setVisible(show_fea_controls)

        self.show_voxel_preview_check.setVisible(view_text == "CAD" and self.original_pv_shell is not None)
        self.show_scalar_field_check.setVisible(view_text == "CAD" and self.external_scalar is not None)
        self.select_toggle_button.setEnabled(self.volumetric_mesh is not None)
        
        self._update_selection_highlight(render=False)

        mesh_to_display = self._get_current_model_for_export()
                
        if not mesh_to_display:
            self.plotter.reset_camera()
            return
        
        scalar_bar_title = self.fea_result_selector.currentText().replace("_", " ").title()
        if show_fea_controls:
            pressure_unit = self.unit_manager.get_ui_label('pressure')
            length_unit = self.unit_manager.get_ui_label('length')
            if 'stress' in self.fea_result_selector.currentText():
                scalar_bar_title += f" ({pressure_unit})"
            elif 'displacement' in self.fea_result_selector.currentText():
                scalar_bar_title += f" ({length_unit})"
        
        mesh_kwargs = {'cmap': "turbo", 'scalar_bar_args': {'title': scalar_bar_title}}

        if is_warped_view_active and 'Displacements' in mesh_to_display.point_data:
            if np.linalg.norm(mesh_to_display.point_data['Displacements']) > 1e-9:
                scale_factor = self.deformation_scale_spin.value()
                mesh_kwargs['scalars'] = self.fea_result_selector.currentText()
                
                undeformed_mesh = mesh_to_display.copy()
                undeformed_mesh.points -= undeformed_mesh.point_data['Displacements'] * scale_factor
                self.plotter.add_mesh(undeformed_mesh, style='wireframe', color='grey', opacity=0.5)

                self.main_mesh_actor = self.plotter.add_mesh(mesh_to_display.warp_by_vector('Displacements', factor=scale_factor), **mesh_kwargs)
            else:
                self.log("Deformation is zero or negligible. Showing undeformed result."); is_warped_view_active = False
        
        if not is_warped_view_active:
            if show_fea_controls:
                mesh_kwargs['scalars'] = self.fea_result_selector.currentText()
            else:
                mesh_kwargs = {'color': 'orange', 'show_edges': True}

            if self.clipping_settings.get("enabled", False): 
                self.main_mesh_actor = self.plotter.add_mesh_clip_plane(mesh_to_display, invert=self.clipping_settings.get("invert", False), **mesh_kwargs)
            else: 
                self.main_mesh_actor = self.plotter.add_mesh(mesh_to_display, **mesh_kwargs)

        if view_text == "CAD" and self.show_voxel_preview_check.isChecked():
            try:
                density = [(b - a) / self.detail_size_spin.value() for a, b in zip(mesh_to_display.bounds[::2], mesh_to_display.bounds[1::2])]
                self.plotter.add_mesh(pv.voxelize(mesh_to_display, density=density), style='surface', color='tan', opacity=0.7)
            except Exception as e: self.log(f"Could not generate voxel preview: {e}", "error")
        
        if view_text == "CAD" and self.show_scalar_field_check.isChecked() and self.external_scalar is not None:
            points, values = self.external_scalar
            scalar_cloud = pv.PolyData(points)
            scalar_cloud['values'] = self.unit_manager.convert_from_solver(values, 'pressure') 
            self.plotter.add_points(
                scalar_cloud, render_points_as_spheres=True, point_size=10,
                scalars='values', cmap='viridis',
                scalar_bar_args={'title': f'Scalar Field ({self.unit_manager.get_ui_label("pressure")})'}
            )

        if self.fixed_node_actor: self.plotter.add_actor(self.fixed_node_actor)
        if self.load_node_actor: self.plotter.add_actor(self.load_node_actor)
        self.plotter.reset_camera()
        
    def _get_model_bounds_for_refinement(self):
        model_for_bounds = self.surface_mesh if self.surface_mesh is not None else self.original_pv_shell
        if model_for_bounds is None:
            self.log("Error: A model must be loaded or generated to get its bounds.", "error")
            QMessageBox.warning(self, "No Model", "A model must be loaded to get its bounds.")
            return
        bounds = model_for_bounds.bounds
        self.ref_xmin_spin.setValue(bounds[0]); self.ref_xmax_spin.setValue(bounds[1])
        self.ref_ymin_spin.setValue(bounds[2]); self.ref_ymax_spin.setValue(bounds[3])
        self.ref_zmin_spin.setValue(bounds[4]); self.ref_zmax_spin.setValue(bounds[5])
        self.log("Set refinement region to current model bounds.")
        
    def log(self, message, level="info", percent=None):
        prefix = f"[{level.upper()}]"; self.log_output.append(f"{prefix}: {message}")
        if percent is not None: self.progress_bar.setValue(percent)
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
        if not file_path: return
        try:
            model_to_export = self._get_current_model_for_export()
            if not model_to_export:
                self.log("No model to export for the current view.", "error")
                return
            
            self.log(f"Exporting model to {file_path}...")
            QApplication.processEvents()
            export_model(model_to_export, file_path)
            self.log(f"Successfully exported model to {file_path}", "success")
        except Exception as e:
            self.log(f"Export error: {e}", "error")

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Volumetric Mesh", "", "VTK Unstructured Grid (*.vtk *.vtk)")
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        with open('stylesheet.qss', 'r') as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: stylesheet.qss not found. Using default style.")

    window = LatticeMakerWindow()
    window.show()
    sys.exit(app.exec_())
