# unit_utils.py

# Base SI units for the solver
SOLVER_UNITS = {'length': 'm', 'force': 'N', 'pressure': 'Pa', 'energy': 'J'}

# Conversion factors FROM the base SI unit TO the target unit.
# (e.g., 1 meter = 1000 mm)
CONVERSION_FACTORS = {
    'length':   {'m': 1.0, 'mm': 1000.0, 'in': 39.3701},
    'force':    {'N': 1.0, 'lbf': 0.224809},
    'pressure': {'Pa': 1.0, 'MPa': 1e-6, 'psi': 0.000145038},
    'energy':   {'J': 1.0, 'mJ': 1000.0}
}

# Definition of available unit systems
SYSTEMS = {
    'SI (m, N, Pa)':         {'length': 'm',   'force': 'N',   'pressure': 'Pa',  'energy': 'J'},
    'Metric (mm, N, MPa)':   {'length': 'mm',  'force': 'N',   'pressure': 'MPa', 'energy': 'mJ'},
    'Imperial (in, lbf, psi)': {'length': 'in',  'force': 'lbf', 'pressure': 'psi', 'energy': 'J'}
}

class UnitManager:
    """A class to manage unit conversions between the UI and the solver."""
    def __init__(self, initial_system='Metric (mm, N, MPa)'):
        self.system_name = None
        self.ui_units = None
        self.set_system(initial_system)

    def set_system(self, system_name: str):
        """Sets the active unit system for the UI."""
        if system_name not in SYSTEMS:
            raise ValueError(f"Unknown unit system: {system_name}")
        self.system_name = system_name
        self.ui_units = SYSTEMS[system_name]

    def get_ui_label(self, unit_type: str) -> str:
        """Gets the abbreviation for a given unit type (e.g., 'mm' for 'length')."""
        return self.ui_units.get(unit_type, '')

    def convert_to_solver(self, value, unit_type: str):
        """Converts a value from the current UI unit to the base SI solver unit."""
        ui_unit = self.get_ui_label(unit_type)
        # --- MODIFICATION: Removed float() to allow NumPy array operations ---
        return value / CONVERSION_FACTORS[unit_type][ui_unit]

    def convert_from_solver(self, value, unit_type: str):
        """Converts a value from the base SI solver unit to the current UI unit."""
        ui_unit = self.get_ui_label(unit_type)
        # --- MODIFICATION: Removed float() to allow NumPy array operations ---
        return value * CONVERSION_FACTORS[unit_type][ui_unit]
