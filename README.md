# LatticeMaker 🚀

A desktop application for generating, analyzing, and optimizing 3D lattice structures. Built with Python, PyQt5, PyVista, Gmsh, and a custom FEA solver accelerated with Numba.



## Features

- **CAD Import:** Natively imports STEP, IGES, STL, and other mesh formats.
- **Implicit Lattice Generation:** Creates complex lattices like Gyroid, Diamond, and Neovius inside any CAD shell.
- **Robust Meshing:** Uses Gmsh for high-quality tetrahedral meshing with automatic pre-processing and repair.
- **FEA Simulation:** A high-performance, native FEA solver for linear static analysis.
- **Bayesian Optimization:** Automatically optimizes lattice thickness to reduce stress concentrations.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Hadimn77/LatticeMaker](https://github.com/Hadimn77/LatticeMaker.git)
   cd LatticeMaker