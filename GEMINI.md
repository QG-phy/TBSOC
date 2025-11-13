# GEMINI.md

## Project Overview

This project, `tbsoc`, is a Python-based command-line tool for computational materials science. Its primary purpose is to estimate the spin-orbit coupling (SOC) strength in solid-state systems. It combines *ab initio* data with tight-binding calculations to achieve this.

The tool is built using Python and relies on libraries like `numpy`, `scipy`, and `matplotlib`. It is packaged using `poetry` and can be installed via `pip`. The project is structured into a library (`tbsoc/lib`) containing the core scientific logic and a command-line interface (`tbsoc/entrypoints`).

## Building and Running

### Installation

The project can be installed from PyPI:

```bash
pip install tbsoc
```

Or from source:

```bash
pip install .
```

### Running the Tool

The tool is run from the command line using the `tbsoc` command. It has three main subcommands:

*   `tbsoc precalc <input.json>`: Pre-calculates the non-SOC tight-binding model.
*   `tbsoc addsoc <input.json>`: Adds the SOC to the non-SOC tight-binding model.
*   `tbsoc fit <input.json>`: Fits the SOC strength to DFT band structure.

The `input.json` file contains the necessary parameters for the calculation. Examples are provided in the `example/` directory.

### Testing

The project uses `pytest` for testing. To run the tests, you would typically run:

```bash
pytest
```

*(Note: The exact test command is not explicitly defined in the project files, but `pytest` is a standard convention.)*

## Development Conventions

*   **Code Style:** The code follows standard Python conventions (PEP 8).
*   **Packaging:** The project uses `poetry` for dependency management and packaging.
*   **Command-Line Interface:** The CLI is built using Python's `argparse` module.
*   **Core Logic:** The scientific calculations are implemented in the `tbsoc/lib` directory, with clear separation of concerns for reading input files, performing calculations, and generating matrices.
*   **Input Files:** The primary input format is a JSON file (`input.json`), and the tool also reads standard file formats from computational materials science like `wannier90_hr.dat`, `POSCAR`, and `KPOINTS`.
