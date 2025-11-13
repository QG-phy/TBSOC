import pytest
import subprocess
import shutil
import json
import numpy as np
from pathlib import Path

# Define the root of the project and the example directories
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_DIR = PROJECT_ROOT / "example"
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"

# List of examples to test
EXAMPLES = ["GaAs", "Silicon", "TaAs"]

@pytest.mark.parametrize("example_name", EXAMPLES)
def test_integration_addsoc(example_name, tmp_path):
    """
    An integration test that runs the 'tbsoc addsoc' command on an example
    and compares the output band structure to a golden reference file.
    """
    source_dir = EXAMPLE_DIR / example_name
    
    # Copy all files from the source example directory to the temporary directory
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy(item, tmp_path)

    # Modify the input.json in the temporary directory to use the correct paths
    input_json_path = tmp_path / "input.json"
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    
    # Update all file paths to be absolute paths within the temp directory
    for key, value in input_data.items():
        if key.endswith('file'): # Heuristic to identify file path entries
            filename = Path(value).name
            input_data[key] = str(tmp_path / filename)

    input_data["outdir"] = str(tmp_path)

    with open(input_json_path, 'w') as f:
        json.dump(input_data, f, indent=4)

    # Run the tbsoc addsoc command
    # We use 'python -m tbsoc' to ensure we're running the code from the project
    command = ["python", "-m", "tbsoc", "addsoc", str(input_json_path), "--outdir", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    # Check that the command ran successfully
    assert result.returncode == 0, f"Command failed with error:\n{result.stderr}"

    # Define the path for the output and the golden reference file
    output_bands_path = tmp_path / "bands_soc.dat"
    golden_file_path = TEST_DATA_DIR / f"{example_name}_bands_soc.dat"

    assert output_bands_path.exists(), "The output file 'bands_soc.dat' was not created."

    # If the golden file exists, compare with it. Otherwise, create it.
    if golden_file_path.exists():
        golden_data = np.loadtxt(golden_file_path)
        output_data = np.loadtxt(output_bands_path)
        assert np.allclose(golden_data, output_data), "The output band structure does not match the golden reference."
    else:
        shutil.copy(output_bands_path, golden_file_path)
        pytest.skip(f"Golden file '{golden_file_path.name}' did not exist. It has been created. Please re-run the test.")
