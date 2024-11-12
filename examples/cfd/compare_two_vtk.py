import pyvista as pv
import numpy as np


def read_field(file_path, field_name):
    """Reads a VTK file and retrieves the specified field data as a NumPy array."""
    grid = pv.read(file_path)
    if field_name not in grid.cell_data:
        raise ValueError(f"Field '{field_name}' not found in {file_path}")
    field_data = grid.cell_data[field_name]
    return grid, field_data


def subtract_fields(file1, file2, field_name, output_file):
    """Subtracts the specified field in file2 from file1 and writes the result to output_file."""
    grid1, field_data1 = read_field(file1, field_name)
    _, field_data2 = read_field(file2, field_name)

    # Perform the subtraction
    result_field = field_data1 - field_data2

    # Add the result as a new field to grid1
    grid1.cell_data[f"{field_name}_difference"] = result_field

    # Save the modified grid to a new VTK file
    grid1.save(output_file)


# Example usage
file1 = "second.vtk"
file2 = "third.vtk"
field_name = "umag"  # Replace with the name of the field to subtract
output_file = "result.vtk"

subtract_fields(file1, file2, field_name, output_file)
