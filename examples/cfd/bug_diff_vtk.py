import pyvista as pv
import numpy as np
import sys

def compute_vector_field_difference(file1, file2, output_file, field_name="test"):
    # Load the two datasets
    mesh1 = pv.read(file1)
    mesh2 = pv.read(file2)

    # Check if both fields exist
    if field_name not in mesh1.cell_data or field_name not in mesh2.cell_data:
        raise ValueError(f"Field '{field_name}' not found in both input files.")

    # Extract the vector fields
    vectors1 = mesh1.cell_data[field_name]
    vectors2 = mesh2.cell_data[field_name]

    # Check that both fields have the same shape
    if vectors1.shape != vectors2.shape:
        raise ValueError("Vector fields do not have the same shape.")

    # Compute the difference
    diff = (vectors1 - vectors2)/(1e-12 + vectors1)

    # Create a new mesh with the same geometry as the first input
    result = mesh1.copy()
    result.cell_data.clear()
    result.cell_data["vector_difference"] = diff

    # Save the result
    result.save(output_file)
    print(f"Vector field difference written to '{output_file}'")


file1 = 'r1/coalescence_factor.vti.vtk'
file2 = 'r2/coalescence_factor.vti.vtk'
output_file = 'vector_field_difference_coalescence_factor.vtk'
compute_vector_field_difference(file1, file2, output_file)

file1 = 'r1/inv_coalescence_factor.vti.vtk'
file2 = 'r2/inv_coalescence_factor.vti.vtk'
output_file = 'vector_field_difference_inv_coalescence_factor.vtk'
compute_vector_field_difference(file1, file2, output_file)

for step in range(0, 11, 1):
    file1 = f'r1/u_lid_driven_cavity_{step}.vti.vtk'
    file2 = f'r2/u_lid_driven_cavity_{step}.vti.vtk'
    output_file = f'vector_field_difference_{step}.vtk'
    compute_vector_field_difference(file1, file2, output_file)
