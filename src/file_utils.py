import os
import file_paths as file


def save_mesh_to_file(mesh):
    # Get the file name (excluding extension) from the input image path
    filename = os.path.splitext(os.path.basename(file.get_image_path()))[0]
    mesh.save(file.output_path + filename + '.stl')