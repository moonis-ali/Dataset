import os
import numpy as np
from jakteristics import las_utils, compute_features

def read_data(filepath):
    ''' Read the point cloud to classify from a .txt file

        Attributes:
            filepath (string)   :   Path to the .txt file
        
        Return:
            X (np.array)   :    Point cloud and features
    '''
    X = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split(' ')
            if 'nan' not in tokens:   
                X.append([float(t) for t_index, t in enumerate(tokens)])
    return np.asarray(X, dtype=np.float64)

def write_point_cloud_txt(xyz, features, output_filepath):
    ''' Write point cloud data to a text file

        Attributes:
            xyz (np.array)          :   XYZ coordinates
            features (np.array)    :   Additional features
            output_filepath (str)  :   Path to the output .txt file
    '''
    # Add relative height as a feature
    min_z = np.min(xyz[:, 2])
    relative_height = xyz[:, 2] - min_z - 1
    features_with_relative_height = np.column_stack((features, relative_height))

    # Remove rows with NaN values in any column
    valid_rows = ~np.any(np.isnan(features_with_relative_height), axis=1)
    xyz = xyz[valid_rows, :]
    features_with_relative_height = features_with_relative_height[valid_rows, :]

    if xyz.shape[0] != features_with_relative_height.shape[0]:
        raise ValueError('Number of points in XYZ and features arrays must be the same.')

    with open(output_filepath, 'w') as f:
        for i in range(xyz.shape[0]):
            line = f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} {' '.join(map(str, features_with_relative_height[i]))}\n"
            f.write(line)

def process_files(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all LAS files in the input folder
    las_files = [file for file in os.listdir(input_folder) if file.endswith(".las")]

    total_files = len(las_files)
    for idx, las_file in enumerate(las_files, start=1):
        input_path = os.path.join(input_folder, las_file)
        output_path = os.path.join(output_folder, f"feature_{las_file[:-4]}.txt")

        # Read point cloud data from LAS file
        xyz = las_utils.read_las_xyz(input_path)

        # Compute features for two different search radii
        features11 = compute_features(xyz, search_radius=0.1, feature_names=['surface_variation'])
        features12 = compute_features(xyz, search_radius=0.1, feature_names=['PCA2', 'surface_variation', 'verticality'])
        features21 = compute_features(xyz, search_radius=0.2, feature_names=['surface_variation'])
        features22 = compute_features(xyz, search_radius=0.2, feature_names=['anisotropy', 'surface_variation', 'sphericity'])
        features41 = compute_features(xyz, search_radius=0.4, feature_names=['surface_variation'])
        features42 = compute_features(xyz, search_radius=0.4, feature_names=['anisotropy', 'surface_variation', 'sphericity'])
        features6 = compute_features(xyz, search_radius=0.6, feature_names=['anisotropy', 'sphericity'])
        features_all = np.hstack((features11, features12, features21, features22, features41, features42, features6))

        # Write the point cloud data with features to a text file
        write_point_cloud_txt(xyz, features_all, output_path)

        # Print progress message
        progress_percentage = (idx / total_files) * 100
        print(f"Processing {las_file} | Progress: {progress_percentage:.2f}%")

# Specify input and output folders
input_folder = 'path/folder/containing/files'
output_folder = 'path/output/folder'

# Process all files in the input folder and save output in the output folder
process_files(input_folder, output_folder)

