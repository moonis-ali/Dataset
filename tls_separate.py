import os
import numpy as np
import laspy as lp
import tlseparation
from numpy import savetxt

# Directory paths
input_directory = "/home/moonis/Haryana_data/ITS/data/data/Plot_05"
output_directory = "/home/moonis/Haryana_data/ITS/data/tlseparate/Plot_05"

# Ensure the output directory exists, create it if necessary
os.makedirs(output_directory, exist_ok=True)

# Get a list of all LAS files in the input directory
las_files = [f for f in os.listdir(input_directory) if f.endswith(".las")]

# Loop through each LAS file
for las_file in las_files:
    # Read LAS file
    las_path = os.path.join(input_directory, las_file)
    point_cloud = lp.read(las_path)

    # Convert X and Y coordinates to floating-point format
    x_values = np.array(point_cloud.x, dtype=np.float64)
    y_values = np.array(point_cloud.y, dtype=np.float64)

    # Normalize X and Y coordinates using minimum values
    x_min, y_min = np.min(x_values), np.min(y_values)
    normalized_x = x_values - x_min
    normalized_y = y_values - y_min
    

    # Extract normalized XYZ coordinates
    xyz = np.vstack((normalized_x, normalized_y, point_cloud.z)).transpose()

    # Classify wood using TLseparation
    wood = tlseparation.classification.classify_wood.reference_classification(
        xyz, knn_list=[40, 50, 80, 100, 120], n_classes=4, prob_threshold=0.95
    )

    # Add back the minimum values to X and Y coordinates
    wood[:, 0] += x_min
    wood[:, 1] += y_min

    # Save wood classification results to a text file
    output_path = os.path.join(output_directory, las_file.replace(".las", ".txt"))
    savetxt(output_path, wood, fmt="%f")

    print(f"Processed: {las_file}")

print("Processing completed.")

