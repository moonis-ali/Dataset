import os
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

feat_to_use = []  # Indices of the features to use. If n is the number of features, from 0 to n-1

def load_features(filepath):
    ''' Load the features indices from a .txt file

        Attributes:
            filepath (string)   :  Path to the .txt file
    '''
    with open(filepath, 'r') as f:
        for line_index, line in enumerate(f.readlines()):
            tokens = line.strip().split(' ')
            if line_index == 0:
                global feat_to_use
                feat_to_use = [int(t) for t in tokens]

def read_model(filepath):
    ''' Read the Random Forest model from a .pkl file

        Attributes:
            filepath (string)   :   Path to the .pkl file
    '''
    return pickle.load(open(filepath, 'rb'))

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

def write_classification(X, Y, filename):
    ''' Write a classified point cloud

        Attributes:
            X (np.array)        :   Point cloud and features
            Y (np.array)        :   Classes
            filename (string)   :   Output file path
    '''
    with open('{}.txt'.format(filename), 'w') as out:
        X = X.tolist()
        Y = Y.tolist()
        for index, x in enumerate(X):
            x_as_str = " ".join([str(i) for i in x])
            out.write('{} {}\n'.format(x_as_str, str(Y[index])))

def classify_all_files(input_folder, output_folder, feature_indices_filepath, model_filepath):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    feature_files = [file for file in os.listdir(input_folder) if file.endswith(".txt")]

    # Load feature indices
    load_features(feature_indices_filepath)

    # Load trained model
    model = read_model(model_filepath)

    for feature_file in feature_files:
        input_path = os.path.join(input_folder, feature_file)
        output_path = os.path.join(output_folder, f"classified_{feature_file}")

        # Read data to classify
        X = read_data(input_path)

        start_time = time.time()
        print(f'Classifying {feature_file} ...')
        # Classify the data
        Y_pred = model.predict(X[:, feat_to_use])
        end_time = time.time()

        # Save classification results
        write_classification(X, Y_pred, output_path)

        # Print the time taken for classification
        print(f'Time taken for {feature_file}: {end_time - start_time:.2f} seconds')

if __name__ == "__main__":
    start = time.time()
    
    # Specify input and output folders
    input_folder = 'path/to/your/folder/containing/input/files'
    output_folder = 'path/to/your/output/folder'
    feature_indices_filepath = 'path/to/your/file/containing/FEATURE_INDEX.txt/file'
    model_filepath = 'path/to/your/RF_model'

    # Classify all files in the input folder and save the results in the output folder
    classify_all_files(input_folder, output_folder, feature_indices_filepath, model_filepath)

    end = time.time()
    print(f'All files classified in: {end - start} seconds')

