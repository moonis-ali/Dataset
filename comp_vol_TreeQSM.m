clc;
clear all;
create_input;

% Specify the directory path
directoryPath = '/home/moonis/Haryana_data/ITS/data/filtered/Plot_02/improved_3';

% Get a list of all files in the directory
fileList = dir(fullfile(directoryPath, '*.txt'));

% Initialize arrays to store Volume and DBH for each file
allVolumes = zeros(length(fileList), 1);
allDBHs = zeros(length(fileList), 1);

% Loop through each file in the directory
for i = 25:length(fileList)
    % Load data from the current file
    filePath = fullfile(directoryPath, fileList(i).name);
    currentData = load(filePath);
    currentData = currentData - mean(currentData);
    
    % Display progress
    disp(['Processing file ', num2str(i), ' of ', num2str(length(fileList))]);
    
    % Apply treeqsm function multiple times
    volumes = zeros(1, 10);
    dbhs = zeros(1, 10);
    
    for j = 1:10
        currentOutput = treeqsm(currentData, inputs);
        volumes(j) = currentOutput.treedata.TotalVolume;
        dbhs(j) = currentOutput.treedata.DBHqsm;
    end
    
    % Average the total volume and DBH over multiple executions
    averageVolume = mean(volumes);
    averageDBH = mean(dbhs);
    
    % Store results
    allVolumes(i) = averageVolume;
    allDBHs(i) = averageDBH;
end

% Display or save the results as needed
disp('Average Volume for all files:');
disp(allVolumes);

disp('Average DBH for all files:');
disp(allDBHs);

% If needed, you can save the results to a file
save('Plot_03.mat', 'allVolumes', 'allDBHs');
