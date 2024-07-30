%Download the CamVid image data from the following URL for full dataset:

%imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';

load network;
load deeplabv3plusResnet18CamVid
net2 = googlenet('Weights','places365');
data = pretrainedNetwork; 

classes2 = [
    "Sky"         %1
    "Building"    %2
    "Pole"        %3
    "Driveways"        %4
    "Pavement"    %5
    "Tree"        %6
    "Traffic_Sign_Symbol"  %7
    "Fence"       %8
    "Car"         %9
    "Pedestrian"  %10
    "Bicyclist"   %11
    ];

inputSize = net2.Layers(1).InputSize(1:2);
classes = net2.Layers(end).Classes;

input_image_path = '0006R0_f01380.png';

I = imread(input_image_path);
I2 = imresize(I,inputSize);
[YPred,scores] = classify(net2,I2);
class_prediction = string(YPred);

[~, filename, ~] = fileparts(input_image_path);

[~,topIdx] = maxk(scores, 4);
topScores = scores(topIdx);
topScores=topScores(2:end);
topScores=round(topScores,4);
topIdx=topIdx(2:end);
topClasses = classes(topIdx);
topClasses = string(topClasses);
max_score = max(scores);
pre_index = find(max_score==scores);
max_probability=round(max_score*100);
cmap = camvidColorMap;
C = semanticseg(I, net);
L = imresize(C, [224,224], 'nearest');

B = labeloverlay(I2,L,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes2);


numericArray = double(L);
numCategories = max(numericArray(:));
prediction = zeros(1,numCategories)';
prediction_cont = zeros(3,numCategories)';
% Loop through each category and replace only the specific category with NaN values
for category = 1:numCategories
    % Create a binary mask for the current category
    mask = (numericArray == category);

    % Replace the specified segment within the mask with NaN
    d = I2;
    d(repmat(mask, [1, 1, 3])) = NaN;

    predictedall = predict(net2,d);
    predicted = predictedall(pre_index);
    prediction(category)=predicted;

    topIdx_pre=predictedall(topIdx);
    prediction_cont(category,:)=topIdx_pre;
    
end

feature_influence2=(max(prediction)-prediction)./(max(prediction)-min(prediction));

feature_influence=(max(prediction)-prediction)./(max(prediction)-min(prediction));
feature_influence=round(feature_influence*10);

feature_influence_count=(max(prediction_cont)-prediction_cont)./(max(prediction_cont)-min(prediction_cont));
feature_influence_count=round(feature_influence_count*10);

indices = find(feature_influence < 6);
siz=size(indices,1);
d2 = I2;
grayValue = 128; % You can adjust this value based on your preference
opacity = 0.5;  % Opacity factor (0 to 1)

for category = 1:siz

mask_inf = (numericArray == indices(category));
d2(repmat(mask_inf, [1, 1, 3])) = grayValue;

end

desired_size = [224, 224];
image1_resized = imresize(I, desired_size);
% Create a figure
figure;
subplot(1, 2, 1);
imshow(image1_resized);
title('Input scene');

% Display the second image in the second subplot
subplot(1, 2, 2);
imshow(d2);
title('Visual explanation');


output_folder = '..\output images';  % Modify this with your desired folder path
output_filename = fullfile(output_folder, strcat(filename, '.jpg'));
saveas(gcf, output_filename);
%close(gcf);

var_names = {'Prediction','Probability(%)','Features','Feature importance',...
    'Probability when the feature is removed (correspondingly)',...
    'First contrastive case', 'First contrastive case probability(%)',... 
    'Feature importance for the first contrastive case',...
    'Probability when the feature is removed for the first contrastive case',...
    'Second contrastive case', 'Second contrastive case probability(%)',... 
    'Feature importance for the second contrastive case',...
    'Probability when the feature is removed for the second contrastive case',...
    'Third contrastive case', 'Third contrastive case probability(%)',... 
    'Feature importance for the third contrastive case',...
    'Probability when the feature is removed for the third contrastive case'};

% Define variable values
variables = {class_prediction, max_probability, classes2, feature_influence, round(prediction*100),...
    topClasses(1), round(topScores(1)*100), feature_influence_count(:,1), round(prediction_cont(:,1)*100),...
    topClasses(2), round(topScores(2)*100), feature_influence_count(:,2), round(prediction_cont(:,2)*100),...
    topClasses(3), round(topScores(3)*100), feature_influence_count(:,3), round(prediction_cont(:,3)*100)};

% Determine the maximum length of variables
max_length = max(cellfun(@numel, variables));

% Specify the folder path
folder_path = '..\csv_files';

csv_filename = fullfile(folder_path, sprintf('%s_%s.csv', filename, class_prediction));
fileID = fopen(csv_filename, 'w');

% Write data to the CSV file
for j = 1:length(var_names)
    % Write variable name
    fprintf(fileID, '%s,', var_names{j});
    
    % Write variable values
    if isempty(variables{j})
        % Add commas for empty values
        fprintf(fileID, repmat(',', 1, max_length));
    elseif ischar(variables{j}) || isstring(variables{j}) % For string variables
        for k = 1:numel(variables{j})
            fprintf(fileID, '%s,', variables{j}(k));
        end
        % Add commas for missing values
        fprintf(fileID, repmat(',', 1, max_length - numel(variables{j})));
    else % For numerical variables
        for k = 1:numel(variables{j})
            fprintf(fileID, '%d,', variables{j}(k));
        end
        % Add commas for missing values
        fprintf(fileID, repmat(',', 1, max_length - numel(variables{j})));
    end
    
    fprintf(fileID, '\n');
end

% Close the file
fclose(fileID);

