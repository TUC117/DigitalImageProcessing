clear;

%% Function to create data from the folder
function [X_train, Y_train, X_test, Y_test, SIZE_IMAGE] = create_data(ROOT_DIR, FILE_EXT, NUM_IMAGES_FOLDERS, TRAIN_SIZE) % Fucntion to create data
    
    % Args
        % ROOT_DIR = Path to ORL/ directory
        % FILE_EXT = Extension the image files
        % NUM_IMAGES_FOLDERS = Number of folders considered 
        % TRAIN_SIZE = Number of Images in a folder considered for training
    % Returns
        % X_train = Training data
        % Y_train = labels for the training images
        % X_test = Testing data
        % Y_test = Labels for the test images
        % SIZE_IMAGE = Size of a image height*width

    X_train = []; % Training set i.e., first 6 images of each folder
    Y_train = []; % Corresponding subjects/faces 
    X_test = []; % Test set i.e., last 4 images of each folder
    Y_test = []; % Corresponding subjects/faces
    for i = 1:NUM_IMAGES_FOLDERS
        CURR_DIR = num2str(i);  % Converting current directory number into string
        image_files = dir(ROOT_DIR + "s" + CURR_DIR +"/"+ FILE_EXT);
        NUM_IMAGES = length(image_files);
        % disp(ROOT_DIR + CURR_DIR + FILE_EXT);
        % disp(NUM_IMAGES);
        for j = 1:NUM_IMAGES
            FILE_NAME = image_files(j).folder + "/" + image_files(j).name;
            CURR_IMAGE = im2double(imread(FILE_NAME)); % Read the image as a double-precision array
            SIZE_IMAGE = size(imread(FILE_NAME));
            if j < TRAIN_SIZE+1
                X_train = cat(2, X_train, CURR_IMAGE(:)); % Concatenate image column vector
                Y_train = cat(2, Y_train, i); % Subject label
            else
                X_test = cat(2, X_test, CURR_IMAGE(:)); % Concatenate test image
                Y_test = cat(2, Y_test, i); % Subject label
            end
        end
    end
end

%% Function to calculate the threshold value for different mertics
function find_threshold(X_train, Y_train, X_test, Y_test, metrics)
    
    % Args
        % X_train = Training data
        % Y_train = labels for the training images
        % X_test = Testing data
        % Y_test = Labels for the test images
        % metrics = different metrics from ["f1_score", "accuracy_score", "mcc", "prec", "rec"]
    
    k = 75; % Since we got better results form 75 let's consider this as k for this problem
    mean_face = mean(X_train, 2);
    mean_face_test = mean(X_test, 2);
    A = X_train - mean_face; % substract mean 
    
    C = A' * A; % Covariance Matrix
    [eigen_vec, eigen_vals] = eig(C, 'vector'); % Calculate eigen values and vectors
    [eigen_vals, indices] = sort(eigen_vals, 'descend'); % Sort the values in descending order
    eigen_vec = eigen_vec(:, indices); % Reordering the eigen vectors bases on eigen values
    projected_imgs = A* eigen_vec; % Project the matrix onto eigen vectors
    % Normalize the projected vectors
    for i  = 1:size(projected_imgs, 2)
        projected_imgs(:, i) = projected_imgs(:, i) / norm(projected_imgs(:, i));
    end

    % We need to find threshold value by 
    threshold_values = linspace(70, 300, 100);

    % Store necessary values
    best_score = 0; % Value of current measuring metrics
    best_recall = 0; % Value of recall at best_score
    best_threshold = 0; % Value of threshold at best_score
    best_f1 = 0; % Value of f1 at best_score
    best_accuracy = 0; % Value of accuracy at best_score
    TP = 0; % Value of True Positive at best_score
    FP = 0; % Value of False Positive at best_score
    TN = 0; % Value of True Negative at best_score
    FN = 0; % Value of False Negative at best_score
    recognition_rate = 0; % Value of recognition rate at best_score
    best_mcc = 0; % Value of matthews correlation coefficient at best_score
    

    for ind=1:length(threshold_values) % itterate over all thresholds
        threshold = threshold_values(ind);
        temp_TN = 0;
        temp_FN = 0;
        temp_TP = 0;
        temp_FP = 0;
    
        eigen_space = projected_imgs(:, 1:k);
        eigen_coef = (eigen_space') * A;
        test_coef = (eigen_space') * (X_test - mean_face_test);
        rc = 0;
        for j = 1:size(X_test, 2)
            % For all images in the X_test
            error = sum((eigen_coef - test_coef(:, j)).^2); % Calculate square error
            [m, index] = min(error); % Find minimum value and index
            if m <= threshold % Then it is correct prediction
                if Y_test(j) == 0 % If prediction is from non trained part False Postive
                    temp_FP  = temp_FP  + 1;
                else % True Positive
                    temp_TP = temp_TP + 1;
                    if Y_train(index) == Y_test(j)
                        rc = rc + 1;
                    end
                end
            else % Then our prediction is wrong
                if Y_test(j) == 0 % if prediction is wrong from non trained part then True Negative
                    temp_TN = temp_TN + 1;
                else % False negative
                    temp_FN = temp_FN + 1;
                end
            end
        end

        % Calulate recall, precision, f1_score, mcc, accuracy for current threshold
        recall = temp_TP / (temp_TP + temp_FN);
        precision = temp_TP /(temp_TP + temp_FP);
        f1_score = 2 * ((precision*recall)/(precision+recall));
        mcc = ((temp_TP * temp_TN) - (temp_FN *temp_FP))/(sqrt((temp_TP+temp_FP)*(temp_TP+temp_FN)*(temp_TN+temp_FP)*(temp_FN+temp_TN)));
        accuracy = (temp_TP + temp_TN)/(temp_TN+temp_TP+ temp_FP + temp_FN);

        % Comapre the metrics and update accordingly
        if strcmp(metrics, "f1")
            if f1_score > best_score
                best_score = f1_score;
                best_accuracy = accuracy;
                best_f1 = f1_score;
                best_threshold = threshold;
                FP = temp_FP;
                best_recall = recall;
                FN = temp_FN;
                TP = temp_TP;
                TN = temp_TN;
                recognition_rate = rc/size(X_test, 2);
                best_mcc = mcc;
            end
        end
        if strcmp(metrics, "mcc")
            if mcc > best_score
                best_score = mcc;
                best_accuracy = accuracy;
                best_f1 = f1_score;
                best_recall = recall;
                best_threshold = threshold;
                FP = temp_FP;
                FN = temp_FN;
                TP = temp_TP;
                TN = temp_TN;
                recognition_rate = rc/size(X_test, 2);
                best_mcc = mcc;
            end
        end
        if strcmp(metrics, "acc")
            if accuracy > best_score
                best_score = accuracy;
                best_recall = recall;
                best_accuracy = accuracy;
                best_f1 = f1_score;
                best_threshold = threshold;
                FP = temp_FP;
                FN = temp_FN;
                TP = temp_TP;
                TN = temp_TN;
                recognition_rate = rc/size(X_test, 2);
                best_mcc = mcc;
            end
        end
        if strcmp(metrics, "prec")
            if precision > best_score
                best_score = precision;
                best_recall = recall;
                best_accuracy = accuracy;
                best_f1 = f1_score;
                best_threshold = threshold;
                FP = temp_FP;
                FN = temp_FN;
                TP = temp_TP;
                TN = temp_TN;
                recognition_rate = rc/size(X_test, 2);
                best_mcc = mcc;
            end
        end
        if strcmp(metrics, "recc")
            if recall > best_score
                best_score = recall;
                best_recall = recall;
                best_accuracy = accuracy;
                best_f1 = f1_score;
                best_threshold = threshold;
                FP = temp_FP;
                FN = temp_FN;
                TP = temp_TP;
                TN = temp_TN;
                recognition_rate = rc/size(X_test, 2);
                best_mcc = mcc;
            end
        end
    end

    % Print important outputs
    fprintf("\n");
    fprintf('With measuring metrics %s\n', metrics);
    fprintf("Accuracy: %f\n", best_accuracy);
    fprintf("F1 Score: %f\n", best_f1);
    fprintf("MCC: %f\n", best_mcc);
    fprintf("Best Threshold: %f\n", best_threshold);
    fprintf("Recall: %f\n", best_recall);
    % Print the confusion matrix
    fprintf("Confusion matrix:\n");
    fprintf("TP: %d\tFP: %d\n", TP, FP);
    fprintf("FN: %d\tTN: %d\n", FN, TN);
    fprintf("Recognition rate: %f\n", recognition_rate);
end

%%
ROOT_DIR = "ORL/";
FILE_EXT = "*.pgm";
NUM_IMAGES_FOLDERS = 32;
TRAIN_SIZE = 6;
[X_train, Y_train, X_test, Y_test, SIZE_IMAGE] = create_data(ROOT_DIR, FILE_EXT, NUM_IMAGES_FOLDERS, TRAIN_SIZE);

% Now let's append last 8 folders to test
for i = 33:40
    CURR_DIR = num2str(i);  % Converting current directory number into string
    image_files = dir(ROOT_DIR + "s" + CURR_DIR +"/"+ FILE_EXT);
    NUM_IMAGES = length(image_files);
    for j = 1:NUM_IMAGES
        if j>=7
            FILE_NAME = image_files(j).folder + "/" + image_files(j).name;
            CURR_IMAGE = im2double(imread(FILE_NAME)); % Read the image as a double-precision array
            X_test = cat(2, X_test, CURR_IMAGE(:)); % Concatenate test image
            Y_test = cat(2, Y_test, 0); % Subject label let us assume all unknown as 0 labelled 
        end
    end    
end
find_threshold(X_train, Y_train, X_test, Y_test, "f1"); % Finding best threshold with f1 score as metric
find_threshold(X_train, Y_train, X_test, Y_test, "acc");% Finding best threshold with accuracy score as metric
find_threshold(X_train, Y_train, X_test, Y_test, "mcc");% Finding best threshold with matthews correlation coefficient as metric
find_threshold(X_train, Y_train, X_test, Y_test, "prec");% Finding best threshold with precision as metric
find_threshold(X_train, Y_train, X_test, Y_test, "recc");% Finding best threshold with recall as metric
