clear;

%% Function to create data from the folder
function [X_train, Y_train, X_test, Y_test, SIZE_IMAGE] = create_data(ROOT_DIR, FILE_EXT, NUM_IMAGES_FOLDERS, TRAIN_SIZE, MISSING) % Fucntion to create data
    
    % Args
        % ROOT_DIR = Path to CroppedYale/ directory
        % FILE_EXT = Extension the image files
        % NUM_IMAGES_FOLDERS = Number of folders considered 
        % TRAIN_SIZE = Number of Images in a folder considered for training
        % MISSING = Folder which was missing 
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
        if i == MISSING % Ignore missing folder
            continue;
        end
        CURR_DIR = num2str(i, '%02d');  % Converting current directory number into string
        image_files = dir(ROOT_DIR + "yaleB"+CURR_DIR +"/"+ FILE_EXT);
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

%% Function to check whether data loded correctly or not
function debug(X_train,Y_train, Y_test, X_test, SIZE_IMAGE)
    num_train_images = size(X_train, 2); % Number of training images
    % Visualize up to 10 training images or fewer if less than 10 exist
    figure;
    for i = 1:min(10, num_train_images)
        img = reshape(X_train(:, i), [112, 92]); % Reshape the column vector to 112x92
        subplot(2, 5, i); % Create a 2x5 grid of images
        imshow(img, []); % Display the image
        title(['Subject ', num2str(Y_train(i))]);
    end
    sgtitle('First 10 Training Images');
    num_test_images = size(X_test, 2); % Number of test images
    
    % Visualize up to 10 test images or fewer if less than 10 exist
    figure;
    for i = 1:min(10, num_test_images)
        img = reshape(X_test(:, i), [112, 92]); % Reshape the column vector to 112x92
        subplot(2, 5, i); % Create a 2x5 grid of images
        imshow(img, []); % Display the image
        title(['Subject ', num2str(Y_test(i))]);
    end
    sgtitle('First 10 Test Images');
    
    % Check whether it is of column matrix
    first_column = X_train(:, 1); 
    if size(first_column, 1) == SIZE_IMAGE(1)*SIZE_IMAGE(2)
        disp('OK');
    else
        disp('NOT OK');
    end
end

%% Function to calculate Recognition Rate using eig method
function reg_rate = face_rec_eig(X_train, Y_train, X_test, Y_test, k)

    % Args
        % X_train = Training data
        % Y_train = labels for the training images
        % X_test = Testing data
        % Y_test = Labels for the test images
        % k = array of k values
     % Returns 
        % reg_rate = Recognition Rate 

    mean_face = mean(X_train, 2); % Mean vector of train
    A = X_train - mean_face; % substract mean 
    
    % For eig part
    C = A' * A; % Covariance Matrix
    [eigen_vec, eigen_vals] = eig(C, 'vector'); % Calculate eigen values and vectors
    [eigen_vals, indices] = sort(eigen_vals, 'descend'); % Sort the values in descending order
    eigen_vec = eigen_vec(:, indices); % Reordering the eigen vectors bases on eigen values
    projected_imgs = A* eigen_vec; % Project the matrix onto eigen vectors
    % Normalize the projected vectors
    for i  = 1:size(projected_imgs, 2)
        projected_imgs(:, i) = projected_imgs(:, i) / norm(projected_imgs(:, i));
    end
    reg_rate = zeros(size(k));
    for i=1:length(k)
        tar_vec = 1:k(i);
        eigen_space = projected_imgs(:, tar_vec);
        eigen_coef = (eigen_space') * (X_train - mean_face);
        test_coef = (eigen_space') * (X_test - mean_face); 
        for j = 1:size(X_test, 2)
            difference = eigen_coef - test_coef(:, j);
            squared_diffs = sum(difference.^2);
            [m, index] = min(squared_diffs);
            if Y_train(index) == Y_test(j)
                reg_rate(i) = reg_rate(i) + 1;
            end
        end
        reg_rate(i) = reg_rate(i) / size(X_test, 2);
    end

    % Plot the Recognition Rate vs k values
    figure;
    plot(1:length(k), reg_rate, '*-'); 
    xlabel("k (index)", 'FontSize', 14, 'FontWeight', 'bold'); 
    ylabel("Recognition Rate", 'FontSize', 14, 'FontWeight', 'bold');
    title("Recognition Rate vs. k for YALE using EIG", 'FontSize', 14, 'FontWeight', 'bold'); 
    grid on;    
    xticks(1:length(k)); 
    xticklabels(k); 

    % Save the figure
    exportgraphics(gcf, 'q4-eig-YALE.png', 'Resolution', 300); 
end

%% Function to calculate Recognition Rate using svd method
function reg_rate = face_rec_eig_ignore(X_train, Y_train, X_test, Y_test, k)

    % Args
        % X_train = Training data
        % Y_train = labels for the training images
        % X_test = Testing data
        % Y_test = Labels for the test images
        % k = array of k values
     % Returns 
        % reg_rate = Recognition Rate 

    mean_face = mean(X_train, 2); % Mean vector of train
    A = X_train - mean_face; % substract mean 

    % For eig part
    C = A' * A; % Covariance Matrix
    [eigen_vec, eigen_vals] = eig(C, 'vector'); % Calculate eigen values and vectors
    [eigen_vals, indices] = sort(eigen_vals, 'descend'); % Sort the values in descending order
    eigen_vec = eigen_vec(:, indices); % Reordering the eigen vectors bases on eigen values
    projected_imgs = A* eigen_vec; % Project the matrix onto eigen vectors
    % Normalize
    for i  = 1:size(projected_imgs, 2)
        projected_imgs(:, i) = projected_imgs(:, i) / norm(projected_imgs(:, i));
    end
    reg_rate = zeros(size(k));
    for i=1:length(k)
        tar_vec = 1:k(i)+3; % Take current + 3 features
        eigen_space = projected_imgs(:, tar_vec);
        eigen_coef = (eigen_space') * (X_train - mean_face); % Calculating eigen coefficeint for train
        eigen_coef = eigen_coef(4:end, :); % Remove first 3 eigen vectors
        test_coef = (eigen_space') * (X_test - mean_face); % Calculating eigen coefficeint for test
        test_coef = test_coef(4:end, :); % Remove first 3 eigen vectors
        for j = 1:size(X_test, 2)
            % Squared error
            difference = eigen_coef - test_coef(:, j);
            squared_diffs = sum(difference.^2);
            [m, index] = min(squared_diffs);
            if Y_train(index) == Y_test(j)
                reg_rate(i) = reg_rate(i) + 1;
            end
        end
        reg_rate(i) = reg_rate(i) / size(X_test, 2);
    end

    % Plot the Recognition Rate vs k values
    figure;
    plot(1:length(k), reg_rate, '*-');
    xlabel("k (index)", 'FontSize', 14, 'FontWeight', 'bold');
    ylabel("Recognition Rate", 'FontSize', 14, 'FontWeight', 'bold');
    title("YALE using EIG without top three", 'FontSize', 14, 'FontWeight', 'bold'); 
    grid on;    
    xticks(1:length(k));
    xticklabels(k);

    % Save the figure    
    exportgraphics(gcf, 'q4-eig-YALE-ignore.png', 'Resolution', 300); 
end

%%
ROOT_DIR = "CroppedYale/"; % Path to ORL/ directory
FILE_EXT = "*.pgm"; % extension the image file
NUM_IMAGES_FOLDERS = 39; % NUMBER of folders inside the main folder considered for our task
TRAIN_SIZE = 40;% Number of images from the total number of images considered for training
MISSING = 14; % FOlder number which was missing
[X_train, Y_train, X_test, Y_test, SIZE_IMAGE] = create_data(ROOT_DIR, FILE_EXT, NUM_IMAGES_FOLDERS, TRAIN_SIZE, MISSING); % get data
% debug(X_train,Y_train, Y_test, X_test, SIZE_IMAGE) % uncomment to check whether loaded correctly or not
% Recognition
k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];
recognition_rate_eig = face_rec_eig(X_train, Y_train, X_test, Y_test, k);
recognition_rate_eig_ignore  = face_rec_eig_ignore(X_train, Y_train, X_test, Y_test, k);