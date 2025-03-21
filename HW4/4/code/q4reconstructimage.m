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
        image_files = dir(ROOT_DIR +"s"+ CURR_DIR +"/"+ FILE_EXT);
        NUM_IMAGES = length(image_files);
        % disp(ROOT_DIR + CURR_DIR + FILE_EXT);
        % disp(NUM_IMAGES);
        for j = 1:NUM_IMAGES
            FILE_NAME = image_files(j).folder + "/" + image_files(j).name;
            CURR_IMAGE = im2double(imread(FILE_NAME)); % Read the image as a double-precision array
            SIZE_IMAGE = size(imread(FILE_NAME)); % Store the size of the image
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

%% Function for face recognition using eig, eigs
function face_rec_eig(X_train, Y_train, X_test, Y_test, k, SIZE_IMAGE)

    % Args
        % ROOT_DIR = Path to ORL/ directory
        % FILE_EXT = Extension the image files
        % NUM_IMAGES_FOLDERS = Number of folders considered 
        % TRAIN_SIZE = Number of Images in a folder considered for training
        % SIZE_IMAGE = Size of a image height*width

    % Returns
        % None

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
    for i=1:length(k)
        tar_vec = 1:k(i); % target array numbers
        eigen_space = projected_imgs(:, tar_vec); % eigen space
        temp = (X_train - mean_face);
        constructed_image = (eigen_space * (eigen_space') * (temp(:, 1))) + mean_face;
        constructed_image = reshape(constructed_image, [SIZE_IMAGE(1), SIZE_IMAGE(2)]);
        figure(i + 1);
        imagesc(constructed_image);
        colormap("gray");
        title("Reconstructed image for k=" + num2str(k(i)));

        % saving them
        exportgraphics(gcf, sprintf('recons_image_for_k=%d.png', k(i)), 'Resolution', 300);
    end

    % Subplot
    eigen_space = projected_imgs(:, 1:25);
    figure;
    for i = 1:25
        eigen_face = reshape(eigen_space(:, i), [SIZE_IMAGE(1), SIZE_IMAGE(2)]);
        subplot(5, 5, i);
        imagesc(eigen_face); 
        colormap("gray");
    end

    % Saving them
    exportgraphics(gcf, sprintf('recons_subplots.png'), 'Resolution', 300);
end

%%
ROOT_DIR = "ORL/"; % Path to ORL/ directory
FILE_EXT = "*.pgm"; % extension the image file
NUM_IMAGES_FOLDERS = 32; % NUMBER of folders inside the main folder considered for our task
TRAIN_SIZE = 6; % Number of images from the total number of images considered for training
[X_train, Y_train, X_test, Y_test, SIZE_IMAGE] = create_data(ROOT_DIR, FILE_EXT, NUM_IMAGES_FOLDERS, TRAIN_SIZE); % get data

% debug(X_train,Y_train, Y_test, X_test, SIZE_IMAGE) % uncomment to check whether loaded correctly or not

% Plot an image
image = im2double(imread("ORL/s1/1.pgm"));
m = size(image, 1); n = size(image, 2);
figure(1);
imagesc(image); colormap("gray"); 
title("Original image");
exportgraphics(gcf, sprintf('real_image.png'), 'Resolution', 300);

% Construct image for different k values
k = [2, 10, 20, 50, 75, 100, 125, 150, 175];
face_rec_eig(X_train, Y_train, X_test, Y_test, k, SIZE_IMAGE);
