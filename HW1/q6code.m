im1 = double(imread('goi1.jpg'));
im2 = double(imread('goi2.jpg'));
figure;
imshow(im1/255);
figure;
imshow(im2/255);
x1 = zeros(1,12);
y1 = zeros(1,12);
x2 = zeros(1,12);
y2 = zeros(1,12);
for i = 1:12
    figure(1); imshow(im1/255);
    [x1(i), y1(i)] = ginput(1);
    
    figure(2); imshow(im2/255);
    [x2(i), y2(i)] = ginput(1);
end

An = ones(3,12);
bn = ones(3,12);
for i = 1:12
   An(:,i)=[x1(i), y1(i), 1];
   bn(:,i)=[x2(i), y2(i), 1];
end

affine_matrix = bn * pinv(An);
[rows, cols, temp] = size(im1);
transformed_image = zeros(rows, cols);

disp (affine_matrix)

for x = 1:cols
    for y = 1:rows
        new_coords = inv(affine_matrix) * [x; y; 1];
        corr_x = round(new_coords(1));
        corr_y = round(new_coords(2));

        if corr_x >= 1 && corr_x <= cols && corr_y >= 1 && corr_y <= rows
            transformed_image(y,x) = im1(corr_y, corr_x);
            %backward wrapping
        end
    end
end

min1 = min(transformed_image(:));
max1 = max(transformed_image(:));
ntransformed_image = uint8(((transformed_image - min1)/(max1 - min1))*255);
% Display images side by side
figure;
subplot(1, 3, 1); imshow(im1/255); title('Original Image');
subplot(1, 3, 2); imshow(im2/255); title('Target Image');
subplot(1, 3, 3); imshow(ntransformed_image); title('Transformed Image (NN)');
imwrite(ntransformed_image,'nearest.jpg');

transformed_image_bilinear = zeros(rows, cols);

for x = 1:cols
    for y = 1:rows
        new_coords = inv(affine_matrix) * [x; y; 1];
        x_low = floor(new_coords(1));
        x_high = ceil(new_coords(1));
        y_low = floor(new_coords(2));
        y_high = ceil(new_coords(2));

        if x_low >= 1 && x_high <= cols && y_low >= 1 && y_high <= rows
            % Bilinear interpolation formula
            pix1 = im1(y_low, x_low);
            pix2 = im1(y_high, x_low);
            pix3 = im1(y_low, x_high);
            pix4 = im1(y_high, x_high);
            fx = new_coords(1) - x_low;
            fy = new_coords(2) - y_low;
              %backward wrapping
            transformed_image_bilinear(y, x) = (1-fx)*(1-fy)*pix1 + (1-fx)*fy*pix2  + fx*(1-fy)*pix3;
            transformed_image_bilinear(y, x) =  transformed_image_bilinear(y, x) + fx*fy*pix4;
        end
    end
end
min2 = min(transformed_image_bilinear(:));
max2 = max(transformed_image_bilinear(:));
ntransformed_image_bilinear = uint8(((transformed_image_bilinear - min2)/(max2 - min2))*255);
% Display images side by side
figure;
subplot(1, 3, 1); imshow(im1/255); title('Original Image');
subplot(1, 3, 2); imshow(im2/255); title('Target Image');
subplot(1, 3, 3); imshow(ntransformed_image_bilinear); title('Transformed Image (Bilinear)');
imwrite(ntransformed_image_bilinear,'bipolar.jpg');

