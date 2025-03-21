function output_img = local_hist_eq(input_img, window_size)
    [rows, cols] = size(input_img);
    pad_size = floor(window_size / 2);
    padded_img = padarray(input_img, [pad_size pad_size], 'symmetric');
    output_img = zeros(rows, cols, 'uint8');
    for i = 1:rows
        for j = 1:cols
            local_window = padded_img(i:i + window_size - 1, j:j + window_size - 1);
            local_window_flat = local_window(:);
            hist_counts = histcounts(local_window_flat, 0:256);
            cdf = cumsum(hist_counts) / numel(local_window_flat);
            central_pixel_value = padded_img(i + pad_size, j + pad_size);
            output_img(i, j) = round(cdf(central_pixel_value + 1) * 255);
        end
    end
end

img1 = imread('LC1.png');
output_img1_7x7 = local_hist_eq(img1, 7);
output_img1_31x31 = local_hist_eq(img1, 31);
output_img1_51x51 = local_hist_eq(img1, 51);
output_img1_71x71 = local_hist_eq(img1, 71);
global_eq_img1 = histeq(img1);
img2 = imread('LC2.jpg');
output_img2_7x7 = local_hist_eq(img2, 7);
output_img2_31x31 = local_hist_eq(img2, 31);
output_img2_51x51 = local_hist_eq(img2, 51);
output_img2_71x71 = local_hist_eq(img2, 71);
global_eq_img2 = histeq(img2);
figure;
imshow(output_img1_7x7);
title('Local Histogram Equalization image 1(7x7)');
imwrite(output_img1_7x7, 'local1_7.jpg');
figure;
imshow(output_img1_31x31);
title('Local Histogram Equalization image 1(31x31)');
imwrite(output_img1_31x31, 'local1_31.jpg');
figure;
imshow(output_img1_51x51);
title('Local Histogram Equalization image 1(51x51)');
imwrite(output_img1_51x51, 'local1_51.jpg');
figure;
imshow(output_img1_71x71);
title('Local Histogram Equalization image 1(71x71)');
imwrite(output_img1_71x71, 'local1_71.jpg');
figure;
imshow(global_eq_img1 );
title('Global Histogram Equalization image 1');
imwrite(global_eq_img1 , 'global1.jpg');
figure;
imshow(output_img2_7x7);
title('Local Histogram Equalization image 2(7x7)');
imwrite(output_img2_7x7, 'local2_7.jpg');
figure;
imshow(output_img2_31x31);
title('Local Histogram Equalization image 2(31x31)');
imwrite(output_img2_31x31, 'local2_31.jpg');
figure;
imshow(output_img2_51x51);
title('Local Histogram Equalization image 2(51x51)');
imwrite(output_img2_51x51, 'local2_51.jpg');
figure;
imshow(output_img2_71x71);
title('Local Histogram Equalization image 2(71x71)');
imwrite(output_img2_71x71, 'local2_71.jpg');
figure;
imshow(global_eq_img2 );
title('Global Histogram Equalization image 2');
imwrite(global_eq_img2 , 'global2.jpg');