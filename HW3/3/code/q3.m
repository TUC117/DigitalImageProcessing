function save(img, title_text, filename)
     

    imshow(img);

    %title(title_text, 'FontSize', 7, 'Color', 'blue');
    
    %frame = getframe(gcf);
    imwrite(uint8(img), filename);
end


function filtered_img = mean_shift_filter(img, sigma_s, sigma_r, epsilon, max_iters)
    img = double(img);
    [rows, cols] = size(img);  
    filtered_img = img;
    spatial_bandwidth = ceil(3 * sigma_s);

    % Precompute the spatial Gaussian filter
    [X, Y] = meshgrid(-spatial_bandwidth:spatial_bandwidth, -spatial_bandwidth:spatial_bandwidth);
    spatial_gaussian = exp(-(X.^2 + Y.^2) / (2 * sigma_s^2));

    for i = 1:rows
        for j = 1:cols
            prev_val = inf;
            current_pixel_value = img(i, j);
            num_iters = 0;
            
            while abs(current_pixel_value-prev_val) > epsilon && num_iters < max_iters
                prev_val = current_pixel_value;
                i_start = max(i - spatial_bandwidth, 1);
                i_end = min(i + spatial_bandwidth, rows);
                j_start = max(j - spatial_bandwidth, 1);  
                j_end = min(j + spatial_bandwidth, cols);
                region = double(img(i_start:i_end, j_start:j_end));
                [window_row,window_col] = size(region);
                intensity_gauss = exp(-(region - current_pixel_value).^2 / (2 * sigma_r^2));
                index_gauss = spatial_gaussian(1:window_row,1:window_col);
                mul_val = index_gauss .* intensity_gauss;
                mul_val = mul_val / sum(mul_val(:));
                new_pixel_value = sum(sum(mul_val .* region));
                current_pixel_value = new_pixel_value;
                num_iters = num_iters + 1;
            end
            filtered_img(i, j) = current_pixel_value;
        end
    end
    filtered_img = uint8(max(min(filtered_img, 255), 0));
end



function final_img = add_noise(img, sigma)
    final_img = img + sigma * randn(size(img)); 
    final_img = min(max(final_img, 0), 255);
    final_img = uint8(final_img);  
end

% Main code
barbara = imread('barbara256.png');
kodak = imread('kodak24.png');
barbara = double(barbara);
kodak = double(kodak);
sigma_1 = 5;
sigma_2 = 10;

noise_img1 = add_noise(barbara, sigma_1);
noise_img2 = add_noise(kodak, sigma_1);
epsilon = 0.1;
max_iter = 10;
ans_img11 = mean_shift_filter(noise_img1, 2, 2,epsilon,max_iter);
ans_img12 = mean_shift_filter(noise_img1, 15, 3,epsilon,max_iter);
ans_img13 = mean_shift_filter(noise_img1, 3, 15,epsilon,max_iter);
ans_img21 = mean_shift_filter(noise_img2, 2, 2,epsilon,max_iter);
ans_img22 = mean_shift_filter(noise_img2, 15, 3,epsilon,max_iter);
ans_img23 = mean_shift_filter(noise_img2, 3, 15,epsilon,max_iter);
save(noise_img1, 'Noise Barbora 5', 'noicebarbora1.png');
save(noise_img2, 'noice kodak 5', 'noicekodak1.png');
noise_img3 = add_noise(barbara, sigma_2);
noise_img4 = add_noise(kodak, sigma_2);
ans_img31 = mean_shift_filter(noise_img3, 2, 2,epsilon,max_iter);
ans_img32 = mean_shift_filter(noise_img3, 15, 3,epsilon,max_iter);
ans_img33 = mean_shift_filter(noise_img3, 3, 15,epsilon,max_iter);
ans_img41 = mean_shift_filter(noise_img4, 2, 2,epsilon,max_iter);
ans_img42 = mean_shift_filter(noise_img4, 15, 3,epsilon,max_iter);
ans_img43 = mean_shift_filter(noise_img4, 3, 15,epsilon,max_iter);

save(noise_img3, 'Noise Barbora 10', 'noicebarbora2.png');
save(noise_img4, 'noice kodak 10', 'noicekodak2.png');


save(ans_img11, 'Filtered Barbora 2,2,5', 'ans_img11.png');
save(ans_img12, 'Filtered Barbora 15,3,5', 'ans_img12.png');
save(ans_img13, 'Filtered Barbora 3,15,5', 'ans_img13.png');
save(ans_img21, 'Filtered kodak 2,2,5', 'ans_img21.png');
save(ans_img22, 'Filtered kodak 15,3,5', 'ans_img22.png');
save(ans_img23, 'Filtered kodak 3,15,5', 'ans_img23.png');
save(ans_img31, 'Filtered Barbora 2,2,10', 'ans_img31.png');
save(ans_img32, 'Filtered Barbora 15,3,10', 'ans_img32.png');
save(ans_img33, 'Filtered Barbora 3,15,10', 'ans_img33.png');
save(ans_img41, 'Filtered kodak 2,2,10', 'ans_img41.png');
save(ans_img42, 'Filtered kodak 15,3,5', 'ans_img42.png');
save(ans_img43, 'Filtered kodak 3,15,5', 'ans_img43.png');