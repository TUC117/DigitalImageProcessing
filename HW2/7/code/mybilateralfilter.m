function save(img, filename)
    figure;
    imshow(img);
    imwrite(img, filename);
end


function final_img = mybilateralfilter1(img, sigma_s, sigma_r)
    [rows, cols] = size(img);
    final_img = zeros(rows, cols);
    window = ceil(3 * sigma_s);
    [X, Y] = meshgrid(-window:window, -window:window);
    spatial_gaussian = exp(-(X.^2 + Y.^2) / (2 * sigma_s^2));
    for i = 1:rows
        for j = 1:cols
            i_start = max(i - window, 1);
            i_end = min(i + window, rows);
            j_start = max(j - window, 1);  
            j_end = min(j + window, cols);
            region = double(img(i_start:i_end, j_start:j_end));
            intensity_gauss = exp(-(region - double(img(i,j))).^2 / (2 * sigma_r^2));
            index_gauss = spatial_gaussian((i_start:i_end) - i + window + 1, (j_start:j_end) - j + window + 1);
            mul_val = index_gauss .* intensity_gauss;
            mul_val = mul_val / sum(mul_val(:));
            mul_val = mul_val .* region;
            final_img(i,j) = sum(mul_val(:));
        end
    end
    final_img = uint8(final_img);
end


function final_img = add_noise(img, sigma)
    final_img = img + sigma * randn(size(img)); 
    final_img = min(max(final_img, 0), 255);
    final_img = uint8(final_img);  
end

barbara = imread('barbara256.png');
kodak = imread('kodak24.png');
barbara = double(barbara);
kodak = double(kodak);
sigma_1 = 5;
sigma_2 = 10;

noise_img1 = add_noise(barbara, sigma_1);
noise_img2 = add_noise(kodak, sigma_1);

ans_img11 = mybilateralfilter1(noise_img1, 2, 2);
ans_img12 = mybilateralfilter1(noise_img1, 0.1, 0.1);
ans_img13 = mybilateralfilter1(noise_img1, 3, 15);
ans_img21 = mybilateralfilter1(noise_img2, 2, 2);
ans_img22 = mybilateralfilter1(noise_img2, 0.1, 0.1);
ans_img23 = mybilateralfilter1(noise_img2, 3, 15);

save(noise_img1,  'noicebarbora1.png');
save(noise_img2,  'noicekodak1.png');

noise_img3 = add_noise(barbara, sigma_2);
noise_img4 = add_noise(kodak, sigma_2);

ans_img31 = mybilateralfilter1(noise_img3, 2, 2);
ans_img32 = mybilateralfilter1(noise_img3, 0.1, 0.1);
ans_img33 = mybilateralfilter1(noise_img3, 3, 15);
ans_img41 = mybilateralfilter1(noise_img4, 2, 2);
ans_img42 = mybilateralfilter1(noise_img4, 0.1, 0.1);
ans_img43 = mybilateralfilter1(noise_img4, 3, 15);

save(noise_img3, 'noicebarbora2.png');
save(noise_img4,  'noicekodak2.png');


save(ans_img11,  'ans_img11.png');
save(ans_img12, 'ans_img12.png');
save(ans_img13,  'ans_img13.png');
save(ans_img21,  'ans_img21.png');
save(ans_img22,  'ans_img22.png');
save(ans_img23,  'ans_img23.png');
save(ans_img31,  'ans_img31.png');
save(ans_img32,  'ans_img32.png');
save(ans_img33,  'ans_img33.png');
save(ans_img41,  'ans_img41.png');
save(ans_img42,  'ans_img42.png');
save(ans_img43,  'ans_img43.png');