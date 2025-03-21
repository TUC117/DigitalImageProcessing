img = imread('barbara256.png');
img = double(img);
[m, n] = size(img);
padded_img = padarray(img, [m, n], 'post');
F = fftshift(fft2(padded_img));
D_values = [40, 80];
[M, N] = size(padded_img);
D = zeros(M, N);
for i = 1:M
    for j = 1:N
        D(i,j) = (i-M/2)^2 + (j-N/2)^2;
    end
end

log_F = log(1 + abs(F));
figure, imshow(log_F, []), title('Log Fourier Transform of Image');
log_F_norm = mat2gray(log_F);
imwrite(log_F_norm, 'log_fourier_image.png');

for D0 = D_values
    
    ideal_filter = zeros(M, N);
    for i = 1:M
      for j = 1:N
         if D(i,j)<= D0^2
             ideal_filter(i,j)=1;
         end
      end
    end
    filtered_img = real(ifft2(ifftshift(F .* ideal_filter)));
    filtered_img = filtered_img(1:m, 1:n);
    
    % Display the filtered image
    figure, imshow(uint8(filtered_img)), title(['Ideal Low Pass Filter with D = ', num2str(D0)]);
    filename = sprintf('ideal_filtered_image_D_%d.png', D0);
    imwrite(uint8(filtered_img), filename);

    % Frequency response (log absolute Fourier format)
    freq_response = log(1 + abs(ideal_filter));
    figure, imshow(freq_response, []), title(['Frequency Response of Ideal Low Pass Filter with D = ', num2str(D0)]);
    freq_response_norm = mat2gray(freq_response);
    filename = sprintf('freq_response_ideal_D_%d.png', D0);
    imwrite(freq_response_norm, filename);

    % Log absolute Fourier transform of the filtered image
    %filtered_img = padarray(filtered_img, [m, n], 'post');
    
    %filtered_F = fftshift(fft2(filtered_img));
    log_filtered_F = log(1 + abs(F .* ideal_filter));
    figure, imshow(log_filtered_F, []), title(['Log Fourier Transform of Filtered Image (D = ', num2str(D0), ')']);
    log_filtered_norm = mat2gray(log_filtered_F);
    filename = sprintf('log_fourier_ideal_filteredimage_D_%d.png', D0);
    imwrite(log_filtered_norm, filename);
end
sigma_values = [40, 80];

for sigma = sigma_values
    gaussian_filter = zeros(M, N);
    for i = 1:M
      for j = 1:N
         gaussian_filter(i,j)=exp(-D(i,j)/(2*(sigma^2)));
      end
    end
    filtered_img = real(ifft2(ifftshift(F .* gaussian_filter)));
    filtered_img = filtered_img(1:m, 1:n);
    
    % Display the filtered image
    figure, imshow(uint8(filtered_img)), title(['Gaussian Low Pass Filter with σ = ', num2str(sigma)]);
    filename = sprintf('gaussian_filtered_image_sigma_%d.png', sigma);
    imwrite(uint8(filtered_img), filename);

    % Frequency response (log absolute Fourier format)
    freq_response = log(1 + abs(gaussian_filter));
    figure, imshow(freq_response, []), title(['Frequency Response of Gaussian Low Pass Filter with σ = ', num2str(sigma)]);
    freq_response_norm = mat2gray(freq_response);
    filename = sprintf('freq_response_Gaussian_sigma_%d.png', sigma);
    imwrite(freq_response_norm, filename);
    % Log absolute Fourier transform of the filtered image
    %filtered_F = fftshift(fft2(filtered_img));
    log_filtered_F = log(1 + abs(F .* gaussian_filter));
    figure, imshow(log_filtered_F, []), title(['Log Fourier Transform of Filtered Image (σ = ', num2str(sigma), ')']);
    log_filtered_norm = mat2gray(log_filtered_F);
    filename = sprintf('log_fourier_gaussian_filteredimage_sigma_%d.png', sigma);
    imwrite(log_filtered_norm, filename);
end
