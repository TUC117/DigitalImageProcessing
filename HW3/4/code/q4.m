
image = zeros(201, 201);
image(1:201, 101) = 255;
F = fft2(image);

F_shifted = fftshift(F);
log_magnitude = log(1 + abs(F_shifted));

figure(1);
imshow(log_magnitude, [min(log_magnitude(:)) max(log_magnitude(:))]);
title('Logarithm of Fourier Magnitude after shifting');
colormap('jet');
colorbar;

imwrite(log_magnitude, 'q4.png');
