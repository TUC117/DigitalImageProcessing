% Part A
% Converting images T1.jpg and T2.jpg to double array as J1 and J2 respectively
J1 = double(imread('T1.jpg'));
J2 = double(imread('T2.jpg'));

% Rotating J2 
angle = 28.5;
J3 = imrotate(J2, angle, 'bilinear', 'crop'); % crop to make size same as prev
figure;
imshow(J3/255);

% Joint histogram function
function joint_histogram = joint_hist(I1, I2, width_bin)
    total_bins = 0:width_bin:255;
    length_of_bin = length(total_bins);
    joint_histogram = zeros(length_of_bin-1, length_of_bin-1);
    for i = 1:numel(I1)
        bin1 = find(total_bins <= I1(i), 1, 'last') - 1;
        bin2 = find(total_bins <= I2(i), 1, 'last') - 1;
        if bin1 > 0 && bin2 > 0
            joint_histogram(bin1, bin2) = joint_histogram(bin1, bin2) + 1;
        end
    end
end

function val = ncc(J1, J4)
    J1c = (J1(:) - mean(J1(:)));
    J4c = (J4(:) - mean(J4(:)));
    val = abs(sum(J1c .* J4c)) / (norm(J1c) * norm(J4c));
end

function [angle_values, ncc_vals, je_vals, qmi_vals] = find_angle(J3, J1)
    angle_values = -45:1:45;

    ncc_vals = zeros(size(angle_values));
    je_vals = zeros(size(angle_values));
    qmi_vals = zeros(size(angle_values));

    for i = 1:length(angle_values)
        angle = angle_values(i);

        % Rotate J3 and the mask J11 using the current angle
        temp = ones(size(J3));
        J4 = imrotate(J3, angle, 'bilinear', 'crop');
        temp = imrotate(temp, angle, 'bilinear', 'crop');
        temp = temp(:);
        %disp(size(temp));
        J4 = double(J4(temp == 1));
        J6 = double(J1(temp == 1));
        % Calculate NCC only for the overlapping regions
        %ncc = sum(sum(J1c .* J4c)) / sqrt(sum(sum(J1c.^2)) * sum(sum(J4c.^2)));
        
        ncc_vals(i) = ncc(J6, J4);

        % Joint entropy JE
        joint_histogram = joint_hist(J6, J4, 10);
        joint_prob = (joint_histogram / sum(joint_histogram(:))) + 1e-12;          
        je = -sum(joint_prob(:) .* log2(joint_prob(:)));
        je_vals(i) = je;

        % Quadratic Mutual Information
        pI1 = sum(joint_prob, 2);
        pI2 = sum(joint_prob, 1);
        temp = joint_prob - (pI1 * pI2);
        qmi = sum(sum(temp.^2));
        qmi_vals(i) = qmi;
    end
end

[angle_values, ncc_vals, je_vals, qmi_vals] = find_angle(J3, J1);

% Plot NCC vs angle
figure;
plot(angle_values, ncc_vals);
xlabel('Angle');
ylabel('NCC values');
title('NCC vs Angle');
saveas(gcf, 'NCC_vs_Theta.pdf');

% Plot JE vs angle
figure;
plot(angle_values, je_vals);
xlabel('Angle');
ylabel('Joint Entropy values');
title('Joint Entropy vs Angle');
saveas(gcf, 'JE_vs_Theta.pdf');

% Plot QMI vs angle
figure;
plot(angle_values, qmi_vals);
xlabel('Angle');
ylabel('Quadratic Mutual Information');
title('Quadratic Mutual Information vs Angle');
saveas(gcf, 'QMI_vs_Theta.pdf');

% Determine optimal angles
[~, best_angle_index] = max(ncc_vals);
ncc_opt_angle = angle_values(best_angle_index);

[~, best_angle_index] = min(je_vals); % Minimum JE indicates optimal angle
je_opt_angle = angle_values(best_angle_index);

[~, best_angle_index] = max(qmi_vals);
qmi_opt_angle = angle_values(best_angle_index);

fprintf('Optimal angle using NCC: %.1f\n', ncc_opt_angle);
fprintf('Optimal angle using JE: %.1f\n', je_opt_angle);
fprintf('Optimal angle using QMI: %.1f\n', qmi_opt_angle);

% Plot joint histogram for the optimal angle based on JE
J4 = imrotate(J3, je_opt_angle, 'bilinear', 'crop');
J4 = imresize(J4, size(J1));
joint_histogram = joint_hist(J1, J4, 10);
figure;
imagesc(joint_histogram);
colorbar;
xlabel('J4');
ylabel('J1');
title('Part e');
