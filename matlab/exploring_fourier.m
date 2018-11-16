clear all; clc;

fft_samples = 256;

data_dirA = '/media/hdd1/Datasets/casia/images';
data_dirB = '/media/hdd2/Datasets/ILSVRC2012/images/centerpatch';
filenamesA_txt = '/media/hdd1/Datasets/casia/filenames.txt';
filenamesB_txt = '/media/hdd2/Datasets/ILSVRC2012/filenames/training_filenames.txt';

fidA = fopen(filenamesA_txt);
fidB = fopen(filenamesB_txt);
filenamesA = textscan(fidA, '%s');
filenamesA = filenamesA{1};
filenamesB = textscan(fidB, '%s');
filenamesB = filenamesB{1};
fclose(fidA);
fclose(fidB);

nA = 10000;
nB = 0000;

fnamesA_smp = filenamesA(1:nA);  % sequentially to only have few labels
fnamesB_smp = filenamesB(randperm(length(filenamesB), nB)); % randperm to get variety

% take Fourier transforms of 0-mean images
fftA = complex(zeros(fft_samples, fft_samples, nA), 0);
fftB = complex(zeros(fft_samples, fft_samples, nB), 0);
% FFT of set A images
for i = 1:nA
    if mod(i, 100) == 0
        fprintf('Set A, image %d out of %d\n', i, nA);
    end
    im = imresize(imread(fullfile(data_dirA, fnamesA_smp{i})), [256 256]);
    im = im2double(im(:,:,1));
    im = im - mean(im(:));
    fftA(:,:,i) = fftshift(fft2(im));
end
% FFT of set B images
for i = 1:nB
    if mod(i, 100) == 0
        fprintf('Set B, image %d out of %d\n', i, nA);
    end
    im = imresize(imread(fullfile(data_dirB, fnamesB_smp{i})), [256 256]);
    im = im2double(im(:,:,1));
    im = im - mean(im(:));
    fftB(:,:,i) = fftshift(fft2(im));
end

% Observe FFT statistics
mean_absA = mean(abs(fftA), 3);
var_absA = var(abs(fftA), 0, 3);
mean_absB = mean(abs(fftB), 3);
var_absB = var(abs(fftB), 0, 3);

%% Online statistics
%   In the previous case, all FFTs are saved first before calculating
%   statistics. This may take a while, so instead 

clear all; clc;

imsize = 256;
fft_samples = 1024;

data_dirA = '/media/hdd1/Datasets/casia/images';
data_dirB = '/media/hdd2/Datasets/ILSVRC2012/images/centerpatch';
filenamesA_txt = '/media/hdd1/Datasets/casia/filenames.txt';
filenamesB_txt = '/media/hdd2/Datasets/ILSVRC2012/filenames/training_filenames.txt';

fidA = fopen(filenamesA_txt);
fidB = fopen(filenamesB_txt);
filenamesA = textscan(fidA, '%s');
filenamesA = filenamesA{1};
filenamesB = textscan(fidB, '%s');
filenamesB = filenamesB{1};
fclose(fidA);
fclose(fidB);

nA = 10000;
nB = 100000;

fnamesA_smp = filenamesA(1:nA);  % sequentially to only have few labels
fnamesB_smp = filenamesB(randperm(length(filenamesB), nB)); % randperm to get variety

% take Fourier transforms of 0-mean images
csum_absA = zeros(fft_samples, fft_samples);
csum2_absA = zeros(fft_samples, fft_samples);
csum_angA = zeros(fft_samples, fft_samples);
csum2_angA = zeros(fft_samples, fft_samples);
csum_absB = zeros(fft_samples, fft_samples);
csum2_absB = zeros(fft_samples, fft_samples);
csum_angB = zeros(fft_samples, fft_samples);
csum2_angB = zeros(fft_samples, fft_samples);
% FFT of set A images
for i = 1:nA
    if mod(i, 100) == 0
        fprintf('Set A, image %d out of %d\n', i, nA);
    end
    im = imresize(imread(fullfile(data_dirA, fnamesA_smp{i})), [imsize imsize]);
    im = im2double(im(:,:,1));
    im = im - mean(im(:));
    fftA = fftshift(fft2(im, fft_samples, fft_samples));
    csum_absA = csum_absA + abs(fftA);
    csum2_absA = csum2_absA + abs(fftA).^2;
    csum_angA = csum_angA + angle(fftA);
    csum2_angA = csum2_angA + angle(fftA).^2;
end
% FFT of set B images
for i = 1:nB
    if mod(i, 100) == 0
        fprintf('Set B, image %d out of %d\n', i, nB);
    end
    im = imresize(imread(fullfile(data_dirB, fnamesB_smp{i})), [imsize imsize]);
    im = im2double(im(:,:,1));
    im = im - mean(im(:));
    fftB = fftshift(fft2(im, fft_samples, fft_samples));
    csum_absB = csum_absB + abs(fftB);
    csum2_absB = csum2_absB + abs(fftB).^2;
    csum_angB = csum_angB + angle(fftB);
    csum2_angB = csum2_angB + angle(fftB).^2;
end

% Observe FFT statistics
mean_absA = csum_absA / nA;
mean_angA = csum_angA / nA;
var_absA = (1/(nA-1)) * (csum2_absA - 2*nA*mean_absA*csum_absA + nA*mean_absA^2);
var_angA = (1/(nA-1)) * (csum2_angA - 2*nA*mean_angA*csum_angA + nA*mean_angA^2);
mean_absB = csum_absB / nB;
mean_angB = csum_angB / nB;
var_absB = (1/(nB-1)) * (csum2_absB - 2*nB*mean_absB*csum_absB + nB*mean_absB^2);
var_angB = (1/(nB-1)) * (csum2_angB - 2*nB*mean_angB*csum_angB + nB*mean_angB^2);

