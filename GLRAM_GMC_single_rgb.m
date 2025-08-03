%% singel RGB image
clear;clc;close all

SamplePath1 = 'data\single image\';
SavePath = 'PATH TO YOUR SAVEPATH';
files = fullfile(SamplePath1,'tiger.jpg');
image = imread(files);

faceH = 2170; % height
faceW = 3254; % width
len_2 = 3;

%% Import images
CompleteCleanData = zeros(faceH,faceW,len_2); % complete data w/o noise
CompleteDirtyData = zeros(faceH,faceW,len_2); % complete data w noise
IncompleteData = zeros(faceH,faceW,len_2); % Incomplete Data
array_Omega = ones(faceH,faceW,len_2); % sampling matrix
array_Omega_c = zeros(faceH,faceW,len_2); % I - sampling matrix

per = 0.8; % observation percentage
num_struc = 300;

for i = 1:len_2
    image = im2double(image);
    CompleteCleanData(:,:,i) = image(:,:,i);
    CompleteDirtyData(:,:,i) = CompleteCleanData(:,:,i); % no noise
end

% Random missingness
Omega = binornd( 1, per, [ faceH, faceW ] );

% Structural missingness
j_r = zeros(1, num_struc);
j_c = zeros(1, num_struc);
for k = 1:num_struc
    j_r(k) = unidrnd(faceH - 5); % generate random row index
    j_c(k) = unidrnd(faceW - 5); % generate random col index
end

for k = 1:num_struc
    Omega(:, j_c(k):j_c(k)) = zeros; % missing cols
    Omega(j_r(k):j_r(k), :) = zeros; % missing rows
end

for i = 1:len_2
    array_Omega(:,:,i) = Omega;
    array_Omega_c(:,:,i) = 1 - array_Omega(:,:,i);
    IncompleteData(:,:,i) = CompleteDirtyData(:,:,i).*array_Omega(:,:,i);
end

tori = figure;
imshow(CompleteDirtyData,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,faceW,faceH]);
% filenameori = fullfile(SavePath, sprintf('Groundtruth_%02d.pdf', hh));
% exportgraphics(tori,filenameori,'BackgroundColor','none','Resolution',300) % without white space

tobs = figure;
imshow(IncompleteData,'border','tight','initialmagnification','fit')
set (gcf,'Position',[0,0,faceW,faceH]);
% filenameobs = fullfile(SavePath, sprintf('Observation_%02d.pdf', hh));
% exportgraphics(tobs,filenameobs,'BackgroundColor','none','Resolution',300) % without white space


peaksnr_1 = [];
ssim1 = [];
rmse1 = [];
t_1 = [];

%% MMC

r1 = 2170;
r2 = 3254;
max_out_iter = 50;
tic
[X_1,ERR_iter_1,K_1] = MMC(IncompleteData,array_Omega_c,CompleteCleanData,r1,r2,max_out_iter);
[mpsnr,mssim,mrmse,~,~,~] = NNSR_MSI_QA(CompleteCleanData, X_1);
t_1 = [t_1 toc];
peaksnr_1 = [peaksnr_1 mpsnr];
ssim1 = [ssim1 mssim];
rmse1 = [rmse1 mrmse];

%% Plot

t1 = figure;
imshow(X_1,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,faceW,faceH]);
filename1 = fullfile(SavePath, sprintf('MMC_Reconstruction_%02d.pdf', hh));
exportgraphics(t1,filename1,'BackgroundColor','none','Resolution',300) % without white space

%% Save data
% time = datestr(now, 'yyyy-mm-dd HH-MM-SS');
% filename = sprintf('Name of this file %s.mat',hh,time);
% save( fullfile(SavePath, filename) )
