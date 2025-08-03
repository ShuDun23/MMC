clear; clc; close all;
rng(1);
addpath('utils');

SamplePath = 'data\multi grayscale image\';
SavePath = 'PATH TO YOUR SAVEPATH';
fileExt = '*.pgm';
files = dir(fullfile(SamplePath,fileExt));

faceH = 112; % height
faceW = 92; % width
len_2 = 10; % length

%% Import images
CompleteCleanData = zeros(faceH,faceW,len_2); % complete data w/o noise
CompleteDirtyData = zeros(faceH,faceW,len_2); % complete data w noise
IncompleteData = zeros(faceH,faceW,len_2); % Incomplete Data
array_Omega = ones(faceH,faceW,len_2); % sampling matrix
array_Omega_c = zeros(faceH,faceW,len_2); % I - sampling matrix

per = 0.8; % observation percentage

for i = 1:len_2
    fileName = strcat(SamplePath,files(i).name);
    image = imread(fileName);
    image = double(image) / 255;
    CompleteCleanData(:,:,i) = image;
    CompleteDirtyData(:,:,i) = CompleteCleanData(:,:,i); % no noise
    % figure;
    % imshow(CompleteDirtyData,'border','tight','initialmagnification','fit');
    % set (gcf,'Position',[0,0,faceW,faceH]);

    % Random missingness
    array_Omega(:,:,i) = binornd( 1, per, [ faceH, faceW ] ); % 每个Omega_k

    % Structural missingness
    j_r1=unidrnd(faceH-5); % generate random row index
    j_r2=unidrnd(faceH-5);
    j_r3=unidrnd(faceH-5);
    j_r4=unidrnd(faceH-5);
    j_r5=unidrnd(faceH-5);
    j_r6=unidrnd(faceH-5);

    j_c1=unidrnd(faceW-5); % generate random col index
    j_c2=unidrnd(faceW-5);
    j_c3=unidrnd(faceW-5);
    j_c4=unidrnd(faceW-5);
    j_c5=unidrnd(faceW-5);
    j_c6=unidrnd(faceW-5);

    array_Omega(:,j_c1:j_c1+1,i) = zeros; % 1 w/ 2 consecutive missing cols
    array_Omega(:,j_c2:j_c2,i) = zeros; % 5 w/ 1 consecutive missing cols
    array_Omega(:,j_c3:j_c3,i) = zeros;
    array_Omega(:,j_c4:j_c4,i) = zeros;
    array_Omega(:,j_c5:j_c5,i) = zeros;
    array_Omega(:,j_c6:j_c6,i) = zeros;

    array_Omega(j_r1:j_r1+1,:,i) = zeros; % 1 w/ 2 consecutive missing rows
    array_Omega(j_r2:j_r2,:,i) = zeros; % 5 w/ 1 consecutive missing rows
    array_Omega(j_r3:j_r3,:,i) = zeros;
    array_Omega(j_r4:j_r4,:,i) = zeros;
    array_Omega(j_r5:j_r5,:,i) = zeros;
    array_Omega(j_r6:j_r6,:,i) = zeros;

    array_Omega_c(:,:,i) = 1 - array_Omega(:,:,i);

    IncompleteData(:,:,i) = CompleteDirtyData(:,:,i).*array_Omega(:,:,i);
    % figure;
    % imshow(IncompleteData,'border','tight','initialmagnification','fit');
    % set (gcf,'Position',[0,0,faceW,faceH]);
end

peaksnr_1 = [];
ssim1 = [];
rmse1 = [];
t_1 = [];

%% MMC

rak = 90; %%%%%%%%%%% Hyperparameter %%%%%%%%%%
r1 = rak;
r2 = r1;
max_out_iter = 50;
tic
[X_1,ERR_iter_1,K_1] = MMC(IncompleteData,array_Omega_c,CompleteCleanData,r1,r2,max_out_iter);
[mpsnr,mssim,mrmse,psnrvector,ssimvector,rmsevector] = NNSR_MSI_QA(CompleteCleanData, X_1);
t_1 = [t_1 toc];
peaksnr_1 = [peaksnr_1 mpsnr];
ssim1 = [ssim1 mssim];
rmse1 = [rmse1 mrmse];

%% Plot

for h = 1:len_2
    t1 = figure;
    imshow(X_1(:,:,h),'border','tight','initialmagnification','fit');
    set (gcf,'Position',[500,500,faceW,faceH]);
    % filename1 = fullfile(SavePath, sprintf('MMC_Reconstruction_%02d.pdf', h));
    % exportgraphics(t1,filename1,'BackgroundColor','none','Resolution',300) % without white space

end

%% Save data
% time = datestr(now, 'yyyy-mm-dd HH-MM-SS');
% filename = sprintf('Name of this file %s.mat',time);
% save( fullfile(SavePath, filename) )
