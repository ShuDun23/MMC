function [mpsnr,mssim,mrmse,psnrvector,SSIMvector,RMSEvector]=NNSR_MSI_QA(imagery1, imagery2)
% Evaluates the quality assessment indices for two HSIs.
% Input:
%   imagery1 - the reference HSI data array
%   imagery2 - the target HSI data array
%   NOTE: MSI data array  is a M*N*K array for imagery with M*N spatial
%	pixels, K bands and DYNAMIC RANGE [0,1];
[~,~,p]  = size(imagery1);
psnrvector=zeros(1,p);
for i=1:1:p
    J=imagery1(:,:,i);
    I=imagery2(:,:,i);
    psnrvector(i)=psnr(J,I);
end 
mpsnr = mean(psnrvector);

SSIMvector=zeros(1,p);
for i=1:1:p
    J=imagery1(:,:,i);
    I=imagery2(:,:,i); 
    SSIMvector(i) = ssim(J,I);
end
mssim=mean(SSIMvector);

RMSEvector=zeros(1,p);
for i=1:1:p
    J=imagery1(:,:,i);
    I=imagery2(:,:,i); 
    RMSEvector(i) = rmse(J,I,"all");
end
mrmse=mean(RMSEvector);

end