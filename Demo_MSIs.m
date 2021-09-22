
%% =================================================================
clc;
clear;
close all;
addpath(genpath('inc_LFBR'));
addpath(genpath('quality_assess'));
%%
methodname    = {'Observed', 'LFBR'};
Mnum          = length(methodname);
Re_tensor     = cell(Mnum,1);
psnr          = zeros(Mnum,1);
ssim          = zeros(Mnum,1);
time          = zeros(Mnum,1);
%% Load initial data
load('fake_and_real_beers.mat')
 X = img/max(img(:));
%% Sampling with random position
sample_ratio = 0.3 ;

fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
T         = X;
Ndim      = ndims(T);
Nway      = size(T);
rand('seed',2);
Omega=gen_W(size(X),1-sample_ratio);     
F=X.*Omega;
Observed=F;
%%
i  = 1;
Re_tensor{i} = F;
[psnr(i), ssim(i)] = quality_ybz(T*255, Re_tensor{i}*255);
enList = 1;

%% Perform  algorithms
i = i+1;
opts=[];
opts.rr=14;
opts.tol   = 1e-5;
opts.maxit=300;
opts.rho   = 0.1;
fprintf('\n');
disp(['performing ',methodname{i}, ' ... ']);
tic;
 [Re_tensor{i},G,Out]        = inc_LFBR_kmeans_end(F,Omega,opts);
toc; 
[psnr(i), ssim(i)]      = quality_ybz(T*255, Re_tensor{i}*255);
enList = [enList,i];
%% Show result
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %8.8s    %5.4s    %5.4s   \n','method','PSNR', 'SSIM' );
for i = 1:length(enList)
    fprintf(' %8.8s    %5.3f    %5.3f    \n',...
        methodname{enList(i)},psnr(enList(i)), ssim(enList(i)));
end
fprintf('================== Result =====================\n');
figure,
show_figResult(Re_tensor,T,min(T(:)),max(T(:)),methodname,enList,1,prod(Nway(3:end)))
