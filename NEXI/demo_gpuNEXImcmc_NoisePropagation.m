addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/askadam/'));
clear;
%% Simulate data

seed        = 8715;
Nsample     = 1e2;
SNR         = 50;

% get current DWI protocol for simulation
bval_sorted     = [2.3, 3.5, 4.8, 6.5, 2.3, 3.5, 4.8, 6.5, 11.0, 2.3, 3.5, 4.8, 6.5, 11.0, 17.5];
BDELTA_sorted   = [13, 13, 13, 13, 21, 21, 21, 21, 21, 30, 30, 30, 30, 30, 30]; %ms

rng(seed);

objGPU = gpuNEXImcmc(bval_sorted, BDELTA_sorted);

ra_range    = [1/250, 0.3];
tex_range   = [1 50];
fa_range    = [0, 1];
Da_range    = [1.5, 3];
De_range    = [0.5, 1.5];
p2_range    = [0, 1];
% noise_range     = [0.01 0.05];

tex_GT  = single(rand(1,Nsample) * diff(tex_range) + min(tex_range) );
Da_GT   = single(rand(1,Nsample) * diff(Da_range)  + min(Da_range));
fa_GT   = single(rand(1,Nsample) * diff(fa_range)  + min(fa_range));
De_GT   = single(rand(1,Nsample) * diff(De_range)  + min(De_range));
p2_GT   = single(rand(1,Nsample) * diff(p2_range)  + min(p2_range));
ra_GT   = (1-fa_GT) ./ tex_GT; 

lmax = 2;
pars        = [];
pars(1,:)   = fa_GT;
pars(2,:)   = Da_GT;
pars(3,:)   = De_GT;
pars(4,:)   = ra_GT;
pars(5,:)   = p2_GT;
s = objGPU.FWD(single(pars), lmax);

% Let assume Gaussian noise to simplify everything
noiseLv = 1/SNR;
s       = s + randn(size(s)) .* noiseLv;
s       = permute(s,[2 3 4 1]);
mask    = ones(size(s,1:3));

%% MCMC estimation
fitting             = [];
fitting.algorithm   = 'GW';
fitting.Nwalker     = 50;
fitting.StepSize    = 2;
fitting.iteration   = 5e3;
fitting.thinning    = 20;        % Sample every 20 iteration
fitting.method      = 'median';
fitting.burnin      = 0.1;       % 10% burn-in
fitting.lmax        = 2;     
fitting.start       = 'likelihood';     
extraData           = [];

out   = objGPU.estimate(s, mask, extraData, fitting);

tiledlayout(2,6);
nexttile;scatter(fa_GT(:),out.expected.fa(:),10,'filled','MarkerFaceAlpha',.4);xlim([fa_range]);ylim([fa_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(Da_GT(:),out.expected.Da(:),10,'filled','MarkerFaceAlpha',.4);xlim([Da_range]);ylim([Da_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(De_GT(:),out.expected.De(:),10,'filled','MarkerFaceAlpha',.4);xlim([De_range]);ylim([De_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(ra_GT(:),out.expected.ra(:),10,'filled','MarkerFaceAlpha',.4);xlim([ra_range]);ylim([ra_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(p2_GT(:),out.expected.p2(:),10,'filled','MarkerFaceAlpha',.4);xlim([p2_range]);ylim([p2_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(tex_GT(:),(1-out.expected.fa(:))./out.expected.ra(:),10,'filled','MarkerFaceAlpha',.4);xlim([tex_range]);ylim([tex_range]);h = refline(1); h.Color = 'k';

%% MCMC estimation
fitting             = [];
fitting.algorithm   = 'MH';
fitting.iteration   = 2e5;
fitting.thinning    = 20;        % Sample every 20 iteration
fitting.method      = 'median';
fitting.burnin      = 0.1;       % 10% burn-in
fitting.lmax        = 2;     
fitting.start       = 'likelihood';     
extraData           = [];

out_MH   = objGPU.estimate(s, mask, extraData, fitting);

% tiledlayout(1,6);
nexttile;scatter(fa_GT(:),out_MH.expected.fa(:),10,'filled','MarkerFaceAlpha',.4);xlim([fa_range]);ylim([fa_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(Da_GT(:),out_MH.expected.Da(:),10,'filled','MarkerFaceAlpha',.4);xlim([Da_range]);ylim([Da_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(De_GT(:),out_MH.expected.De(:),10,'filled','MarkerFaceAlpha',.4);xlim([De_range]);ylim([De_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(ra_GT(:),out_MH.expected.ra(:),10,'filled','MarkerFaceAlpha',.4);xlim([ra_range]);ylim([ra_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(p2_GT(:),out_MH.expected.p2(:),10,'filled','MarkerFaceAlpha',.4);xlim([p2_range]);ylim([p2_range]);h = refline(1); h.Color = 'k';
nexttile;scatter(tex_GT(:),(1-out_MH.expected.fa(:))./out_MH.expected.ra(:),10,'filled','MarkerFaceAlpha',.4);xlim([tex_range]);ylim([tex_range]);h = refline(1); h.Color = 'k';