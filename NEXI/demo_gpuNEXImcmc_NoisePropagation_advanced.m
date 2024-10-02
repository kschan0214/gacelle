
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/dwi/C2_protocoldesign'));
clear

%% Simulation setting
SNR     = 50;
Ngdir   = 32;
Nsample = 1e3;

bval_unique = [2.3 3.5 4.8 6.5 11.5 17.5];

delta_little    = 6;                                % ms
DELTA_big       = [13, 21, 30];                     % ms
Nshell          = [4, 5, 6];
bval            = [bval_unique(1:Nshell(1)) ...       % D=13 ms
                   bval_unique(1:Nshell(2)) ...       % D=21 ms
                   bval_unique(1:Nshell(3))].';       % D=30 ms
method = 'matlab';          % 'mrtrix' or 'matlab': MRtrix is much faster 

pd = protocoldesign();
% Create b-table
tic;
bvec = pd.dirgen(Ngdir,method);

%% Generate signal based on narrow pulse solution (NEXI)
Delta = [DELTA_big(1)*ones(Nshell(1),1); DELTA_big(2)*ones(Nshell(2),1); DELTA_big(3)*ones(Nshell(3),1);];
delta = delta_little*ones(numel(bval),1);
Nav   = Ngdir*ones(numel(bval),1);

N = Nsample;

NEXIobj = NEXI(bval, Delta);
SMEXobj = SMEXSH(bval, Delta, delta);
intervals = [0.01 0.99  ;   % fa: intra-neurite volume fraction
                0.1 3   ;   % Da: intra-neurite axial diffusivity
                0.1 3   ;   % De/Da: ratio of Da to De
                1 50    ;   % (1-fa)/r: exchange time
              0.5 1.5]  ;   % kappa
pars      = intervals(:,1) + rand(size(intervals,1),N).*diff(intervals,[],2);
% pars(3,:) = pars(2,:) .* pars(3,:);         % De: extra-cellular diffusivity
pars(4,:) = 1./pars(4,:).*(1-pars(1,:));    % r: exchange rate (intra to extra)

tmp = pars;
ind = find(pars(2,:)<pars(3,:));
pars(2,ind) = tmp(3,ind);
pars(3,ind) = tmp(2,ind);

F = [];
lmax        = 8;
S_SH_NEXI   = zeros(numel(bval), Ngdir, N);
theta       = acos(bvec(:,3));
parfor k = 1:N
    kappa       = pars(5,k);    %4.73;
    ang         = NEXIobj.WatsonAng(kappa)/pi*180;
    pl_NEXI     = NEXIobj.WatsonSH(kappa,lmax);
    F           = NEXIobj.NEXIsh(pars(1,k),pars(2,k),pars(3,k),pars(4,k),lmax);
    Si          = NEXIobj.SHconv(F, pl_NEXI, theta);
    S_SH_NEXI(:,:,k) = squeeze(Si);
end

% add b0 assuming 1 b0 in every 16 DWI
S_SH_NEXI(:,end+1:end+Ngdir/16,:) = 1;
% add noise
noise = 1/SNR*randn(size(S_SH_NEXI)) + 1i/SNR*randn(size(S_SH_NEXI));
S_SH_NEXI_n = S_SH_NEXI + noise;

S_SH_NEXI_n = permute(utils.vectorise_NDto2D(permute(real(S_SH_NEXI_n),[3 4 5 2 1])),[1 3 4 2]);

bval_all = repmat(bval(:).',Ngdir,1); bval_all(end+1:end+Ngdir/16,:) = 0; bval_all = bval_all(:);
bvec_all = bvec; bvec_all(end+1:end+Ngdir/16,:) = 0; bvec_all = repmat(bvec_all,numel(bval),1);
DELTA_all = repmat(Delta(:).',Ngdir+Ngdir/16,1); DELTA_all = DELTA_all(:);
delta_all = repmat(delta(:).',Ngdir+Ngdir/16,1); delta_all = delta_all(:);

pl = NEXI.WatsonSHexact(pars(5,:));
 
GT.fa = pars(1,:);
GT.Da = pars(2,:);
GT.De = pars(3,:);
GT.ra = pars(4,:);
GT.p2 = pl(2,:);

%% askadam estimation
fitting                     = [];
fitting.iteration           = 4000;
fitting.initialLearnRate    = 0.001;
fitting.convergenceValue    = 1e-8;
fitting.lossFunction        = 'l1';
fitting.tol                 = 1e-3;
fitting.isdisplay           = false;
fitting.isPrior             = 1;   
fitting.lmax                = 2;  
fitting.randomness          = 0;  

mask = ones(size(S_SH_NEXI_n,1:3),'logical');
extraData = [];
extraData.bval      = bval_all.';
extraData.bvec      = bvec_all.';
extraData.ldelta    = delta_all.';
extraData.BDELTA    = DELTA_all.';

objGPU      = gpuNEXI(bval, Delta);
out         = objGPU.estimate(S_SH_NEXI_n, mask, extraData, fitting);

%% plot result
field = fieldnames(GT);
tiledlayout(1,numel(field)+1);
for k = 1:numel(field)
    nexttile;
    scatter(GT.(field{k}),out.final.(field{k}),5,'filled','MarkerFaceAlpha',.4);
    h = refline(1);
    h.Color = 'k';
    title(field{k});
    xlabel('GT');ylabel('Estimate');
end
nexttile;
scatter((1-GT.fa)./GT.ra,(1-out.final.fa)./out.final.ra,5,'filled','MarkerFaceAlpha',.4);
h = refline(1);
h.Color = 'k';
title('tex');
xlabel('GT');ylabel('Estimate');
