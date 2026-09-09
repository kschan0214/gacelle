%% Step01_create_epgx_dictionary_rfphase50.m
%
% Generate EPG-X dictionary for MLP training
%
% This version of dictionary has a wider range
%
% Kwok-Shing Chan
% Date created: & August 2026
% 

addpath(genpath('/path/to/mwi')); % https://github.com/kschan0214/mwi
clear

t = rng('shuffle');

% check parpool, create if not exist
pool = gcp('nocreate'); if isempty(pool); parpool(16); end
%% Define dictionary space
Nsample = 50000;        % number of sample per batch
Nbatch  = 45;           % 2e6 samples in total

% Fix parameters
RFphase     = 50;       % quadratic RF spoiling phase
T1Myelin    = 234e-3;   % T1 of myelin

Nfa         = 40;       % number of flip angle samples across FA_range
Npulse      = 200;      % number of pulses to reach steady state
NCSF        = 0.05;     % CSF signal proportion in the dictionary

% Tissue parameter range
MWF_range   = [0,       0.3];       % [0-30]%
T1IEW_range = [500e-3,  3000e-3];
T1CSF_range = [4000e-3, 6000e-3];
T2IEW_range = [25e-3,   150e-3];
T2CSF_range = [3000e-3, 4000e-3];
kIEWM_range = [0,       10];
% Acquisition range
TR_range    = [25e-3,   85e-3];
FA_range    = [1,       90];

% Other fixed tissue properties
t2_mw                   = 15e-3;   	% ms
kappa_mw                = 0.36; % jung
kappa_iew               = 0.86; % jung
fixed_params.rho_mw     = kappa_mw/kappa_iew;
% fixed_params.E          = 0.02;
% fixed_params.x_i        = -0.1;
% fixed_params.x_a        = -0.1;
% fixed_params.B0         = 3;
% fixed_params.B0dir      = [0 0 1];
fixed_params.t1_mw      = T1Myelin;
fixed_params.freq_shift = 0;

fa  = linspace(min(FA_range),max(FA_range),Nfa);

out_dir = fullfile(pwd,strcat('MCRMWI_EPGX_dictionary_20240927_rfphase',num2str(RFphase),'_t1mylin',num2str(T1Myelin*1e3)));
if ~exist(out_dir,'dir'); mkdir(out_dir); end

%% Set up EPG-X
% EPG-X setting
epgx_params            = [];
epgx_params.isExchange = true;
epgx_params.isEPG      = true;
epgx_params.npulse     = Npulse;
epgx_params.rfphase    = RFphase;
% compute phase cycle
[ phiCycle ] = MCRMWI.RF_phase_cycle( epgx_params.npulse, epgx_params.rfphase );

% compute T matrix
for kfa = 1:Nfa
    T3D_all{kfa} = MCRMWI.PrecomputeT(phiCycle,MCRMWI.d2r(fa(kfa)));
end

%% Simulate dictionary
tic
for kBatch = 1:Nbatch

disp(['#Batch=' num2str(kBatch) ' ...'])

input_parameter     = zeros(Nsample,5);
output_parameter    = zeros(Nsample,Nfa,2, "single");

parfor k = 1:Nsample

% Variable, sequence parameters
tr      = rand_in_range(min(TR_range),max(TR_range));   % s

% Variable, tissue properties
mvf     = rand_in_range(min(MWF_range)/fixed_params.rho_mw,max(MWF_range)/fixed_params.rho_mw);             % MVF
kiewmw  = rand_in_range(min(kIEWM_range),max(kIEWM_range));              % s^-1
if mod(k,Nsample*NCSF) ~= 0
    % 95% normal WM tissue 
    t1_iew  = rand_in_range(min(T1IEW_range),max(T1IEW_range));  % s
    t2_iew  = rand_in_range(min(T2IEW_range),max(T2IEW_range));    % s
else
    % 5% CSF
    t1_iew  = rand_in_range(min(T1CSF_range),max(T1CSF_range));  % s
    t2_iew  = rand_in_range(min(T2CSF_range),max(T2CSF_range));  % s
end

t1x = [t1_iew,fixed_params.t1_mw];
t2x = [t2_iew,t2_mw];

freq_shift = fixed_params.freq_shift;
% 
input_parameter(k,:) = [mvf,t1_iew,kiewmw,tr,t2_iew];

ss_iew  = zeros(Nfa,1);
ss_m    = zeros(Nfa,1);
for j = 1:Nfa
[F0,~,~,~] = MCRMWI.EPGX_GRE_BMsplit_PrecomputedT(T3D_all{j},phiCycle,tr,t1x,t2x,mvf,kiewmw,'delta',freq_shift);

% steady-state
ss_iew(j)	= F0{1,1}(end);
ss_m(j)	    = F0{1,2}(end);

end
output_parameter(k,:,:) = single(cat(3,ss_iew,ss_m));

end

input_parameter = single(input_parameter);

out_fn  = ['EPGX_steadystate_batch-' num2str(kBatch)];
save(fullfile(out_dir,out_fn),'input_parameter','output_parameter');

fprintf('Completed. /n');

end
toc