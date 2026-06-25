addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/despot1'))
addpath(genpath('/autofs/space/linen_001/users/kwokshing/tools/mwi'))

te = linspace(0,50e-3,15);
tr = 55e-3;
fa = [5,10,20,30,40,50,70];
b1 = 1;

Amw = 0.15;
Aiw = 0.6;
Aew = 1-0.15-0.6;

t2smw = 10e-3;
t2siw = 64e-3;
t2sew = 48e-3;

t1mw = 234e-3;
t1iew = 1;
kiewmw = 2;

freq_mw = 15;
freq_iw = -2;
freq_ew = 0;

fbg = [1:7]*10;
pini = -1;


DIMWI.isFreqMW  = false;
DIMWI.isFreqIW  = false;
DIMWI.isR2sEW   = false;

EPGX.isExchange = 1;
EPGX.isEPG      = 0;
EPGX.rho_mw     = 0.42;
EPGX.npulse     = 200;
EPGX.rfphase    = 50;
phiCycle = RF_phase_cycle(EPGX.npulse,EPGX.rfphase);
for kfa=1:length(fa)
T3D_all{kfa} = PrecomputeT(phiCycle,d2r(fa(kfa)*b1));
end

EPGX.T3D_all =  T3D_all;

s = mwi_model_2T13T2scc_dimwi(te,tr,fa,b1,Amw,Aiw,Aew,t2smw,t2siw,t2sew,t1mw,t1iew,kiewmw,freq_mw,freq_iw,freq_ew,fbg,pini,DIMWI,EPGX)

%%
kappa_mw                = 0.36; % Jung, NI., myelin water density
kappa_iew               = 0.86; % Jung, NI., intra-/extra-axonal water density
fixed_params.B0     	= 3;    % field strength, in tesla
fixed_params.rho_mw    	= kappa_mw/kappa_iew; % relative myelin water density
fixed_params.E      	= 0.02; % exchange effect in signal phase, in ppm
fixed_params.x_i      	= -0.1; % myelin isotropic susceptibility, in ppm
fixed_params.x_a      	= -0.1; % myelin anisotropic susceptibility, in ppm
fixed_params.B0dir      = B0_dir;
fixed_params.t1_mw      = 234e-3;

objGPU                     = gpuMCRMWI(te,tr,fa,fixed_params);


pars.S0 = Amw + Aiw + Aew;
pars.MWF = Amw;
pars.IWF = Aiw ./ (Aiw + Aew);
pars.R2sMW = 1./t2smw;
pars.R2sIW = 1./t2siw;
pars.R2sEW = 1./t2sew;
pars.R1IEW = 1./t1iew;
pars.kIEWM = kiewmw;
pars.dfreqBKG = permute(fbg(:) / (fixed_params.B0*objGPU.gyro),[2,3,4,5,1]);
pars.dpini = pini;
pars.freqMW = freq_mw / (fixed_params.B0*objGPU.gyro);
pars.freqIW = freq_iw / (fixed_params.B0*objGPU.gyro);

fitting.DIMWI.isFitIWF      = 1;
fitting.DIMWI.isFitFreqMW   = 1;
fitting.DIMWI.isFitFreqIW   = 1;
fitting.DIMWI.isFitR2sEW    = 1;
fitting.isFitExchange       = 1;
fitting.isEPG               = 0;

fitting.isComplex = 1;

ann_epgx_phase          = load('./EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_phase.mat'  ,'dlnet');  ann_epgx_phase.dlnet.alpha  = 0.01;
ann_epgx_magn           = load('./EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_magn.mat','dlnet');    ann_epgx_magn.dlnet.alpha   = 0.01;

extraData = [];
extraData.freqBKG = 0;
extraData.pini = 0;
extraData.b1 = 1;
extraData.ff = ones(size(Amw));
s_dl = objGPU.FWD(pars,fitting,extraData,ann_epgx_phase.dlnet,ann_epgx_magn.dlnet);
s_dl = reshape(s_dl,[numel(te),numel(fa),2]);
% s_dl = gather(extractdata( s_dl(:,:,1) + 1i*s_dl(:,:,2) ));
s_dl = gather(( s_dl(:,:,1) + 1i*s_dl(:,:,2) ));

