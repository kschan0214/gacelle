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

EPGX.isExchange = 0;
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


