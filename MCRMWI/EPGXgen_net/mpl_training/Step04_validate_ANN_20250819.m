
addpath(genpath('/path/to/mwi')) % https://github.com/kschan0214/mwi
addpath(genpath('/path/to/despot1')) % https://github.com/kschan0214/despot1
addpath('/path/to/gacelle');addpath_gacelle; % https://github.com/kschan0214/gacelle

ann_epgx_phase = load('../MCRMWI_MLP_EPGX_RFphase50_T1M234_phase','dlnet');  ann_epgx_phase.dlnet.alpha  = 0.01;
ann_epgx_magn  = load('../MCRMWI_MLP_EPGX_RFphase50_T1M234_magn','dlnet');   ann_epgx_magn.dlnet.alpha   = 0.01;
dlnet_phase = ann_epgx_phase.dlnet;
dlnet_magn  = ann_epgx_magn.dlnet;

FA = deg2rad(1:90);
Nfa = numel(FA);

TR = 30e-3;

MVF = 0.3;
t1iew = 1;
r1iew = 1/ t1iew;
t1mw = 234e-3;
r1mw = 1./t1mw;
kiewm = 0;
t2siw = 48e-3;
t2smw = 10e-3;
r2siw = 1./t2siw;

npulse = 200;
rfphase = 50;
phiCycle = RF_phase_cycle(npulse,rfphase);
for kfa=1:length(FA)
    T3D_all{kfa} = PrecomputeT(phiCycle,(FA(kfa)));
end


%% BM
[S0MW_BM, S0IEW_BM]   = (gpuMCRMWI.model_BM_2T1_analytical(TR, FA , MVF,r1iew,r1mw,kiewm));

%% ANN EPG-X
features = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu( repmat(MVF(:),Nfa,1),      1./repmat(r1iew(:),Nfa,1), ...
                                                     repmat(kiewm(:),Nfa,1),    (squeeze(FA) ).', ...    % true_famp = (FA .* extraData.b1(:).').';
                                                     TR, 1./repmat(r2siw(:),Nfa,1), t1mw);
features    = gpuArray( dlarray(features,'CB'));

% phase of long T2 components
S0IEW_phase   = mlp_model_leakyRelu(dlnet_phase.parameters,features,dlnet_phase.alpha);

% signal_steadystate_phase   = reshape(signal_steadystate_phase,size(mvf,1),size(mvf,2),size(mvf,3),1,nFA);
Ss_diff    = mlp_model_leakyRelu(dlnet_magn.parameters,features,dlnet_magn.alpha);

S0MW_ANN  = (S0MW_BM  + Ss_diff(2,:)) ;
S0IEW_ANN = (S0IEW_BM + Ss_diff(1,:)) ;

%% EPG-X real
% EPG-X
t1x = [t1iew, t1mw]; 
t2x = [t2siw,t2smw]; % assuming T2* of iw has similar T2 of long T1 compartment
%         fx = (Amw/EPGX.rho_mw)/(Aiw+Aew+Amw/EPGX.rho_mw); % myelin volume fraction
fx = MVF;  % myelin volume fraction
fs = 0;                     % frequency difference between long and short T1 compartments
% compute saturation factor
SF = zeros(length(FA),2);
for ii=1:length(FA)
    % true flip angle
    alpha = r2d(FA(ii));

    % 2 pools, with exchange
    % start with steady-state signals, longitudinal magnetisation (Mz)
    z1 = Signal_GRE_T1wMono(1-fx, alpha, t1iew, TR)/sind(alpha);
    z2 = Signal_GRE_T1wMono(fx,	alpha, t1mw, TR)/sind(alpha);
    % EPG-X core
    tmp = EPGX_GRE_BMsplit_PrecomputedT(T3D_all{ii},phiCycle,TR,t1x,t2x,fx,kiewm,'delta',fs,'kmax',10,'ss',[z1,z2]);

    % saturation FActors, 1: IEW; 2:Myelin  
    SF(ii,1) = tmp{2}(end);
    SF(ii,2) = tmp{1}(end); 
end
S0MW_EPGX   = SF(:,1).';
S0IEW_EPGX = SF(:,2).';

%%
figure('Position',[437        1706         520         596])
tiledlayout(3,2,'TileSpacing','tight');
co = colororder; alpha = 1;
lw = 2;
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{IEW}')
ylabel('Intensity (a.u.)')
legend('EPG-X','BM','ANN')
grid on
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('difference')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('angle(S_{IEW})')
ylabel('phase (rad)')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX) - S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{M}')
ylabel('Intensity (a.u.)')
xlabel('Flip angle (degree)')
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
xlabel('Flip angle (degree)')

% exportgraphics(gcf,'example1_ANN_validation.png')

%% Example 2

TR = 54e-3;

MVF = 0.2;
t1iew = 1.4;
r1iew = 1/ t1iew;
t1mw = 234e-3;
r1mw = 1./t1mw;
kiewm = 2;
t2siw = 65e-3;
t2smw = 20e-3;
r2siw = 1./t2siw;

npulse = 200;
rfphase = 50;
phiCycle = RF_phase_cycle(npulse,rfphase);
for kfa=1:length(FA)
    T3D_all{kfa} = PrecomputeT(phiCycle,(FA(kfa)));
end


%% BM
[S0MW_BM, S0IEW_BM]   = (gpuMCRMWI.model_BM_2T1_analytical(TR, FA , MVF,r1iew,r1mw,kiewm));

%% ANN EPG-X
features = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu( repmat(MVF(:),Nfa,1),      1./repmat(r1iew(:),Nfa,1), ...
                                                     repmat(kiewm(:),Nfa,1),    (squeeze(FA) ).', ...    % true_famp = (FA .* extraData.b1(:).').';
                                                     TR, 1./repmat(r2siw(:),Nfa,1), t1mw);
features    = gpuArray( dlarray(features,'CB'));

% phase of long T2 components
S0IEW_phase   = mlp_model_leakyRelu(dlnet_phase.parameters,features,dlnet_phase.alpha);

% signal_steadystate_phase   = reshape(signal_steadystate_phase,size(mvf,1),size(mvf,2),size(mvf,3),1,nFA);
Ss_diff    = mlp_model_leakyRelu(dlnet_magn.parameters,features,dlnet_magn.alpha);

S0MW_ANN  = (S0MW_BM  + Ss_diff(2,:)) ;
S0IEW_ANN = (S0IEW_BM + Ss_diff(1,:)) ;

%% EPG-X real
% EPG-X
t1x = [t1iew, t1mw]; 
t2x = [t2siw,t2smw]; % assuming T2* of iw has similar T2 of long T1 compartment
%         fx = (Amw/EPGX.rho_mw)/(Aiw+Aew+Amw/EPGX.rho_mw); % myelin volume fraction
fx = MVF;  % myelin volume fraction
fs = 0;                     % frequency difference between long and short T1 compartments
% compute saturation factor
SF = zeros(length(FA),2);
for ii=1:length(FA)
    % true flip angle
    alpha = r2d(FA(ii));

    % 2 pools, with exchange
    % start with steady-state signals, longitudinal magnetisation (Mz)
    z1 = Signal_GRE_T1wMono(1-fx, alpha, t1iew, TR)/sind(alpha);
    z2 = Signal_GRE_T1wMono(fx,	alpha, t1mw, TR)/sind(alpha);
    % EPG-X core
    tmp = EPGX_GRE_BMsplit_PrecomputedT(T3D_all{ii},phiCycle,TR,t1x,t2x,fx,kiewm,'delta',fs,'kmax',10,'ss',[z1,z2]);

    % saturation FActors, 1: IEW; 2:Myelin  
    SF(ii,1) = tmp{2}(end);
    SF(ii,2) = tmp{1}(end); 
end
S0MW_EPGX   = SF(:,1).';
S0IEW_EPGX = SF(:,2).';

%%
figure('Position',[437        1706         520         596])
tiledlayout(3,2,'TileSpacing','tight');
co = colororder; alpha = 1;
lw = 2;
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{IEW}')
ylabel('Intensity (a.u.)')
% legend('EPG-X','BM','ANN')
grid on
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('difference')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('angle(S_{IEW})')
ylabel('phase (rad)')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX) - S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{M}')
ylabel('Intensity (a.u.)')
xlabel('Flip angle (degree)')
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
xlabel('Flip angle (degree)')

% exportgraphics(gcf,'example2_ANN_validation.png')

%% Example 3

TR = 60e-3;

MVF = 0.05;
t1iew = 2;
r1iew = 1/ t1iew;
t1mw = 234e-3;
r1mw = 1./t1mw;
kiewm = 4;
t2siw = 75e-3;
t2smw = 25e-3;
r2siw = 1./t2siw;

npulse = 200;
rfphase = 50;
phiCycle = RF_phase_cycle(npulse,rfphase);
for kfa=1:length(FA)
    T3D_all{kfa} = PrecomputeT(phiCycle,(FA(kfa)));
end


%% BM
[S0MW_BM, S0IEW_BM]   = (gpuMCRMWI.model_BM_2T1_analytical(TR, FA , MVF,r1iew,r1mw,kiewm));

%% ANN EPG-X
features = feature_preprocess_MCRMWI_MLP_EPGX_leakyrelu( repmat(MVF(:),Nfa,1),      1./repmat(r1iew(:),Nfa,1), ...
                                                     repmat(kiewm(:),Nfa,1),    (squeeze(FA) ).', ...    % true_famp = (FA .* extraData.b1(:).').';
                                                     TR, 1./repmat(r2siw(:),Nfa,1), t1mw);
features    = gpuArray( dlarray(features,'CB'));

% phase of long T2 components
S0IEW_phase   = mlp_model_leakyRelu(dlnet_phase.parameters,features,dlnet_phase.alpha);

% signal_steadystate_phase   = reshape(signal_steadystate_phase,size(mvf,1),size(mvf,2),size(mvf,3),1,nFA);
Ss_diff    = mlp_model_leakyRelu(dlnet_magn.parameters,features,dlnet_magn.alpha);

S0MW_ANN  = (S0MW_BM  + Ss_diff(2,:)) ;
S0IEW_ANN = (S0IEW_BM + Ss_diff(1,:)) ;

%% EPG-X real
% EPG-X
t1x = [t1iew, t1mw]; 
t2x = [t2siw,t2smw]; % assuming T2* of iw has similar T2 of long T1 compartment
%         fx = (Amw/EPGX.rho_mw)/(Aiw+Aew+Amw/EPGX.rho_mw); % myelin volume fraction
fx = MVF;  % myelin volume fraction
fs = 0;                     % frequency difference between long and short T1 compartments
% compute saturation factor
SF = zeros(length(FA),2);
for ii=1:length(FA)
    % true flip angle
    alpha = r2d(FA(ii));

    % 2 pools, with exchange
    % start with steady-state signals, longitudinal magnetisation (Mz)
    z1 = Signal_GRE_T1wMono(1-fx, alpha, t1iew, TR)/sind(alpha);
    z2 = Signal_GRE_T1wMono(fx,	alpha, t1mw, TR)/sind(alpha);
    % EPG-X core
    tmp = EPGX_GRE_BMsplit_PrecomputedT(T3D_all{ii},phiCycle,TR,t1x,t2x,fx,kiewm,'delta',fs,'kmax',10,'ss',[z1,z2]);

    % saturation FActors, 1: IEW; 2:Myelin  
    SF(ii,1) = tmp{2}(end);
    SF(ii,2) = tmp{1}(end); 
end
S0MW_EPGX   = SF(:,1).';
S0IEW_EPGX = SF(:,2).';

%%
figure('Position',[437        1706         520         596])
tiledlayout(3,2,'TileSpacing','tight');
co = colororder; alpha = 1;
lw = 2;
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{IEW}')
ylabel('Intensity (a.u.)')
% legend('EPG-X','BM','ANN')
grid on
nexttile;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0IEW_EPGX) - abs(S0IEW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('difference')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('angle(S_{IEW})')
ylabel('phase (rad)')
grid on
nexttile;
plot(r2d(FA),angle(S0IEW_EPGX) - S0IEW_phase,'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX),'-','LineWidth',lw,'color',[co(1,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);
plot(r2d(FA),abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
title('S_{M}')
ylabel('Intensity (a.u.)')
xlabel('Flip angle (degree)')
grid on
nexttile;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_BM),':','LineWidth',lw,'color',[co(2,:) alpha]);hold on;
plot(r2d(FA),abs(S0MW_EPGX) - abs(S0MW_ANN),'--','LineWidth',lw,'color',[co(3,:) alpha]);
xlim([0 90]);
grid on
xlabel('Flip angle (degree)')

% exportgraphics(gcf,'example3_ANN_validation.png')