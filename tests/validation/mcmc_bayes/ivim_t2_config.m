function cfg = ivim_t2_config()
% IVIM_T2_CONFIG Shared IVIM setting for the Phase 2 validation scripts (T2.1-T2.3).
%
% Units: b in ms/um^2 (= b[s/mm^2]/1000), D and Dstar in um^2/ms (= 1e-3 mm^2/s).
% b-values of the draft plan: 0 10 20 40 80 110 140 170 200 300 400 500 600 700 800 900 s/mm^2.
% Prior: flat in the native box below (D, F, Dstar); S0 and sigma are marginalised
% ('marginal_S0noise', Zellner broad limit). The D and Dstar boxes do not overlap,
% so the two compartments cannot swap labels.
%
% Kwok-Shing Chan @ MGH
% Date created: 26 September 2026
%
cfg.b           = [0 10 20 40 80 110 140 170 200 300 400 500 600 700 800 900] / 1000;

% 5 ground-truth voxels [D, F, Dstar], S0
cfg.truth.D     = [0.8  1.2  0.6  1.5  1.0 ];
cfg.truth.F     = [0.10 0.05 0.20 0.15 0.08];
cfg.truth.Dstar = [15   20   10   30   50  ];
cfg.truth.S0    = [1.00 0.95 1.05 0.90 1.10];

% full model parameters as a model wrapper would define them
cfg.modelParams = {'S0';'D';'F';'Dstar';'noise'};
cfg.lb          = [0;    0.1;  0;    4;   0.001];
cfg.ub          = [2;    3;    0.5;  100; 1    ];
cfg.xStepSize   = [0.01; 0.02; 0.01; 2;   0.001];
cfg.nonlinear   = {'D','F','Dstar'};
end
