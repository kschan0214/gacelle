% This folder contains the scripts used to train a MLP network to substitute EPG-X simulation 
% for MCR-MWI
%
% Kwok-Shing Chan
%
% 
% Step 1: to create a dictionary for training
Step01_create_epgx_dictionary_rfphase50;

% Step 2: train a small MLP to learn the phase of long T1/T2 signal
Step02_train_mlp_epgx_ANN_phase_N2e6

% Stepp 3: train a small MLP to learn the magnitude signal difference between EPG-X and Bloch-McConnell
Step03_train_mlp_epgx_ANN_magn_N2e6

% Step 4: Simple visual validation
Step04_validate_ANN_20250819