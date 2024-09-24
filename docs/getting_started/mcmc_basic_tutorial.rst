.. _gettingstarted-mcmc_basic_tutorial:
.. role::  raw-html(raw)
    :format: html

MCMC basic tutorial
======================

This tutorial demonstrates an example of how to use the mcmc solver in this package for model parameter estimation. 

Let's say we have a simple monoexponential decay model:

.. math::

    S = S0 \times e^{-R_{2}^{*}t}

In this model, we have two parameters to be estimated: :math:`S0` and :math:`R_{2}^{*}`.

The first thing is to create a function to generate the forward signal. Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_mcmc.m
    :language: matlab

Note that the design of the forward function is slightly stricter for the MCMC solver. The output signal s must have a dimension of Nmeas by Nvoxel.

We can simulate the measurements using this function

.. code-block:: matlab

    % reproducibility
    seed = 5438973; rng(seed); gpurng(seed);

    % set up estimation parameters; must be the same as in FWD function
    model_param = {'S0','R2star','noise'};

    % define number of voxels and SNR
    Nsample = 50;
    SNR     = 100;

    mask        = ones(1,Nsample)>0;
    t           = linspace(0,40e-3,15); 
    % GT
    S0          = 1 + randn(1,Nsample)*0.3;
    R2star      = 30 + 5*randn(1,Nsample);
    % forward signal generation
    pars.(model_param{1}) = S0; 
    pars.(model_param{2}) = R2star;
    S                     = Example_monoexponential_FWD_mcmc(pars,t);

    % realistic signal with certain SNR
    noise   = mean(S0) / SNR;           % estimate noise level
    y       = S + noise*randn(size(S)); % add Gaussian noise
    y       = permute(y, [2 3 4 1]);    % make sure spatial info in the first 3 dimensions while measurements in 4th dimension

.. note::
    In contrast to the askAdam solver, the first 3 dimensions of y are always preserved for spatial information. The measurement/observaetion dimension(s) should be starting from 4th dimension onwards.

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. code-block:: matlab

    pars0.(model_param{1}) = 1 + randn(1,Nsample)*0.3;  % S0
    pars0.(model_param{2}) = 30 + 5*randn(1,Nsample);   % R2*
    pars0.(model_param{3}) = ones(1,Nsample)*0.001;     % noise

2. Set up the model parameters and fitting boundary

.. code-block:: matlab

    % set up fitting algorithm
    fitting                     = [];
    % define model parameter name and fitting boundary
    fitting.model_params        = model_param;
    fitting.lb                  = [0, 0, 0.001];   % lower bound 
    fitting.ub                  = [2, 50, 0.1];  % upper bound

3. Set up optimisation setting

.. code-block:: matlab

    fitting.iteration    = 1e4;
    fitting.algorithm    = 'GW';
    fitting.burnin       = 0.1;     % 10% iterations
    fitting.thinning     = 5;
    fitting.StepSize     = 2;
    fitting.Nwalker      = 50;

4. Define the forward function

.. code-block:: matlab

    modelFWD = @Example_monoexponential_FWD_mcmc;

5. Define fitting weights (optional)

.. code-block:: matlab

    weights = []; % equal weights

6. Start the optimisation

.. code-block:: matlab

    mcmc_obj    = mcmc;
    out         = mcmc_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

7. Plot the estimation results

.. code-block:: matlab

    figure;
    nexttile;scatter(S0,pars0.(model_param{1}));hold on; scatter(S0,out.mean.S0);refline(1);
    xlabel('GT'); ylabel('S0')
    nexttile;scatter(R2star,pars0.(model_param{2}));hold on; scatter(R2star,out.mean.R2star);refline(1)
    xlabel('GT'); ylabel('R2*')
    legend('Start','fitted')

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_mcmc.m>`_.
    