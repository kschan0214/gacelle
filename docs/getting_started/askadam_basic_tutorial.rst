.. _gettingstarted-askadam_basic_tutorial:
.. role::  raw-html(raw)
    :format: html

askAdam basic tutorial
======================

This tutorial demonstrates an example of how to use the askAdam solver in this package for model parameter estimation. 

Let's say we have a simple monoexponential decay model:

.. math::

    S = S0 \times e^{-R_{2}^{*}t}

In this model, we have two parameters to be estimated: :math:`S0` and :math:`R_{2}^{*}`.

The first thing is to create a function to generate the forward signal. Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam.m
    :language: matlab

We can simulate the measurements using this function

.. code-block:: matlab

    % reproducibility
    seed = 5438973; rng(seed); gpurng(seed);

    % set up estimation parameters; must be the same as in FWD function
    model_param = {'S0','R2star'};

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
    S                     = Example_monoexponential_FWD_askadam(pars,mask,t);

    % realistic signal with certain SNR
    noise   = mean(S0) / SNR;           % estimate noise level
    y       = S + noise*randn(size(S)); % add Gaussian noise

.. note::
    The dimenion and arrangement of y must be the same as the output of the forward function.

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. code-block:: matlab

    pars0.(model_param{1}) = 1 + randn(1,Nsample)*0.3;  % S0
    pars0.(model_param{2}) = 30 + 5*randn(1,Nsample);   % R2*

2. Set up the model parameters and fitting boundary

.. code-block:: matlab

    % set up fitting algorithm
    fitting                     = [];
    % define model parameter name and fitting boundary
    fitting.model_params        = model_param;
    fitting.lb                  = [0, 0];   % lower bound 
    fitting.ub                  = [2, 50];  % upper bound

3. Set up optimisation setting

.. code-block:: matlab

    fitting.iteration           = 4000;
    fitting.initialLearnRate    = 0.001;
    fitting.lossFunction        = 'l1';
    fitting.tol                 = 1e-3;
    fitting.convergenceValue    = 1e-8;
    fitting.convergenceWindow   = 20;
    fitting.isdisplay           = false;

4. Define the forward function

.. code-block:: matlab

    modelFWD = @Example_monoexponential_FWD_askadam;

5. Define fitting weights (optional)

.. code-block:: matlab

    weights = []; % equal weights

6. Start the optimisation

.. code-block:: matlab

    askadam_obj = askadam;
    out         = askadam_obj.optimisation(y,mask,weights,pars0,fitting,modelFWD,t);

7. Plot the estimation results

.. code-block:: matlab

    figure;
    nexttile;scatter(S0,pars0.(model_param{1}));hold on; scatter(S0,out.final.S0);refline(1);
    xlabel('GT'); ylabel('S0')
    nexttile;scatter(R2star,pars0.(model_param{2}));hold on; scatter(R2star,out.final.R2star);refline(1)
    xlabel('GT'); ylabel('R2*')
    legend('Start','fitted')

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_askadam.m>`_.
    