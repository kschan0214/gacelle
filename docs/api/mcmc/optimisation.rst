.. _api-mcmc-optimisation:
.. role::  raw-html(raw)
    :format: html

mcmc.optmisation
===================

Wrapper function for mcmc.metropolis_hastings and mcmc.goodman_weare.

Usage
-----

.. code-block::

    obj = mcmc;
    out = obj.optimisation( data, mask, weights, parameters, fitting, FWDfunc, varargin);

I/O overview
------------

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | (Unmasked) N-D (imaging) data  , first 3 diemnsions reserved for spatial info (x,y,z)                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | [1 or 3]D signal mask                                                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| weights                   | N-D wieghts, same dimension as 'data' (optional)                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| parameters                | structure variable containing starting points of all model parameters to be estimated                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting                   | structure contains fitting algorithm parameters                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.model_params      | 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.lb                | 1xM numeric variable, fitting lower bound, same order as field 'model_params', e.g. [0.5, 0, 0.001];         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.ub                | 1xM numeric variable, fitting upper bound, same order as field 'model_params', e.g. [2, 1, 0.1];             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.algorithm         | MCMC algorithm, 'MH' (Metropolis-Hastings)|'GW' (Affline-invariant ensemble)                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.iteration         | # MCMC iterations                                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.repetition        | # repetition of MCMC proposal                                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.thinning          | sampling interval between iterations                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.burnin            | iterations to be discarded at the beginning, if >1, the exact number will be used; else iteration*burnin     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.xStepSize         | step size of model parameter in MCMC proposal, same size and order as 'model_params' ('MH' only)             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.StepSize          | step size for 'GW' in MCMC proposal ('GW' only)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.Nwalker           | # random walkers ('GW' only)                                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.metric            | cell variable, metric(s) derived from posterior distribution, 'mean'|'std'|'median'|'iqr' (can be multiple)  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| FWDfunc                   | function handle for forward signal generation; size of the output must match size of 'data'                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| varargin                  | additional input for FWDfunc other than 'parameter' and 'mask' (same order as FWDfunc)                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 

+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| Output                            | Description                                                                                                  |
+===================================+==============================================================================================================+
| out                               | structure contains optimisation result                                                                       |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.posterior                     | structure contains MCMC posterior samples                                                                    |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.posterior.(model_params{k})   | Model parameter MCMC posterior samples, masked and unshaped for memory preservation                          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.{metric}.(model_params{k})    | Posterior statistics chosen in fitting.metric                                                                |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
    'noise' is always required in fitting.model_params.

See also :ref:`gettingstarted-mcmc_basic_tutorial`.
