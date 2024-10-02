.. _supportedmodels-AxCaliberSMT:
.. role::  raw-html(raw)
    :format: html

AxCaliberSMT
===================

Axon diamter mapping based on diffusion MRI (dMRI).

gpuAxCaliberSMT
---------------

AxCaliberSMT with askAdam solver.

Usage
^^^^^

.. code-block::

    obj = gpuAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf, varargin)
    out = obj.estimate(s, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. code-block::
    
    % a     : Axon diameter[um], 
    % f     : neurite fraction (f=fa/(fa+fe)), 
    % fcsf  : CSF fraction
    % DeR   : hindered diffusion diffusivity [um2/ms]
    model_params    = {'a';                   'f';'fcsf';                'DeR'};
    ub              = [ 20;                     1;     1;                    3];
    lb              = [0.1;                     0;     0;                 0.01];
    startpoint      = [1.5925;	0.777777777777778;   0.1;    0.482105263157895];

I/O overview
^^^^^^^^^^^^

``obj = gpuAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| b                         | 1xNshell b-values vector [ms/um2]                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| delta                     | 1xNshell diffusion gradient pulse width vector, aka little delta, same size as 'bval' [ms]                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Delta                     | 1xNshell diffusion time, aka big delta, same size as 'bval' [ms]                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| D0                        | intra-cellular intrinsic diffusivity [um2/ms]                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Da                        | intra-cellular axial diffusivity [um2/ms]                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| DeL                       | extra-cellular axial diffusivity [um2/ms]                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Dcsf                      | CSF diffusivity [um2/ms]                                                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``out = obj.estimate(dwi, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| dwi                       | 4D dMRI data, can be either full acquisition or SMT signal [x,y,z,diffusion]                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata                 | Structure array with additional data (Optional)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bval            | 1D b-values [1xdiffusion], same order as 'dwi' [ms/um2] (Optional, only if 'dwi' is full acquisition)        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bvec            | 2D b-vector [3xdiffusion], same order as 'dwi' (Optional, only if 'dwi' is full acquisition)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.ldelta          | 1D gradient duration [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.BDELTA          | 1D diffusion time [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.optimiser         | Algorithm for parameter update, 'adam' (default) | 'sgdm' | 'rmsprop'                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.isdisplay         | boolean, display optimisation process in graphic plot                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.convergenceValue  | tolerance in loss gradient to stop the optimisation                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.convergenceWindow | # of elements in which 'convergenceValue' is computed                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.iteration         | maximum # of optimisation iterations                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.initialLearnRate  | initial learn rate of Adam optimiser                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.tol               | tolerance in loss                                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.lambda            | regularisation parameter(s)                                                                                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.regmap            | model parameter(s) in which regularisation is applied                                                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.TVmode            | Mode for total variation (TV) regularisation, '2D' | '3D'                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.lossFunction      | loss function, 'L1' | 'L2' | 'huber' | 'mse'                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.isPrior           | Starting point estimated based on likelihood method instead of fix/random location                           |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMT_NoisePropagation.m
    :language: matlab

Example script for real data:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMT_invivoData.m
    :language: matlab


gpuAxCaliberSMTmcmc
-------------------

AxCaliberSMT with MCMC solver.

Usage
^^^^^

.. code-block::

    obj = gpuAxCaliberSMTmcmc(b, delta, Delta, D0, Da, DeL, Dcsf, varargin)
    out = obj.estimate(s, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. code-block::
    
    % a     : Axon diameter[um], 
    % f     : neurite fraction (f=fa/(fa+fe)), 
    % fcsf  : CSF fraction
    % DeR   : hindered diffusion diffusivity [um2/ms]
    % noise : noise level
    model_params    = {'a';                   'f';'fcsf';                'DeR';'noise'};
    ub              = [ 20;                     1;     1;                    3;    0.1];
    lb              = [0.1;                     0;     0;                 0.01;   0.01];
    step            = [0.24875;              0.05;  0.05;   0.0393421052631579;  0.005];
    startpoint      = [1.5925;	0.777777777777778;   0.1;    0.482105263157895;	  0.05];

I/O overview
^^^^^^^^^^^^

``obj = gpuAxCaliberSMTmcmc(b, delta, Delta, D0, Da, DeL, Dcsf);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| bval                      | 1xNshell unique b-values vector [ms/um2]                                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| BDELTA                    | 1xNshell diffusion time, same size as 'bval' [ms]                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``out = obj.estimate(dwi, mask, extradata, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| dwi                       | 4D dMRI data, can be either full acquisition or SMT signal [x,y,z,diffusion]                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata                 | Structure array with additional data (Optional)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bval            | 1D b-values [1xdiffusion], same order as 'dwi' [ms/um2] (Optional, only if 'dwi' is full acquisition)        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bvec            | 2D b-vector [3xdiffusion], same order as 'dwi' (Optional, only if 'dwi' is full acquisition)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.ldelta          | 1D gradient duration [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.BDELTA          | 1D diffusion time [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation                                                               |
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
| fitting.start             | Starting point methods, 'likelihood' | 'default | 1xM parameters array                                       |
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


Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMTmcmc_NoisePropagation.m
    :language: matlab

Example script for real data:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMTmcmc_invivoData.m
    :language: matlab

