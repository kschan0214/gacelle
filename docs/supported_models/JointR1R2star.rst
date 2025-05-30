.. _supportedmodels-JointR1R2star:
.. role::  raw-html(raw)
    :format: html

JointR1R2star
==============

Joint R2-R2* single compartment relaxometry using variable flip angle and multi-echo GRE data

gpuJointR1R2star
----------------

With askadam solver

Usage
^^^^^

.. code-block::

    obj = gpuJointR1R2starMapping(te,tr,fa);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. code-block::
    
    % M0    : Proton density weighted signal
    % R1    : (=1/T1) in s^-1
    % R2star: R2* in s^-1   
    model_params    = {'M0';'R1';'R2star'};
    ub              = [  2;  10;  200];
    lb              = [  0; 0.1;  0.1];
    startpoint      = [   1;   1;  30];

I/O overview
------------

``obj = gpuJointR1R2starMapping(te,tr,fa);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| te                        | 1xNte echo time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| tr                        | repetition time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fa                        | 1xNfa flip angle vector [degree]                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 5D MRI data, [x,y,z,t,fa]                                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.b1              | 3D B1+ map [ratio], [x,y,z]                                                                                  |
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
| fitting.regmap            | model parameter(s) in which regularisation is applied,                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.TVmode            | Mode for total variation (TV) regularisation, '2D' | '3D'                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.lossFunction      | loss function, 'L1' | 'L2' | 'huber' | 'mse'                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.isWeighted        | is cost weighted, true|false, default = true                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.weightMethod      | Weighting method, '1stecho' (default) | 'norm'                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.weightPower       | power order of the weight, default = 2                                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMapping_NoisePropagation.m
    :language: matlab

Example script for real data:

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMapping_RealData.m
    :language: matlab


gpuJointR1R2starmcmc
--------------------

With MCMC solver

Usage
^^^^^

.. code-block::

    obj     = gpuJointR1R2starMappingmcmc(te,tr,fa);
    [out]   = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. code-block::
    
    % M0    : Proton density weighted signal
    % R1    : (=1/T1) in s^-1
    % R2star: R2* in s^-1   
    model_params    = {'M0';'R1';'R2star';'noise'};
    ub              = [   2;  10;     200;    0.1];
    lb              = [   0; 0.1;     0.1;  0.001];
    startpoint      = [   1;   1;      30;   0.05];
    step            = [0.01;0.01;       1;  0.005];

I/O overview
------------

``obj = gpuJointR1R2starMappingmcmc(te,tr,fa);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| te                        | 1xNte echo time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| tr                        | repetition time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fa                        | 1xNfa flip angle vector [degree]                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 5D MRI data, [x,y,z,t,fa]                                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.b1              | 3D B1+ map [ratio], [x,y,z]                                                                                  |
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
| fitting.isWeighted        | is cost weighted, true|false, default = true                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.weightMethod      | Weighting method, '1stecho' (default) | 'norm'                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.weightPower       | power order of the weight, default = 2                                                                       |
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

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMappingmcmc_NoisePropagation.m
    :language: matlab

Example script for real data:

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMappingmcmc_RealData.m
    :language: matlab
