.. _api-mcmc-goodman_weare:
.. role::  raw-html(raw)
    :format: html

mcmc.goodman_weare
==================

Usage
-----

.. code-block::

    obj = mcmc;
    xPosterior = obj.goodman_weare(y,x0,weights,fitting,modelFWD,varargin)

I/O overview
------------

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| y                         | measurements, [Nmeas, Nvoxels]                                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| x0                        | structure variable containing starting points of all model parameters to be estimated                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| weights                   | N-D wieghts, same dimension as 'data' (optional)                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting                   | structure contains fitting algorithm parameters                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.modelParams      | 1xM cell variable,    name of the model parameters, e.g. {'S0','R2star','noise'};                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.lb                | 1xM numeric variable, fitting lower bound, same order as field 'modelParams', e.g. [0.5, 0, 0.001];         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.ub                | 1xM numeric variable, fitting upper bound, same order as field 'modelParams', e.g. [2, 1, 0.1];             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.iteration         | # MCMC iterations                                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.repetition        | # repetition of MCMC proposal                                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.thinning          | sampling interval between iterations                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.burnin            | iterations to be discarded at the beginning, if >1, the exact number will be used; else iteration*burnin     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.StepSize          | step size for 'GW' in MCMC proposal ('GW' only)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.Nwalker           | # random walkers ('GW' only)                                                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| FWDfunc                   | function handle for forward signal generation; size of the output must match size of 'data'                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| varargin                  | additional input for FWDfunc other than 'parameter' and 'mask' (same order as FWDfunc)                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 

+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| Output                            | Description                                                                                                  |
+===================================+==============================================================================================================+
| xPosterior                        | structure contains MCMC posterior samples                                                                    |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+
| xPosterior.(modelParams{k})      | Model parameter MCMC posterior samples                                                                       |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
    'noise' is always required in fitting.modelParams.

Ensemble update scheme
-----------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``fitting.Ensembleupdate``
     - ``'simultaneous'``
     - ``'simultaneous'``: original behaviour - all walkers proposed/updated in one pass using a single derangement as partners. ``'redblack'``: affine-invariance-correct parallel update (Foreman-Mackey et al. 2013) - splits the ensemble into two complementary halves and updates each half in turn using the *other*, frozen half as fixed anchors for that sub-step.

.. note::
    ``'redblack'`` requires ``fitting.Nwalker >= 2`` (conventionally even,
    and ``>= 2 * numel(fitting.modelParams)`` in practice); an odd
    ``Nwalker`` produces unequal halves and triggers a warning.

Memory management
-------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``fitting.autoMemManage``
     - ``true``
     - See :doc:`../advanced/automatic_memory_management` - segments the input into density-balanced, halo-padded chunks fitted sequentially when the full volume would not fit in available VRAM. Applies to ``goodman_weare`` the same way it does to ``askadam.m``.
   * - ``fitting.segmentOverlap``
     -
     - See :doc:`../advanced/automatic_memory_management`
   * - ``fitting.NSegmentUser``
     -
     - See :doc:`../advanced/automatic_memory_management`

See also :ref:`gettingstarted-mcmc_affineinvariantensemble_tutorial`.
