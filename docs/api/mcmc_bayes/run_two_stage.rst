.. _api-mcmc_bayes-run_two_stage:
.. role::  raw-html(raw)
    :format: html

mcmc_bayes.run_two_stage
========================

*Experimental.* Two-stage empirical Bayes for the hierarchical + spatial (MRF) prior. ``mcmc_bayes.optimisation`` (and any model class with ``fitting.mcmcClass = 'mcmc_bayes'``) calls it automatically when ``fitting.prior.hierarchical`` is free and ``fitting.prior.mrf`` is set.

1. **Stage 1**: free hierarchical prior without the MRF, on all voxels or, with ``prior.hierarchical.subsetFraction < 1``, on a random voxel subset (:ref:`api-mcmc_bayes-estimate_hyper_subset`).
2. **Stage 2**: the population parameters are fixed at their stage-1 posterior means; hierarchical + MRF prior with the same likelihood, on all voxels in one call, starting from the stage-1 posterior means (voxels outside the subset start from ``parameters``).

The hyperparameter uncertainty is not propagated to stage 2, so this is an approximation of the full joint posterior.

Usage
-----

.. code-block:: matlab

    obj = mcmc_bayes;
    out = obj.run_two_stage( data, mask, weights, parameters, fitting, FWDfunc, varargin);

I/O overview
------------

Inputs as :ref:`api-mcmc_bayes-optimisation`. ``fitting.prior.hierarchical`` (not fixed) and ``fitting.prior.mrf`` are required. ``fitting.iteration``, ``fitting.repetition`` etc. apply to both stages.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Output
     - Description
   * - ``out``
     - stage-2 output, as :ref:`api-mcmc_bayes-optimisation`
   * - ``out.stage1``
     - stage-1 summary: ``.hyper``, ``.diagnostics``, ``.settings``, ``.voxelIndex`` (linear indices into ``mask``), ``.subsetFraction``
   * - ``out.settings.empiricalBayes``
     - description of the approximation and the fixed population parameters
