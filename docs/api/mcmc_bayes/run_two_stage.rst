.. _api-mcmc_bayes-run_two_stage:
.. role::  raw-html(raw)
    :format: html

mcmc_bayes.run_two_stage
========================

*Experimental.* Two-stage empirical Bayes for the hierarchical + spatial (MRF) prior. ``mcmc_bayes.optimisation`` (and any model class with ``fitting.mcmcClass = 'mcmc_bayes'``) calls it automatically when ``fitting.prior.hierarchical`` is free and ``fitting.prior.mrf`` is set.

1. **Stage 1**: free hierarchical prior without the MRF, on all voxels or, with ``prior.hierarchical.subsetFraction < 1``, on a random voxel subset (:ref:`api-mcmc_bayes-estimate_hyper_subset`).
2. **Stage 2**: the population parameters are fixed at their stage-1 posterior means; hierarchical + MRF prior with the same likelihood, on all voxels in one call, starting from the stage-1 posterior means (voxels outside the subset start from ``parameters``).

The hyperparameter uncertainty is not propagated to stage 2, so this is an approximation of the full joint posterior.

**Convergence check.** Before stage 2, the largest split-R-hat of the stage-1 ``mu`` (for a ``K > 1`` mixture also ``pi``; for segmentation labels over all groups) must not exceed ``fitting.prior.hierarchical.stage1RhatMax`` (default ``1.1``); otherwise the error ``mcmc_bayes:stage1NotConverged`` is raised, because stage 2 would fix the prior at a mean over chains in different modes. ``stage1RhatMax = Inf`` disables the check. With ``fitting.repetition = 1`` (or fewer than 4 kept samples) there is no R-hat: warning ``mcmc_bayes:stage1NoRhat``, and stage 2 runs.

**Segmentation labels** (``prior.hierarchical.labels``): stage 1 learns ``mu_k, Sigma_k`` per label group, stage 2 fixes them with the same labels and groups (``out.settings.empiricalBayes.labels``); with ``subsetFraction < 1`` the stage-1 subset is drawn per group.

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
