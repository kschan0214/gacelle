.. _api-mcmc_bayes-optimisation:
.. role::  raw-html(raw)
    :format: html

mcmc_bayes.optimisation
=======================

*Experimental.* Metropolis-Hastings sampling with marginal likelihoods, hierarchical and spatial (MRF) priors. See :ref:`mcmc-bayes` for the concepts and recommendations.

With all Bayesian and sampler options absent or at their defaults, the call passes through to :ref:`api-mcmc-optimisation` and the output is bitwise identical. With a free hierarchical prior **and** an MRF prior, it runs :ref:`api-mcmc_bayes-run_two_stage` instead.

Usage
-----

.. code-block:: matlab

    obj = mcmc_bayes;
    out = obj.optimisation( data, mask, weights, parameters, fitting, FWDfunc, varargin);

    % from a model class
    fitting.solver    = 'mcmc';
    fitting.mcmcClass = 'mcmc_bayes';
    out = objGPU.estimate(data, mask, extraData, fitting);

I/O overview
------------

The inputs are those of :ref:`api-mcmc-optimisation`, plus the options below. ``fitting.algorithm`` must be ``'MH'`` on the Bayesian path.

Likelihood
^^^^^^^^^^

.. list-table::
   :widths: 30 18 52
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.likelihood``
     - ``'gaussian'``
     - ``'gaussian'`` | ``'marginal_noise'`` | ``'marginal_S0noise'`` | ``'marginal_S0noise_flat'`` | ``'rician'`` | ``'gaussian_ricianmean'``; see :ref:`mcmc-bayes` (Choosing the likelihood)
   * - ``fitting.S0Param``
     - ``''``
     - name of the linear amplitude parameter in ``fitting.modelParams``; required by, and only allowed with, ``'marginal_S0noise(_flat)'``
   * - ``fitting.ricianNav``
     - ``[]``
     - ``'gaussian_ricianmean'`` only: number of magnitude measurements averaged into each measurement, scalar or one entry per measurement (``> 0``); ``[]`` = 1. The single-measurement noise is ``sigma*sqrt(ricianNav/w)``
   * - ``fitting.ricianSigma``
     - ``[]``
     - known noise of one magnitude measurement, ``'rician'`` and ``'gaussian_ricianmean'``: a scalar ``> 0``, or a per-voxel map with the spatial size of ``mask`` (or ``[1, Nvoxel]``), positive and finite inside the mask. ``'rician'``: the noise is fixed (not sampled; ``out.posterior.noise`` = the given value). ``'gaussian_ricianmean'``: fixes the noise inside the Rician mean, instead of ``ricianNav``. Model classes fill it from ``extraData.noiseSigma`` (map or scalar in input-data units, divided by the class's data normalisation) and slice it per GPU segment. Here it is in the units of the data passed to ``optimisation``

Sampler
^^^^^^^

``fitting.parameterTransform``, ``fitting.updateScheme``, ``fitting.adaptStepSize``, ``fitting.adaptInterval``, ``fitting.adaptTarget``, ``fitting.adaptCovariance`` and ``fitting.overdisp``, as in :ref:`mcmc-sampler-options`.

Hierarchical prior: ``fitting.prior.hierarchical``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A structure, or ``true`` for all defaults.

.. list-table::
   :widths: 30 18 52
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``.params``
     - ``[]``
     - cell array of parameters under the prior; ``[]`` = all sampled parameters except ``noise``
   * - ``.hyperprior``
     - ``'niw'``
     - ``'niw'`` | ``'jeffreys_half'``
   * - ``.m0``
     - ``[]``
     - NIW mean; ``[]`` = mean of the starting values (transformed space)
   * - ``.kappa0``
     - ``1e-3``
     - NIW precision factor of the mean
   * - ``.nu0``
     - ``[]``
     - NIW degrees of freedom; ``[]`` = ``d + 2``
   * - ``.Psi0``
     - ``[]``
     - NIW scale matrix; ``[]`` = ``diag`` of the variance of the starting values (floored)
   * - ``.fixed``
     - ``false``
     - ``true``: use ``.mu`` / ``.Sigma`` (/ ``.pi``) as given
   * - ``.mu``, ``.Sigma``
     - ``[]``
     - fixed population mean ``[d,1]`` (``[d,K]``) and covariance ``[d,d]`` (``[d,d,K]``), transformed space
   * - ``.K``
     - ``1``
     - number of mixture groups
   * - ``.alpha``
     - ``1``
     - Dirichlet concentration of the group weights (``K > 1``)
   * - ``.init``
     - ``'kmeans'``
     - initialisation of the groups (``K > 1``)
   * - ``.pi``
     - ``[]``
     - fixed group weights ``[K,1]`` (``fixed = true``, ``K > 1``)
   * - ``.distribution``
     - ``'normal'``
     - population distribution: ``'normal'`` | ``'t'`` (multivariate Student-t with scale matrix ``Sigma``; ``K = 1`` only)
   * - ``.nu``
     - ``4``
     - degrees of freedom of ``'t'`` (positive, fixed; only with ``distribution = 't'``)
   * - ``.labels``
     - ``[]``
     - fixed segmentation labels: an integer map with the spatial size of ``mask`` (or ``[1, Nvoxel]``, mask order). Every distinct value inside the mask (``0`` included) is one group with its own ``N(mu_k, Sigma_k)``, learned from its voxels only (or fixed: ``.mu [d,K]``, ``.Sigma [d,d,K]``); ``K`` is implied (do not set another ``.K``); no ``.pi``, not with ``'t'``; ``'jeffreys_half'`` needs more than ``2d-1`` voxels per group. Model classes fill it from ``extraData.priorLabels`` (not rescaled, sliced per GPU segment)
   * - ``.labelValues``
     - ``[]``
     - the label value of each group, strictly increasing ``[1,K]``; ``[]`` = the sorted distinct labels inside the mask. Group ``k`` <-> ``labelValues(k)`` (also ``out.hyper.labels``)
   * - ``.subsetFraction``
     - ``1``
     - fraction of voxels for stage 1 (two-stage / ``estimate_hyper_subset`` only; with labels drawn per group, at least ``min(n_k, max(10, 2d))`` voxels of each)
   * - ``.stage1RhatMax``
     - ``1.1``
     - two-stage only: stage 2 runs if the largest stage-1 split-R-hat of ``mu`` (and ``pi`` for a mixture) is at most this value, else error ``mcmc_bayes:stage1NotConverged``; ``Inf`` = no check
   * - ``.maxGPUMemory``
     - ``[]``
     - bytes available for the coupled run; ``[]`` = available GPU memory

Spatial prior: ``fitting.prior.mrf``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A structure, or ``true`` for all defaults. Requires ``fitting.prior.hierarchical``; with free population parameters the two-stage procedure is used.

.. list-table::
   :widths: 30 18 52
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``.potential``
     - ``'l1'``
     - ``'l1'`` | ``'huber'`` | ``'quadratic'``
   * - ``.tau``
     - ``1``
     - temperature (coupling strength ``1/tau``)
   * - ``.W``
     - ``[]``
     - per-parameter weights ``[d,1]``; ``[]`` = ``1./sqrt(diag(Sigma))`` (``K > 1``: of the mixture's marginal covariance; ``'t'``: of the scale matrix; labels: of the pooled covariance of the groups, weights ``n_k/n``)
   * - ``.huberDelta``
     - ``1``
     - Huber threshold in units of ``sqrt(Sigma_pp)`` (``'huber'`` only)
   * - ``.edgeWeights``
     - ``[]``
     - fixed symmetric edge weights ``[Nneighbour, Nvoxel]``; ``[]`` = 1
   * - ``.mode``
     - ``'3d'``
     - ``'3d'`` | ``'2d'`` (in-plane)
   * - ``.radius``
     - ``1``
     - neighbourhood radius
   * - ``.connectivity``
     - ``[]``
     - ``'face'`` (radius 1 only) | ``'full'``; ``[]`` = ``'face'`` for ``'3d'`` radius 1, else ``'full'``
   * - ``.maxGPUMemory``
     - ``[]``
     - bytes for the memory guard
   * - ``.subsetForward``
     - ``true``
     - evaluate the forward model on the active colour only (performance; checked at setup, falls back to full evaluation if the forward model is not column-separable)

Output
^^^^^^

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Field
     - Description
   * - ``out.posterior``, ``out.(metric)``
     - as :ref:`api-mcmc-optimisation`; with a marginal likelihood, ``noise`` (and the amplitude) are exact post-hoc conditional draws
   * - ``out.diagnostics``
     - ``.acceptance``, ``.stepSize``, ``.rhat``, ``.ess``, ``.adaptCovariance`` (see :ref:`mcmc-sampler-options`)
   * - ``out.hyper``
     - hierarchical prior: ``.params``, ``.transform``, ``.posterior.mu``/``.Sigma`` (``.pi``), ``.mean``, ``.median``, ``.ess``, ``.rhat``; ``K > 1``: ``.membership [x,y,z,K]``, ``.mapLabel [x,y,z]``; ``'t'``: ``.distribution``, ``.nu``, ``.lambda [x,y,z]`` (posterior mean of the per-voxel scale weight); labels: ``.K``, ``.labels`` (label value of each group), ``.groupSize [K,1]``, ``mu [d,K]``, ``Sigma [d,d,K]`` per group (no ``.pi`` / ``.membership``)
   * - ``out.settings``
     - resolved options (``.likelihood``, ``.prior``, ``.mrf``, ``.nuisance``, RNG states, ...); ``.rician`` (log-likelihood form, resolved ``ricianNav`` / ``ricianSigma``) for ``'rician'`` and ``'gaussian_ricianmean'``

Errors
^^^^^^

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Identifier
     - Cause
   * - ``mcmc_bayes:unsupportedAlgorithm``
     - a Bayesian option with ``fitting.algorithm`` other than ``'MH'``
   * - ``mcmc_bayes:hierarchicalTransform``
     - a hierarchical parameter whose transform does not map to the whole real line
   * - ``mcmc_bayes:hierarchicalStudentT``
     - ``prior.hierarchical.distribution`` not ``'normal'`` / ``'t'``; ``.nu`` not a positive finite scalar or set without ``'t'``; ``'t'`` with ``K > 1``
   * - ``mcmc_bayes:hierarchicalLabels``
     - ``prior.hierarchical.labels`` not integer-valued and finite inside the mask, of the wrong size, or with a label that is not in ``.labelValues``; ``.labelValues`` not strictly increasing or set without labels; ``.K`` different from the number of groups; with ``.pi`` or ``distribution = 't'``; ``'jeffreys_half'`` with a group of at most ``2d-1`` voxels; (model classes) ``extraData.priorLabels`` without ``fitting.prior.hierarchical``, together with ``prior.hierarchical.labels``, or not of the spatial size of the mask
   * - ``mcmc_bayes:hierarchicalLabelsEmpty`` (warning)
     - free labelled prior with a group that has no voxel in this call (only with a user ``.labelValues``): its ``mu_k, Sigma_k`` are drawn from the hyperprior
   * - ``mcmc_bayes:priorLabelsUnused`` (warning)
     - (model classes) ``extraData.priorLabels`` without ``fitting.mcmcClass = 'mcmc_bayes'``: ignored
   * - ``mcmc_bayes:stage1NotConverged``
     - (two-stage) the stage-1 population prior has not converged: max R-hat of ``mu`` (mixture: also ``pi``) above ``prior.hierarchical.stage1RhatMax``, see :ref:`api-mcmc_bayes-run_two_stage`
   * - ``mcmc_bayes:hierarchicalMemory``, ``mcmc_bayes:mrfMemory``
     - the coupled run does not fit in GPU memory
   * - ``mcmc_bayes:ricianOption``, ``mcmc_bayes:ricianNav``, ``mcmc_bayes:ricianSigma``
     - ``ricianNav`` / ``ricianSigma`` with another likelihood or both set; not positive and finite (inside the mask); ``ricianNav`` neither scalar nor one entry per measurement; a ``ricianSigma`` map that does not match the mask; ``extraData.noiseSigma`` together with ``fitting.ricianSigma``
   * - ``mcmc_bayes:noiseSigmaUnused`` (warning)
     - (model classes) ``extraData.noiseSigma`` without ``fitting.mcmcClass = 'mcmc_bayes'``: ignored
   * - ``mcmc_bayes:ricianNegativeData``
     - ``'rician'`` with negative data (non-zero weight): the Rician density needs magnitudes
   * - ``mcmc:forwardSize``
     - the forward model output is not ``[Nmeas, Nvoxel]`` (e.g. single-slice data passed as ``[x,y,Nmeas]`` instead of ``[x,y,1,Nmeas]``)
   * - ``<Class>:singleSegment``
     - (model classes) a coupled prior while the data would be split into several GPU segments

See also :ref:`mcmc-bayes` and :ref:`tutorial-mcmc_bayes_tutorial`.
