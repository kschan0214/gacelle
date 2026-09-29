.. _mcmc-bayes:

Bayesian priors for MCMC (mcmc_bayes)
=====================================

.. warning::
   ``mcmc_bayes`` is **experimental**. Its interface and behaviour may change without notice. Check convergence (see `Diagnostics`_) before interpreting any result.

``mcmc_bayes`` is a subclass of ``mcmc`` that adds Bayesian priors to the Metropolis-Hastings sampler. It reimplements, with corrections, the ideas of Spinner, Federau & Kozerke (2021), *Bayesian inference using hierarchical and spatial priors for intravoxel incoherent motion MR imaging in the brain*, Medical Image Analysis 73:102144 (`doi:10.1016/j.media.2021.102144 <https://doi.org/10.1016/j.media.2021.102144>`_), and makes them available to every GACELLE model that supports MCMC.

It adds three things to the standard voxel-wise fit:

* **Marginal likelihoods**: the noise level (and optionally the signal amplitude S0) is integrated out analytically instead of being sampled.
* **A hierarchical (population) prior, "BSP"**: the voxel parameters are assumed to come from a common population distribution, which is learned from the data at the same time. Poorly determined voxels borrow strength from the rest of the brain.
* **A spatial prior**: a Markov random field (MRF) that favours similar values in neighbouring voxels, fitted as a two-stage empirical Bayes procedure ("BSP + spatial prior").

When to use it
--------------

The priors help most when a parameter is weakly determined by the data of a single voxel. In our semi-synthetic tests:

* **Axon radius/diameter**: the hierarchical prior cut the RMSE by 41-58% (AxCaliberSMT, multi-echo AxCaliberSMT) while keeping 90% credible intervals near their nominal coverage in white and grey matter.
* **Multi-compartment diffusion (SANDI)**: RMSE of Rs, fs, f and De dropped by 34-57%.
* **IVIM**: on the numerical phantom of Spinner et al., the hierarchical prior matched their BSP in white and grey matter.

They can also **shrink atypical tissue towards the population**: iron-rich deep grey matter in R2* maps, CSF partial volume, or lesions. A two-group mixture prior (``K = 2``, see `Mixture prior`_) largely fixes this for tissue that forms its own group. For well-determined parameters (e.g. R2* from many echoes at high SNR) the priors change little.

Quick start with a model class
------------------------------

Every model class that supports MCMC accepts ``fitting.mcmcClass = 'mcmc_bayes'``. Without any of the new options, ``mcmc_bayes`` passes straight through to ``mcmc`` and the output is bitwise identical.

.. code-block:: matlab

    objGPU = gpuAxCaliberSMT(bval, ldelta, BDELTA, D0, Da_fixed, DeL_fixed, Dcsf);

    fitting                     = [];
    fitting.solver              = 'mcmc';
    fitting.mcmcClass           = 'mcmc_bayes';      % experimental Bayesian sampler
    fitting.algorithm           = 'MH';
    fitting.iteration           = 2e4;
    fitting.burnin              = 1e4;
    fitting.thinning            = 10;
    fitting.repetition          = 4;                 % 4 chains for R-hat
    fitting.overdisp            = 0.01;
    fitting.parameterTransform  = 'sigmoid';
    fitting.adaptStepSize       = true;
    fitting.adaptCovariance     = true;
    fitting.likelihood          = 'marginal_noise';  % noise integrated out

    % Demo #1: BSP (hierarchical prior)
    fitting.prior.hierarchical  = struct('params', {{'a','f','fcsf','DeR'}});
    out_bsp = objGPU.estimate(dwi, mask, extraData, fitting);

    % Demo #2: BSP + spatial prior (two-stage empirical Bayes)
    fitting.prior.mrf           = struct('mode', '3d', 'tau', 3);
    objGPU  = gpuAxCaliberSMT(bval, ldelta, BDELTA, D0, Da_fixed, DeL_fixed, Dcsf);   % new object per fit
    out_mrf = objGPU.estimate(dwi, mask, extraData, fitting);

Supported models: ``gpuR2starMapping``, ``gpuJointR1R2starMapping``, ``gpuGREMWI``, ``gpuAxCaliberSMT``, ``gpuNEXI``, ``gpuSANDI``, ``gpumcmicro`` and ``gpuIVIM`` (``gpuMCRMWI`` has no MCMC support). Each of these models, except IVIM, has an in vivo demo ``demo_<class>_invivo_bayes.m`` in its folder; see :ref:`tutorial-mcmc_bayes_tutorial` for a simulated example.

.. important::
   The hierarchical prior (with free population parameters) and the spatial prior couple all voxels, so the whole volume must be fitted in **one GPU call**. If the automatic memory manager would split the data into several segments, ``estimate()`` stops with the error ``<Class>:singleSegment``. Fit a slab of slices or a region of interest, or use a GPU with more memory. The in vivo demos use a central slab of 8 slices.

Likelihood
----------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - ``fitting.likelihood``
     - Description
   * - ``'gaussian'`` (default)
     - The legacy likelihood: Gaussian noise with the ``noise`` parameter sampled.
   * - ``'marginal_noise'``
     - The noise variance is integrated out with a :math:`1/\sigma^2` prior: :math:`\log L = -\tfrac{m}{2}\log R`, with :math:`R` the weighted residual sum of squares and :math:`m` the number of measurements with non-zero weight. Recommended for models whose data are normalised (no amplitude parameter).
   * - ``'marginal_S0noise'``
     - Also integrates out a linear amplitude parameter (``fitting.S0Param``, e.g. ``'M0'`` or ``'S0'``) under a broad Zellner prior, as in Spinner et al. The forward model is evaluated with the amplitude set to 1.
   * - ``'marginal_S0noise_flat'``
     - As above with a flat prior on the amplitude. It gave better-calibrated intervals with few measurements (e.g. 4 echoes), and is used in the R2* demos.

With a marginal likelihood, the noise (and amplitude) still appear in ``out.posterior``, ``out.median`` etc.: after sampling, each kept sample gets an exact draw from their conditional posterior.

Hierarchical prior (BSP)
------------------------

``fitting.prior.hierarchical`` puts a multivariate Normal prior :math:`u_i \sim N(\mu, \Sigma)` on the transformed parameters :math:`u_i` of every voxel :math:`i`. The population mean :math:`\mu` and covariance :math:`\Sigma` are sampled together with the voxel parameters, using exact Gibbs updates after every MH sweep.

.. list-table::
   :widths: 28 16 56
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``.params``
     - ``[]``
     - cell array of the parameters under the prior; ``[]`` = all sampled parameters except ``noise``. Leave out nuisance parameters that vary smoothly in space (e.g. background field offsets).
   * - ``.hyperprior``
     - ``'niw'``
     - prior on :math:`(\mu, \Sigma)`: ``'niw'`` (weak Normal-inverse-Wishart, proper) or ``'jeffreys_half'`` (:math:`p(\Sigma) \propto |\Sigma|^{-1/2}`, as in Spinner et al.)
   * - ``.K``
     - ``1``
     - number of groups of a mixture prior, see `Mixture prior`_
   * - ``.fixed``
     - ``false``
     - ``true``: use the given ``.mu`` and ``.Sigma`` (and ``.pi`` for ``K > 1``) instead of learning them; voxels are then independent and can be segmented
   * - ``.mu``, ``.Sigma``, ``.pi``
     - ``[]``
     - fixed population parameters (transformed space), required with ``fixed = true``
   * - ``.m0``, ``.kappa0``, ``.nu0``, ``.Psi0``
     - data-based
     - NIW hyperparameters; the defaults are weak and centred on the starting values
   * - ``.subsetFraction``
     - ``1``
     - fraction of voxels used to learn the population prior (stage 1 of the two-stage scheme only)

The hierarchical parameters must use a transform that maps to the whole real line: ``'sigmoid'`` (recommended), ``'log'`` with ``lb = 0`` and ``ub = Inf``, or ``'linear'`` with infinite bounds.

Mixture prior
^^^^^^^^^^^^^

With ``K > 1`` the population prior is a mixture of ``K`` Normal distributions with weights :math:`\pi_k`. Each voxel's group is a latent variable, sampled along with everything else, so **no segmentation is needed**. Use it when the tissue is heterogeneous:

* In R2* maps, ``K = 2`` stopped iron-rich deep grey matter from being pulled towards the population (globus pallidus coverage 0.11 with K = 1 to 0.74 with K = 2).
* In multi-echo AxCaliberSMT, ``K = 2`` separated CSF partial-volume voxels from tissue without any labels and restored their credible-interval coverage.

``K = 3`` was not well determined in our tests. ``'jeffreys_half'`` is not allowed with ``K > 1``. The output adds ``out.hyper.membership`` (``[x,y,z,K]``, the posterior probability of each group) and ``out.hyper.mapLabel`` (the most likely group). Groups are ordered by the mean of the first hierarchical parameter.

Spatial prior (BSP + MRF)
-------------------------

``fitting.prior.mrf`` adds a Markov random field on the hierarchical parameters:

.. math::

   \Phi(u) = \frac{1}{\tau} \sum_p W_p \sum_{(i,j)\ \text{neighbours}} \rho(u_i^p - u_j^p)

.. list-table::
   :widths: 22 16 62
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``.tau``
     - ``1``
     - temperature; smaller = stronger smoothing
   * - ``.potential``
     - ``'l1'``
     - :math:`\rho`: ``'l1'`` (edge-preserving), ``'huber'`` or ``'quadratic'``
   * - ``.mode``
     - ``'3d'``
     - ``'3d'`` or ``'2d'`` (in-plane neighbours only)
   * - ``.radius``
     - ``1``
     - neighbourhood radius
   * - ``.connectivity``
     - ``[]``
     - ``'face'`` (6 neighbours in 3D, 4 in 2D, radius 1 only) or ``'full'`` (cube/square); ``[]`` = ``'face'`` for 3D radius 1
   * - ``.W``
     - ``[]``
     - per-parameter weights; ``[]`` = :math:`1/\sqrt{\Sigma_{pp}}` of the population prior

The spatial prior is sampled with chromatic updates: voxels are coloured so that no two neighbours share a colour, and each colour is updated in turn given the others. This keeps the target distribution well defined; the original BayesIVIM code updated all voxels at once with state-dependent weights.

When ``prior.hierarchical`` is free and ``prior.mrf`` is set, the fit runs **two-stage empirical Bayes** automatically (``run_two_stage``):

1. **Stage 1** fits the hierarchical prior without the MRF and estimates :math:`\mu, \Sigma` (optionally on a voxel subset, ``prior.hierarchical.subsetFraction``).
2. **Stage 2** fixes :math:`\mu, \Sigma` at their stage-1 posterior means and fits the hierarchical + MRF prior, starting from the stage-1 estimates.

The uncertainty of :math:`\mu, \Sigma` is not propagated to stage 2. A stage-1 summary is returned in ``out.stage1``.

.. note::
   There is no principled rule for choosing ``tau`` yet. In our semi-synthetic tests ``tau = 1`` over-smoothed (AxCaliberSMT, R2*), ``tau = 2.5-3`` kept the edges and gave a moderate gain over the hierarchical prior alone, and ``tau >= 10`` had almost no effect. The spatial prior also pulls voxels at tissue boundaries (e.g. next to CSF) towards their neighbours, which can reduce credible-interval coverage there.

Diagnostics
-----------

Always check convergence. Run at least 4 chains (``fitting.repetition = 4``) with over-dispersed starts (``fitting.overdisp``):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Field
     - Description
   * - ``out.diagnostics.rhat.(param)``
     - split R-hat of the voxel chains, ``[x,y,z]``; aim for ``<= 1.01``
   * - ``out.diagnostics.ess.(param)``
     - bulk effective sample size, ``[x,y,z]``
   * - ``out.hyper.rhat.mu``, ``out.hyper.rhat.Sigma``
     - R-hat of the population prior
   * - ``out.stage1.hyper.rhat``
     - R-hat of the stage-1 population prior (two-stage)
   * - ``out.diagnostics.acceptance``
     - post-burn-in acceptance rate

.. warning::
   The population parameters (:math:`\mu, \Sigma`) can mix slowly. The IVIM and SANDI population priors did not converge in our tests: R-hat of :math:`\mu` was 1.1-25 for IVIM at 10k iterations and still 1.4-4.2 at 50k, and 1.4-2.5 for SANDI at 40k. The original BayesIVIM code did not converge either. Unconverged population priors mostly affect small, atypical regions such as lesions. Longer chains, ``K = 2`` or a fixed population prior (``fixed = true``) are the current options.

Output
------

In addition to the usual MCMC output (``out.posterior``, ``out.median``, ...; see :ref:`api-mcmc-optimisation`):

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Field
     - Description
   * - ``out.hyper``
     - population prior: ``.params``, ``.transform``, ``.posterior.mu`` / ``.Sigma`` (samples), ``.mean`` / ``.median``, ``.ess``, ``.rhat``; ``K > 1``: also ``.pi``, ``.membership``, ``.mapLabel``
   * - ``out.diagnostics``
     - as in :ref:`mcmc-sampler-options`
   * - ``out.settings``
     - resolved options, likelihood, prior and MRF settings, RNG states
   * - ``out.stage1``
     - two-stage only: stage-1 ``.hyper``, ``.diagnostics``, ``.settings``
   * - ``out.settings.empiricalBayes``
     - two-stage only: description of the approximation and the fixed :math:`\hat\mu, \hat\Sigma`

The population prior is reported in the **transformed** space :math:`u` of each parameter (e.g. logit of the rescaled value for ``'sigmoid'``); ``out.hyper.transform`` describes the mapping.

Direct use
----------

``mcmc_bayes`` can also be called directly with any forward model, like ``mcmc``:

.. code-block:: matlab

    out = mcmc_bayes().optimisation(data, mask, weights, pars0, fitting, @FWDfunc, varargin{:});

See :ref:`api-mcmc_bayes-optimisation`, :ref:`api-mcmc_bayes-run_two_stage` and :ref:`api-mcmc_bayes-estimate_hyper_subset`.

Limitations
-----------

* **Algorithms:** Metropolis-Hastings only; the Bayesian options cannot be combined with the ensemble sampler.
* **Memory:** the coupled priors need a single GPU call, see above.
* **Noise model:** the marginal likelihoods assume Gaussian noise. For magnitude data at low SNR (roughly below 10) the Rician bias is not modelled.
* **Weights:** the marginal likelihoods treat the fitting weights as relative precisions of the measurements (correct for per-shell noise, e.g. a different number of directions per shell). Other uses of the weights are not specified yet.
* **Two-stage approximation:** the uncertainty of the population prior is not propagated to stage 2.
