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

Choosing the likelihood
-----------------------

``fitting.likelihood`` sets the noise model. :math:`\nu` is the forward-model signal, :math:`\sigma` the noise, :math:`w_i` the fitting weight of measurement :math:`i` (noise variance :math:`\sigma^2/w_i`) and :math:`m` the number of measurements with non-zero weight.

.. list-table::
   :widths: 22 30 20 28
   :header-rows: 1

   * - ``fitting.likelihood``
     - Noise model
     - Noise (and amplitude)
     - Intended data
   * - ``'gaussian'`` (default)
     - Gaussian around :math:`\nu` (the legacy likelihood)
     - ``noise`` sampled
     - any data at high SNR; complex or real-valued data
   * - ``'marginal_noise'``
     - Gaussian around :math:`\nu`
     - integrated out (:math:`1/\sigma^2` prior): :math:`\log L = -\tfrac{m}{2}\log R`, :math:`R` the weighted residual sum of squares
     - normalised data (no amplitude parameter), SNR above about 10
   * - ``'marginal_S0noise'``
     - Gaussian around :math:`S_0 g`
     - noise and the linear amplitude ``fitting.S0Param`` integrated out (broad Zellner prior, as in Spinner et al.)
     - data with an amplitude parameter (``'M0'``, ``'S0'``), SNR above about 10
   * - ``'marginal_S0noise_flat'``
     - Gaussian around :math:`S_0 g`
     - as above with a flat amplitude prior
     - as above; better-calibrated intervals with few measurements (e.g. 4 echoes); used in the R2* demos
   * - ``'rician'``
     - exact Rician density of each magnitude measurement
     - ``noise`` sampled, or fixed by a known noise level (see `Known noise`_)
     - single (unaveraged) magnitude measurements at low SNR: late echoes (R2*, R1/R2*, GRE-MWI), high b-values per direction
   * - ``'gaussian_ricianmean'``
     - Gaussian around the Rician mean :math:`E[\,|\nu + \text{noise}|\,]`
     - ``noise`` sampled
     - averaged or combined magnitudes: spherical means (AxCaliberSMT, SANDI, NEXI, mcmicro), averaged repetitions

With a marginal likelihood, the noise (and amplitude) still appear in ``out.posterior``, ``out.median`` etc.: after sampling, each kept sample gets an exact draw from their conditional posterior. The two Rician likelihoods have no marginal form (:math:`\sigma` enters non-linearly), so ``noise`` must be in the model parameters and is sampled, as with ``'gaussian'`` (unless the noise is known, see `Known noise`_). It stays outside the hierarchical prior. Both work with the hierarchical, mixture and spatial priors: only the likelihood term changes.

**Rician likelihood.** A magnitude measurement :math:`y_i = |\nu_i + \sigma_i (n_1 + i n_2)|` with :math:`n_1, n_2 \sim N(0,1)` has the density

.. math::

   p(y_i) = \frac{y_i}{\sigma_i^2} \exp\!\left(-\frac{y_i^2 + \nu_i^2}{2\sigma_i^2}\right) I_0\!\left(\frac{y_i \nu_i}{\sigma_i^2}\right), \qquad \sigma_i^2 = \sigma^2/w_i .

``mcmc_bayes`` evaluates it with the exponentially scaled Bessel function, so it neither overflows nor loses precision at high SNR, where it tends to the Gaussian density. The data must be magnitudes (``y >= 0``; otherwise the error ``mcmc_bayes:ricianNegativeData``). A negative model signal is treated as its magnitude, since the density depends on :math:`\nu` only through :math:`\nu^2` and :math:`y\nu` in an even function.

**Averaged magnitudes are not Rician.** The average of :math:`N` Rician magnitudes (e.g. over the directions of a shell) is not Rician. By the central limit theorem it is close to Gaussian. Its mean is the Rician mean of the single measurements, so it keeps the noise-floor bias, and its spread is about :math:`\sigma_s/\sqrt{N}`, where :math:`\sigma_s` is the noise of one measurement. A Rician density with a single :math:`\sigma` cannot describe this: with :math:`\sigma = \sigma_s` it is far too wide, and with :math:`\sigma = \sigma_s/\sqrt{N}` its floor is far too low. ``'gaussian_ricianmean'`` models such data directly:

.. math::

   y_i \sim N\!\left(E_i, \sigma_i^2\right), \qquad E_i = \sigma_{s,i}\sqrt{\pi/2}\; L_{1/2}\!\left(-\frac{\nu_i^2}{2\sigma_{s,i}^2}\right),

where :math:`L_{1/2}` is the Laguerre function of order 1/2. :math:`E_i \to \sigma_{s,i}\sqrt{\pi/2}` for :math:`\nu_i \to 0` and :math:`E_i \to \nu_i + \sigma_{s,i}^2/(2\nu_i)` at high SNR. By default :math:`\sigma_{s,i} = \sigma_i \sqrt{N_i}`, with :math:`N_i` given by ``fitting.ricianNav``:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``fitting.ricianNav``
     - ``[]`` (= 1)
     - number of magnitude measurements averaged into each measurement: a scalar, or one entry per measurement (e.g. the number of directions of each shell)

``ricianNav`` is only used with ``'gaussian_ricianmean'`` (error ``mcmc_bayes:ricianOption`` otherwise). A known single-measurement noise replaces :math:`\sigma\sqrt{N_i}`, see `Known noise`_. ``out.settings.rician`` records the resolved values.

Known noise
^^^^^^^^^^^

If the noise :math:`\sigma_s` of one magnitude measurement is known (e.g. from a noise scan or the background), give it as a scalar in ``fitting.ricianSigma`` or as a map in ``extraData.noiseSigma``:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``fitting.ricianSigma``
     - scalar :math:`\sigma_s > 0`, in the units of the fitted (normalised) data
   * - ``extraData.noiseSigma``
     - :math:`\sigma_s` as a 3D map ``[x,y,z]`` (or a scalar), positive and finite inside the mask; passed to the model class's ``estimate()`` (``gpuR2starMapping``: optional 4th argument, ``estimate(data, mask, fitting, extraData)``). Not together with ``fitting.ricianSigma``

What it does depends on the likelihood:

* ``'rician'``: :math:`\sigma` is **fixed** at the given value (:math:`\sigma_i^2 = \sigma_s^2/w_i`) and not sampled. ``noise`` is removed from the sampled parameters, and ``out.posterior.noise`` (and ``out.mean.noise`` etc.) equal the given value in every sample, so its ESS and R-hat are not meaningful. Without a known noise, :math:`\sigma` is sampled.
* ``'gaussian_ricianmean'``: it fixes :math:`\sigma_s` inside the Rician mean (instead of :math:`\sigma\sqrt{N_i}`). The spread :math:`\sigma` of the averaged data is still sampled. Not together with ``ricianNav``.

Units: give ``extraData.noiseSigma`` in the units of your **input data** (the noise of one raw magnitude measurement). Every model class divides it by the same per-voxel factor as the data: ``gpuR2starMapping``, ``gpuJointR1R2starMapping``, ``gpuGREMWI`` and the sandbox ``gpuAxonalT2model`` by their global scale factor, ``gpuIVIM`` by the lowest-b signal, and ``gpuAxCaliberSMT``, ``gpuNEXI``, ``gpuSANDI`` and ``gpumcmicro`` (and the sandbox ``gpuMEAxCaliberSMT``) by the b = 0 signal when they compute the spherical mean from full DWI data. The fixed ``noise`` output is then in the fitted (normalised) units, as without a map. If the input data are already normalised (e.g. spherical means divided by b = 0), the class does not normalise them and the map is used as given, so it must be in the same normalised units (:math:`\sigma/S_0`). A scalar ``fitting.ricianSigma`` is never rescaled: it is in the units of the fitted data.

The map works with the automatic GPU memory manager: the model classes slice it with the data for each segment. It must have the spatial size of the full volume; a map of another size is an error (``mcmc_bayes:ricianSigma``). The map is only used with ``fitting.mcmcClass = 'mcmc_bayes'``; otherwise it is ignored with the warning ``mcmc_bayes:noiseSigmaUnused``.

Low-SNR guidance
^^^^^^^^^^^^^^^^

* If every measurement has :math:`\nu/\sigma` above about 10, the Rician bias (about :math:`\sigma^2/(2\nu)`, i.e. at most 5% of :math:`\sigma`) is negligible and the Gaussian and marginal likelihoods are fine.
* If some single magnitude measurements approach the noise floor (late echoes of fast-decaying tissue, high b-values per direction), use ``'rician'``, and provide the noise if you can (``extraData.noiseSigma`` or ``fitting.ricianSigma``). With a sampled :math:`\sigma` at SNR 5-10, the posterior of fast-decaying tissue can be broad and biased upwards (see the limitations below); a known noise level reduces this. Complex or phase-corrected real-valued data are Gaussian; keep ``'gaussian'`` or a marginal likelihood there.
* For direction-averaged or otherwise combined magnitudes, use ``'gaussian_ricianmean'`` with ``ricianNav`` set to the number of averaged measurements, or with a known single-measurement noise. Do not use ``'rician'`` on averaged data.

Limitations of the Rician likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* At very low SNR the voxel-wise posterior can be broad: the data can be explained either by the signal or by a larger :math:`\sigma` and a faster decay into the noise floor. Posterior means can then be biased even though the likelihood is correct; check the credible intervals and posterior medians. A known noise level removes the :math:`\sigma` part of this trade-off, but at SNR 5 a fast decay that reaches the noise floor after a few echoes stays weakly determined.
* ``'gaussian_ricianmean'`` applies the Rician mean to the model's averaged signal. The exact expectation of a direction average is the average of the Rician means of the directional signals, which is larger for anisotropic tissue at low SNR (the Rician mean is convex). The difference is zero for isotropic signals and grows with anisotropy and b-value.
* One :math:`\sigma` is used for all measurements (up to the weights). The spread of an averaged magnitude near the noise floor is smaller than :math:`\sigma_s/\sqrt{N}` (by up to a factor of about 0.65 at zero signal), which the model does not describe.

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
   * - ``.distribution``, ``.nu``
     - ``'normal'``, ``4``
     - ``'t'``: heavy-tailed Student-t population prior with ``nu`` degrees of freedom, see `Student-t prior`_
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

Student-t prior
^^^^^^^^^^^^^^^

With ``distribution = 't'`` the population prior is a multivariate Student-t, :math:`u_i \sim t_\nu(\mu, \Sigma)`, with ``nu`` degrees of freedom (default 4, fixed) and scale matrix :math:`\Sigma`:

.. code-block:: matlab

    fitting.prior.hierarchical = struct('params', {{'R2star'}}, 'distribution', 't', 'nu', 4);

Its heavy tails let a small or atypical group of voxels sit away from the population mean without being pulled towards it as strongly as under the Normal prior. Use it when the tissue has a dominant population plus a minority that is not a well-defined group of its own, for example iron-rich nuclei among grey and white matter, a lesion, or a small tract inside white matter (e.g. larger axons in the corticospinal tract).

How it differs from a mixture (``K = 2``):

* There is no group choice and no label that can switch between groups, so the fit cannot become multimodal over group assignments, and it does not need a second group large enough to estimate its own mean and covariance.
* It does not model a second population: every voxel is shrunk towards the same :math:`\mu`, only less strongly in the tails. If the tissue really has two populations of similar size, use ``K = 2``.

``nu`` sets the tail weight: small ``nu`` (2-4) discounts outlying voxels strongly, ``nu`` of 30 or more is close to the Normal prior. The prior is sampled as a scale mixture: each voxel has a weight :math:`\lambda_i` (prior mean 1) with :math:`u_i \mid \lambda_i \sim N(\mu, \Sigma/\lambda_i)`, sampled exactly after every sweep, and :math:`\mu, \Sigma` are updated from the :math:`\lambda`-weighted statistics (both hyperpriors). With ``fixed = true`` (and in stage 2 of the two-stage scheme) the Student-t density is used directly.

The output adds ``out.hyper.lambda`` (``[x,y,z]``), the posterior mean of :math:`\lambda_i`. Voxels the prior treats as outliers have :math:`\lambda_i` well below 1 (their prior precision is scaled down by :math:`\lambda_i`); typical voxels are near or slightly above 1. ``out.hyper.Sigma`` is the scale matrix; the population covariance is :math:`\nu/(\nu-2)\,\Sigma` for :math:`\nu > 2`. The default MRF weights use the scale matrix. ``distribution = 't'`` requires ``K = 1``.

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
     - population prior: ``.params``, ``.transform``, ``.posterior.mu`` / ``.Sigma`` (samples), ``.mean`` / ``.median``, ``.ess``, ``.rhat``; ``K > 1``: also ``.pi``, ``.membership``, ``.mapLabel``; ``'t'``: also ``.distribution``, ``.nu``, ``.lambda``
   * - ``out.diagnostics``
     - as in :ref:`mcmc-sampler-options`
   * - ``out.settings``
     - resolved options, likelihood, prior and MRF settings, RNG states; ``.rician`` for the two Rician likelihoods
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
* **Noise model:** the marginal likelihoods assume Gaussian noise. For magnitude data at low SNR (roughly below 10) use ``'rician'`` or ``'gaussian_ricianmean'``, see `Choosing the likelihood`_.
* **Weights:** the marginal likelihoods treat the fitting weights as relative precisions of the measurements (correct for per-shell noise, e.g. a different number of directions per shell). Other uses of the weights are not specified yet.
* **Two-stage approximation:** the uncertainty of the population prior is not propagated to stage 2.
