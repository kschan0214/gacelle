.. _mcmc-sampler-options:

MCMC sampler options
====================

``mcmc.m`` has a set of opt-in options for its Metropolis-Hastings sampler (``fitting.algorithm = 'MH'``) that make the chains mix faster and let you check convergence. They are all off by default: without them ``mcmc.m`` runs exactly as before and returns the same output. The affine-invariant ensemble sampler (``fitting.algorithm = 'ensemble'``) supports ``parameterTransform`` only.

The same options are available in the experimental :ref:`mcmc_bayes <mcmc-bayes>` sampler, which adds Bayesian priors on top of them.

Overview
--------

.. list-table::
   :widths: 26 14 60
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.parameterTransform``
     - ``'linear'``
     - Space in which the chain moves: ``'linear'`` | ``'sigmoid'`` | ``'log'``, or a cell array with one entry per model parameter. See `Parameter transforms`_.
   * - ``fitting.updateScheme``
     - ``'joint'``
     - ``'joint'``: all parameters of a voxel are proposed together (one forward-model evaluation per iteration). ``'componentwise'``: one parameter at a time (one evaluation per parameter).
   * - ``fitting.adaptStepSize``
     - ``false``
     - Adapt the proposal step size of every voxel during burn-in, then freeze it. See `Step-size adaptation`_.
   * - ``fitting.adaptInterval``
     - ``50``
     - Number of iterations between two adaptation steps.
   * - ``fitting.adaptTarget``
     - ``[]``
     - Target acceptance rate; ``[]`` uses 0.234 (``'joint'``) or 0.44 (``'componentwise'``).
   * - ``fitting.adaptCovariance``
     - ``false``
     - Learn a correlated (full-covariance) proposal per voxel during burn-in. Needs ``updateScheme = 'joint'`` and ``adaptStepSize = true``. See `Adaptive covariance`_.
   * - ``fitting.overdisp``
     - ``0``
     - Over-dispersed starting points for chains 2, 3, ... (``fitting.repetition > 1``), needed for a meaningful R-hat. See `Multiple chains and convergence`_.

Setting any of these to a non-default value switches ``mcmc.m`` to the adaptive sampler (``metropolis_hastings_adaptive``), which adds ``out.diagnostics`` and ``out.settings`` to the output. The target distribution is the same as the legacy sampler: a Gaussian likelihood with the sampled ``noise`` parameter and a uniform prior on ``[lb, ub]``.

A recommended starting point:

.. code-block:: matlab

    fitting.algorithm           = 'MH';
    fitting.iteration           = 2e4;
    fitting.burnin              = 1e4;          % adaptation happens during burn-in only
    fitting.repetition          = 4;            % 4 chains for R-hat
    fitting.overdisp            = 0.01;
    fitting.parameterTransform  = 'sigmoid';
    fitting.adaptStepSize       = true;
    fitting.adaptCovariance     = true;

With these settings, the chains of AxCaliberSMT reproduced the legacy sampler's posterior at 3-11 times the effective samples per minute.

Parameter transforms
--------------------

The chain moves in a transformed variable :math:`u = T(x)`, and the Jacobian :math:`\sum_p \log|dx_p/du_p|` is added to the log-target, so the posterior of :math:`x` is unchanged.

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - Transform
     - Mapping
     - Bounds
   * - ``'linear'``
     - :math:`x = u`
     - enforced by rejecting proposals outside ``[lb, ub]`` (legacy behaviour)
   * - ``'sigmoid'``
     - :math:`x = lb + (ub - lb)\,\sigma(u)`
     - needs finite ``lb < ub``; never rejects, so the chain does not stick at a bound
   * - ``'log'``
     - :math:`x = e^{u}`
     - needs finite ``0 <= lb < ub``; the box is enforced by rejection

Starting points are clamped to ``[lb + eps, ub - eps]`` with ``eps = 1e-4*(ub - lb)`` before the transform. A cell array sets the transform per parameter, e.g. ``{'sigmoid','log','linear'}`` in the order of ``fitting.modelParams``.

With the ensemble sampler, the walkers' stretch moves happen in :math:`u` (affine invariance in :math:`u`), the box is enforced in native space, and samples are stored in native units. It is not supported with global parameters (the experimental ``goodman_weare_wglobal_constant``).

Step-size adaptation
--------------------

Every ``adaptInterval`` iterations during burn-in, the proposal scale of each voxel is updated from the acceptance rate of the last window (Robbins-Monro):

.. math::

   \log\sigma \leftarrow \log\sigma + 2\,j^{-0.6}\,(\text{acc}_j - \text{adaptTarget})

With ``'joint'`` one scale per voxel is adapted, with ``'componentwise'`` one per parameter. The scale is frozen at the end of burn-in, so the kept samples come from a standard MH chain with a fixed proposal. Make the burn-in at least ``2*adaptInterval``; the sampler warns otherwise.

Adaptive covariance
-------------------

With ``adaptCovariance = true`` the proposal of each voxel becomes :math:`u' = u + \lambda_i L_i \epsilon`, where :math:`L_i L_i^T` is the running covariance of that voxel's chain (Haario et al. 2001). It helps most when parameters are correlated in the posterior, which is common in multi-compartment models.

* The covariance is accumulated from iteration ``2*adaptInterval + 1`` to the end of burn-in, and first used once at least ``10 x Nparam`` states are available.
* :math:`\lambda_i` starts at :math:`2.38/\sqrt{d}` and is then adapted like the step size.
* A voxel keeps its previous proposal if its covariance cannot be factorised or it accepted too few moves.
* Everything is frozen at the end of burn-in. If the burn-in is too short to switch, the diagonal proposal is used throughout (with a warning).

Multiple chains and convergence
-------------------------------

``fitting.repetition`` runs several chains one after another. By default they all start from the same point, which makes R-hat meaningless. With ``fitting.overdisp > 0``, chain :math:`ii > 1` starts from

.. math::

   u_0 + \text{overdisp}\cdot W \cdot \epsilon, \qquad W_p = T_p(ub_p) - T_p(lb_p), \quad \epsilon \sim N(0, 1)

clamped back into the box. ``overdisp = 0.01`` (1% of the transformed range) is a reasonable default.

The output then reports, per voxel and parameter:

* ``out.diagnostics.rhat.(param)``: split R-hat (Gelman et al. 2013). Values above 1.01 mean the chains have not converged.
* ``out.diagnostics.ess.(param)``: multi-chain bulk effective sample size (Vehtari et al. 2021, without rank normalisation).

Both are ``NaN`` when a chain has fewer than 4 kept samples.

Output
------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Field
     - Description
   * - ``out.diagnostics.acceptance``
     - post-burn-in acceptance rate, ``[x,y,z,Nblock,Nrepetition]`` (``Nblock`` = 1 joint, ``Nparam`` componentwise)
   * - ``out.diagnostics.stepSize.(param)``
     - final proposal scale in transformed space, ``[x,y,z,Nrepetition]``
   * - ``out.diagnostics.rhat.(param)``
     - split R-hat, ``[x,y,z]``
   * - ``out.diagnostics.ess.(param)``
     - bulk ESS, ``[x,y,z]``
   * - ``out.diagnostics.adaptCovariance``
     - final proposal covariance (``.proposalCov``), ``.lambda``, ``.valid``, ``.switchIteration`` (``adaptCovariance`` only)
   * - ``out.settings``
     - resolved options and RNG states

Model classes
-------------

The options pass through a model class's ``estimate()`` like any other ``fitting`` field:

.. code-block:: matlab

    objGPU              = gpuR2starMapping(te);
    fitting             = [];
    fitting.solver      = 'mcmc';
    fitting.parameterTransform = 'sigmoid';
    fitting.adaptStepSize      = true;
    fitting.adaptCovariance    = true;
    fitting.repetition         = 4;
    fitting.overdisp           = 0.01;
    out = objGPU.estimate(img, mask, fitting);

See also :ref:`api-mcmc-optimisation` and :ref:`mcmc-bayes`.
