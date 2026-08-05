.. _askadam-convergence:

askAdam.m Convergence Options
==============================

This page describes the convergence and stopping criteria available in ``askadam.m`` (v1.1). These options extend the basic ``fitting.convergenceValue`` / ``fitting.iteration`` controls with more robust alternatives designed for whole-volume qMRI fitting, where the aggregate loss can be dominated by a minority of poorly fitting voxels.

All options are set as fields of the ``fitting`` structure before calling ``obj.optimisation(...)``. All new fields default to sensible values — existing scripts require no changes.

Overview of Stopping Criteria
-------------------------------

``askadam.m`` stops when **any one** of the following conditions is met:

1. **Loss threshold**: the total loss drops below ``fitting.tol``.
2. **Loss convergence**: the convergence signal (either linear slope or EMA-based) is below ``fitting.convergenceValue`` for ``fitting.patienceConvergence`` consecutive checks.
3. **Step norm convergence**: the relative change in parameter values between iterations is below ``fitting.convergenceStepTol`` for ``fitting.patienceStep`` checks.
4. **Gradient norm convergence**: the gradient norm is below ``fitting.convergenceGradTol`` for ``fitting.patienceGrad`` checks.
5. **Maximum iterations**: ``fitting.iteration`` is reached.

Criteria 3 and 4 are independent of ``fitting.robustConvergence``. Criterion 2 is affected by both ``fitting.convergenceModel`` and ``fitting.robustConvergence``.

Convergence Model
------------------

``fitting.convergenceModel`` controls how the convergence signal for the loss is computed at each iteration.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.convergenceModel``
     - ``'ema'``
     - ``'linear'``: slope of loss over the last ``convergenceWindow`` iterations. ``'ema'``: relative change in the exponential moving average (EMA) of the loss.
   * - ``fitting.convergenceWindow``
     - ``20``
     - Number of iterations over which the slope is computed. Used only when ``convergenceModel = 'linear'``.
   * - ``fitting.emaDecay``
     - ``0.95``
     - EMA decay factor. Higher values give a slower-responding, smoother signal. Used only when ``convergenceModel = 'ema'``.
   * - ``fitting.convergenceValue``
     - ``1e-6``
     - Threshold applied to the convergence signal (slope or EMA relative change). Optimisation stops when the signal is below this value for ``patienceConvergence`` consecutive checks.
   * - ``fitting.patienceConvergence``
     - ``5``
     - Number of consecutive checks below ``convergenceValue`` required before stopping.

**Formalism**

*Linear model.* The loss values from the last ``convergenceWindow`` iterations are held in a buffer :math:`\{L_1, L_2, \dots, L_N\}` (:math:`N =` ``convergenceWindow``). A first-order polynomial is fitted to this buffer by ordinary least squares:

.. math::

   L_i \approx m \cdot i + c, \qquad i = 1, \dots, N

Concretely, with design matrix :math:`A = [\,(1{:}N)^\top,\ \mathbf{1}\,]`, the slope and intercept are obtained from :math:`[m, c]^\top = A^{\dagger} \mathbf{L}` (least-squares solve). The convergence signal is the **negative slope**:

.. math::

   \text{convergence} = -m

so that a positive value corresponds to a decreasing loss (improvement), matching the sign convention described for ``fitting.convergenceValue`` (positive = decreasing loss). Optimisation is considered converged once this value drops below ``convergenceValue`` for ``patienceConvergence`` consecutive checks. Because :math:`m` has the same units as the loss itself, this signal is **not scale-invariant**: ``convergenceValue`` must be re-tuned if the loss magnitude changes substantially (e.g. switching loss functions, changing regularisation weight, or changing data scaling).

*EMA model.* Rather than fitting a window of past losses, an exponential moving average of the loss is updated incrementally at every iteration:

.. math::

   \text{EMA}_t = \delta \cdot \text{EMA}_{t-1} + (1 - \delta) \cdot L_t

where :math:`\delta` is ``fitting.emaDecay`` and :math:`L_t` is the current iteration's loss. The convergence signal is the **relative change** in the EMA between consecutive iterations:

.. math::

   \text{convergence} = \frac{\left| \text{EMA}_t - \text{EMA}_{t-1} \right|}{\left| \text{EMA}_{t-1} \right| + \varepsilon}

with :math:`\varepsilon = 10^{-8}` added to the denominator purely to avoid division by zero. Because this signal is a *relative* change, it is scale-invariant across loss functions, regularisation weights, and data magnitudes — a key practical advantage over the linear model. Note also that the EMA convergence signal is unsigned (it uses absolute value), so it cannot distinguish a decreasing loss from an increasing one; it only reports whether the loss has stabilised. ``fitting.convergenceValue`` should therefore be a small positive number regardless of sign convention used elsewhere.

.. note::
   The two models are not numerically comparable. A ``convergenceValue`` tuned for ``'linear'`` mode (an absolute slope, in loss units per iteration) will not have the same meaning under ``'ema'`` mode (a unitless relative change). If you switch ``convergenceModel``, re-tune ``convergenceValue`` rather than reusing the same number.

**When to use** ``'ema'``: The linear slope can oscillate around zero during the later stages of optimisation, causing premature or delayed stopping depending on the window position. The EMA signal smooths out these short-term fluctuations and responds to genuine sustained improvement rather than transient dips. ``'ema'`` is the default and is recommended for most use cases, especially with spatial regularisation where the loss surface is less smooth.

**When to use** ``'linear'``: If you need exact reproducibility with results obtained with earlier versions of GACELLE, set ``fitting.convergenceModel = 'linear'`` and ``fitting.convergenceWindow = 20``.

.. note::
   With ``convergenceModel = 'ema'``, the EMA already smooths transient dips before thresholding, so ``patienceConvergence = 1`` or ``2`` is often sufficient. The default of ``5`` adds a conservative extra layer that costs little in practice.

Step Norm and Gradient Norm Convergence
-----------------------------------------

These two signals are independent of ``fitting.robustConvergence`` and the convergence model. They provide complementary stopping criteria based on parameter movement and loss landscape flatness respectively.

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.convergenceStepTol``
     - ``1e-6``
     - Relative parameter step norm threshold. Optimisation stops when the norm of the parameter update (relative to the parameter norm) is below this value. Set to ``0`` to disable.
   * - ``fitting.patienceStep``
     - ``5``
     - Patience for step norm criterion (consecutive checks below threshold required).
   * - ``fitting.convergenceGradTol``
     - ``1e-6``
     - Gradient norm threshold. Optimisation stops when the gradient norm is below this value. Set to ``0`` to disable.
   * - ``fitting.patienceGrad``
     - ``5``
     - Patience for gradient norm criterion (consecutive checks below threshold required).

With Adam, a small step norm does not necessarily imply a small gradient norm, because Adam normalises gradients via its second moment estimate. The step norm catches parameter stagnation; the gradient norm catches loss landscape flatness. Both are useful but can be disabled by setting their tolerance to ``0`` if you prefer to rely on the loss convergence signal alone.

Quick Reference: Default Behaviour (v1.1)
------------------------------------------

With all defaults, ``askadam.m`` v1.1 will:

- Use **sigmoid parameter reparameterisation** (``parameterTransform = 'sigmoid'``), eliminating boundary-sticking at the cost of a modified loss surface near bounds.
- Use **EMA-based convergence** (``convergenceModel = 'ema'``, ``emaDecay = 0.95``) rather than the linear slope.
- **Robust convergence is disabled by default** (``robustConvergence = false``). Enable it explicitly (``= true``) if your data show persistent outlier voxels; see `Robust Convergence and Outlier Handling`_ below.
- Stop on any of: loss threshold, EMA convergence, step norm, gradient norm, or maximum iterations.

To replicate v1.0 behaviour exactly::

   fitting.parameterTransform  = 'linear';
   fitting.convergenceModel    = 'linear';
   fitting.robustConvergence   = false;
   fitting.convergenceStepTol  = 0;
   fitting.convergenceGradTol  = 0;

Example: Default Settings
--------------------------

No additional fields are required — the defaults are active automatically:

.. code-block:: matlab

   fitting.optimiser    = 'adam';
   fitting.iteration    = 1e4;
   fitting.lossFunction = 'L1';

   out = askadam().optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

Example: Disabling Robust Convergence
---------------------------------------

.. code-block:: matlab

   fitting.robustConvergence  = false;
   fitting.convergenceModel   = 'ema';    % EMA still active
   fitting.convergenceValue   = 1e-6;

   out = askadam().optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

Example: Replicating v1.0 Behaviour
--------------------------------------

.. code-block:: matlab

   fitting.parameterTransform  = 'linear';
   fitting.convergenceModel    = 'linear';
   fitting.convergenceWindow   = 20;
   fitting.convergenceValue    = 1e-8;    % restore your original value if different from 1e-6
   fitting.robustConvergence   = false;
   fitting.convergenceStepTol  = 0;
   fitting.convergenceGradTol  = 0;

   out = askadam().optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

Robust Convergence and Outlier Handling (Experimental)
-------------------------------------------------------

When ``fitting.robustConvergence = true``, ``askadam.m`` identifies voxels whose loss is not improving relative to the main population and reduces their gradient contribution. The convergence signal (loss-based) is then computed on the main population only, preventing a small number of persistently difficult voxels from masking genuine convergence of the majority.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.robustConvergence``
     - ``false``
     - Enable outlier-aware gradient downweighting and main-population convergence checking.
   * - ``fitting.outlierWeight``
     - ``0.1``
     - Gradient contribution of voxels classified as outliers, as a fraction of normal weight. ``0`` suppresses outlier gradients entirely; ``1`` disables downweighting.
   * - ``fitting.weightUpdateInterval``
     - ``5``
     - Number of iterations between outlier mask updates. Updating every iteration is noisier; larger values are more stable but slower to respond.
   * - ``fitting.outlierThresholdMethod``
     - ``'behaviour'``
     - Method for classifying outlier voxels. Currently ``'behaviour'`` (loss trajectory based); further options may be added in future releases.

**Outlier classification criteria**

A voxel is flagged as an outlier if it satisfies either of two criteria:

*Criterion A — stagnation relative to population (recent window)*

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.outlierCheckWindow``
     - ``5``
     - Number of outlier mask update steps (each separated by ``weightUpdateInterval`` iterations) over which improvement is assessed for criterion A.
   * - ``fitting.outlierVoxelThres``
     - ``0.01``
     - A voxel must have improved its loss by less than this fraction (1%) over the check window to be flagged.
   * - ``fitting.outlierPopThres``
     - ``0.05``
     - The median population improvement over the same window must exceed this fraction (5%) for criterion A to activate. This prevents flagging voxels when the whole volume has stagnated.

*Criterion B — stagnation relative to initialisation (cumulative)*

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.outlierInitThres``
     - ``0.05``
     - A voxel must have improved by less than this fraction (5%) from its initial loss to be flagged under criterion B.
   * - ``fitting.outlierInitPopThres``
     - ``0.20``
     - The median population improvement from initialisation must exceed this fraction (20%) for criterion B to activate.

*Reinstatement*

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.outlierMinFlagDuration``
     - ``5``
     - Minimum number of update steps a voxel must remain flagged before it can be reinstated. Prevents rapid oscillation in the outlier mask.

.. note::
   The outlier mask has a one-iteration lag: it is computed from the loss at iteration *t* and applied from iteration *t+1*. This is an intentional design choice — computing the mask inside the autodiff graph would require ``extractdata`` calls that break gradient flow. TV regularisation gradients are unaffected by voxel downweighting.

.. warning::
   ``fitting.robustConvergence`` is disabled by default (``false``). Enable it (``= true``) for datasets where a minority of voxels are expected to stagnate persistently relative to the rest of the volume; it is not needed for typical well-conditioned data. Because GACELLE v1.0 did not include this mechanism, the default already reproduces v1.0 behaviour on this option.

See also :ref:`askadam-convergence` and `askAdam basic tutorial <https://gacelle.readthedocs.io/en/latest/getting_started/askadam_basic_tutorial.html>`_.