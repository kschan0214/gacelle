.. _askadam-parameter-transform:

Parameter Space Transform
==========================

``askadam.m`` must optimise parameters that are physically constrained to a bounded interval ``[lb, ub]``. How those bounds are enforced during optimisation matters: a poor choice can cause parameters to become numerically stuck near the boundary, slowing or preventing convergence. This page explains the two available strategies, controlled by ``fitting.parameterTransform`` available from *GACELLE* v1.1.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fitting.parameterTransform``
     - ``'sigmoid'``
     - ``'sigmoid'``: unconstrained optimisation via logit/sigmoid reparameterisation (recommended). ``'linear'``: direct optimisation in the rescaled ``[0, 1]`` interval with hard clamping (v1.0 behaviour).

The Problem: Boundary Sticking in Linear Mode
----------------------------------------------

In ``'linear'`` mode, each parameter :math:`\theta` is linearly rescaled to the unit interval before optimisation:

.. math::

   \hat{\theta} = \frac{\theta - \text{lb}}{\text{ub} - \text{lb}} \in [0, 1]

Adam then optimises :math:`\hat{\theta}` directly, with hard clamping applied after each update to keep values inside ``[0, 1]``. This approach has a practical failure mode: Adam's second moment estimate :math:`v_t` accumulates the squared gradient over time. When a parameter approaches a boundary, gradients are repeatedly clipped to zero (by the clamp), causing :math:`v_t` to shrink toward zero. The effective Adam step size is proportional to :math:`1/\sqrt{v_t}`, which consequently diverges near the boundary — but because the clamp then immediately nullifies any update that would cross the bound, the parameter asymptotically crawls toward the boundary and effectively freezes there.

This is not a model identifiability problem. It is a numerical artifact of combining hard clamping with an adaptive step-size optimizer that accumulates second moment history. It can occur even when the true optimum is well inside the feasible region.

The Solution: Sigmoid Reparameterisation
-----------------------------------------

In ``'sigmoid'`` mode, the optimisation variable is transformed to an **unconstrained** space. Adam never sees the boundary at all.

**Initialisation (logit transform)**

Given a starting point :math:`\theta_0 \in [\text{lb}, \text{ub}]`, it is first normalised to ``(0, 1)`` and then mapped to the real line via the logit function:

.. math::

   \hat{\theta}_0 = \frac{\theta_0 - \text{lb}}{\text{ub} - \text{lb}}, \qquad
   z_0 = \log\!\left(\frac{\hat{\theta}_0}{1 - \hat{\theta}_0}\right)

Adam optimises :math:`z \in (-\infty, +\infty)` without any constraints or clamping.

**Recovery (sigmoid transform)**

At each iteration, physical parameter values are recovered from :math:`z` for use in the forward model and loss computation:

.. math::

   \theta = \text{lb} + (\text{ub} - \text{lb}) \cdot \sigma(z), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}

Because :math:`\sigma(z) \in (0, 1)` for all finite :math:`z`, the recovered :math:`\theta` is always strictly inside ``(lb, ub)`` without any clamping. The transform is differentiable everywhere, so automatic differentiation operates normally through it.

**The eps_bound safeguard**

Before applying the logit transform at initialisation, starting points are clamped to ``[lb + eps, ub - eps]`` where ``eps = 1e-4 * (ub - lb)``. This is a purely numerical safeguard: logit diverges at exactly 0 and 1, and a user-provided starting point at or very near a boundary would produce ``Inf`` or ``NaN`` in :math:`z_0`. The clamp is intentionally narrow (0.01% of the parameter range) so that it does not meaningfully distort a well-chosen starting point.

.. note::
   ``eps_bound`` is not a soft constraint and does not define a forbidden zone during optimisation. Once :math:`z_0` is initialised, Adam optimises freely in unconstrained space and the sigmoid recovery can produce values arbitrarily close to ``lb`` or ``ub`` if the data support it.

Properties and Trade-offs
--------------------------

**Slow convergence near boundaries is expected and acceptable**

The sigmoid reparameterisation does not eliminate slow behaviour near boundaries — it just means that slowness reflects genuine model physics rather than a numerical artifact. If the true optimum is near a boundary, the gradient of the loss with respect to :math:`z` passes through the sigmoid derivative :math:`\sigma(z)(1-\sigma(z))`, which is small when :math:`z` is large in magnitude. A parameter that genuinely belongs near a boundary will converge slowly because the *data* provide little gradient signal there, not because of the parameterisation.

**The loss surface geometry changes near bounds**

Compared to linear mode, the effective loss surface in :math:`z`-space is compressed near the boundaries and expanded near the centre of the feasible region. This can be beneficial (smoother gradients away from boundaries) but means that the scale of ``fitting.initialLearnRate`` has a slightly different interpretation. In practice, the default learn rate works well for most qMRI models.

**Bounds should reflect physical reality**

Because ``eps_bound`` is narrow and Adam optimises without constraints, ``fitting.lb`` and ``fitting.ub`` should be set to physically meaningful limits rather than conservative numerical guards. Overly tight bounds that exclude plausible physiological values will genuinely constrain the fit in sigmoid mode (since the true optimum could be mapped to a large :math:`|z|` where the gradient is weak).

**Per-parameter transforms are not supported**

``fitting.parameterTransform`` applies the same transform to all parameters. A per-parameter specification was considered and rejected as unnecessarily complex for the marginal benefit; the sigmoid transform is well-behaved for all bounded parameters.

Selecting a Mode
-----------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Situation
     - Recommendation
   * - New fits, default usage
     - Use ``'sigmoid'`` (the default). It eliminates a real numerical artifact with negligible downside for most qMRI models.
   * - Reproducing v1.0 results exactly
     - Set ``fitting.parameterTransform = 'linear'``. Results will be numerically identical to GACELLE v1.0.
   * - Parameters with very wide bounds relative to the expected posterior
     - ``'sigmoid'`` is preferable — ``'linear'`` is most prone to boundary sticking when the ratio of prior range to posterior width is large.
   * - Debugging a forward model that produces NaN/Inf
     - Temporarily switch to ``'linear'`` to simplify the computation graph during debugging; ``'sigmoid'`` adds a logit/sigmoid layer that can obscure the source of numerical issues.

Example
--------

No additional configuration is needed to use sigmoid mode — it is the default:

.. code-block:: matlab

   % Sigmoid mode (default): no extra fields required
   obj = askadam;
   out = obj.optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

To revert to linear mode:

.. code-block:: matlab

   fitting.parameterTransform = 'linear';

   obj = askadam;
   out = obj.optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

See also :ref:`askadam-convergence` and :ref:`automatic_memory_management`.