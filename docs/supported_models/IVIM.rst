.. _supportedmodels-ivim:
.. role::  raw-html(raw)
    :format: html

gpuIVIM
=======

gpuIVIM fits the bi-exponential intravoxel incoherent motion (IVIM) model to multi-b diffusion MRI:

.. math::

    S(b) = S_0 \left[ f\,e^{-b D^*} + (1-f)\,e^{-b D} \right]

with perfusion fraction :math:`f`, tissue diffusion coefficient :math:`D` and pseudo-diffusion coefficient :math:`D^*`. b-values are in ms/µm² (1000 s/mm² = 1 ms/µm²), as in GACELLE's other diffusion models, so D and D* are in µm²/ms. The default bounds follow BayesIVIM (D in [0, 2.5], D* in [0, 50] µm²/ms) and suit brain IVIM; D* > D is not enforced, so set ``fitting.lb``/``fitting.ub`` (order S0, f, D, Dstar, plus noise for mcmc) to constrain it, e.g. for body imaging.

Reference: `Le Bihan, D., Breton, E., Lallemand, D., Aubin, M.L., Vignaud, J., Laval-Jeantet, M., 1988. Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging. Radiology 168, 497-505. <https://doi.org/10.1148/radiology.168.2.3393671>`_

Usage
^^^^^

.. code-block::

    obj = gpuIVIM(b);
    [out] = obj.estimate( data, mask, extraData, fitting, pars0);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../IVIM/gpuIVIM.m
    :language: matlab
    :lines: 8-12,23-27

I/O overview
^^^^^^^^^^^^

``obj = gpuIVIM(b);``

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Input
     - Description
   * - b
     - b-value of each volume of 'data', same order [ms/um2]; at least 3 distinct values

``[out] = obj.estimate( data, mask, extraData, fitting, pars0);``

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Input
     - Description
   * - data
     - 4D DWI, [x,y,z,dwi], one volume per b-value (direction-averaged or trace-weighted)
   * - mask
     - 3D mask, [x,y,z]
   * - extraData
     - Not used, kept for the common interface ([] is fine)
   * - fitting
     - Structure array for model parameter estimation (only class-specific options shown)
   * - fitting.solver
     - Solver used for estimation, 'askadam' (default) | 'mcmc'
   * - fitting.start
     - Starting point method, 'prior' (default, segmented fit) | 'default' | 1xM parameters array
   * - fitting.bThreshold
     - b-value separating the diffusion (b >= bThreshold) and perfusion regimes in the segmented start, 0.2 [ms/um2]
   * - pars0
     - Structure array of starting points, one field per model parameter, same spatial size as 'data' (Optional)

.. note::
   The data are normalised voxel-wise by the mean signal at the lowest b-value before fitting, so ``S0`` is relative (about 1). ``out.signalScale`` holds that normalisation; ``S0`` in data units is ``out.final.S0 .* out.signalScale`` (askadam) or ``out.median.S0 .* out.signalScale`` (mcmc). Background voxels (lowest-b signal below 1% of its 99th percentile) are removed from the mask; use ``out.mask`` in later analysis.

.. note::
   The segmented start fits D and the intercept by a log-linear fit on ``b >= bThreshold``, sets ``f = 1 - intercept/S0``, and fits D* to the residual on ``b < bThreshold``. Voxels where a step is not possible keep the default starting point.

.. note::
   As with the other model classes, ``fit`` adapts the object's parameter list to the solver (``noise`` exists for mcmc only), so construct a new object for each fit with a different solver.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../IVIM/demo_noise_propagation.m
    :language: matlab

Bayesian priors (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``fitting.solver = 'mcmc'`` and ``fitting.mcmcClass = 'mcmc_bayes'``, ``gpuIVIM`` can be fitted with a hierarchical (population) prior and a spatial prior. The coupled priors need the whole volume in one GPU call. See :ref:`mcmc-bayes`.

See :ref:`tutorial-mcmc_bayes_tutorial` for a worked example. In our tests the IVIM population prior mixed slowly (see the convergence warning in :ref:`mcmc-bayes`), so check ``out.hyper.rhat`` with care.
