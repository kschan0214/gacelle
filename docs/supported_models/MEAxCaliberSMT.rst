.. _supportedmodels-meaxcalibersmt:
.. role::  raw-html(raw)
    :format: html

gpuMEAxCaliberSMT
=================

.. warning::
   ``gpuMEAxCaliberSMT`` is **experimental**. Its interface and behaviour may change.

Multi-echo AxCaliberSMT (ME-AxCaliberSMT) estimates the axon radius together with compartmental relaxation from spherical-mean diffusion MRI acquired at two or more echo times. It extends :ref:`supportedmodels-AxCaliberSMT` with intra- and extra-axonal R2: the intra-axonal R2 depends on the axon radius (R2a + k2a/r), which adds sensitivity to r through the echo-time dependence.

The model describes myelinated axons. In grey matter and CSF the fitted r has no physical meaning, so report r only within white matter.

Usage
^^^^^

.. code-block::

    obj = gpuMEAxCaliberSMT(b, ldelta, BDELTA, te, tissueProperties, model, Nav);
    [out] = obj.estimate( data, mask, fitting, extraData, pars0);

.. note::
   The argument order of ``estimate`` is ``(data, mask, fitting, extraData, pars0)``, which differs from the other model classes (``extraData`` before ``fitting``).

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../AxCaliberSMT/gpuMEAxCaliberSMT.m
    :language: matlab
    :lines: 14-43

I/O overview
^^^^^^^^^^^^

``obj = gpuMEAxCaliberSMT(b, ldelta, BDELTA, te, tissueProperties, model, Nav);``

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Input
     - Description
   * - b
     - b-value of each shell/echo combination [ms/um2]
   * - ldelta, BDELTA
     - gradient pulse duration and diffusion time [ms], same size as b
   * - te
     - echo time [s], same size as b
   * - tissueProperties
     - fixed tissue properties (diffusivities, intrinsic R2a, k2a); [] for the defaults
   * - model
     - model variant; [] for the default
   * - Nav
     - (optional) number of directions averaged per unique shell, in the class's sorted shell order; sets the fitting weights (default: equal)

``[out] = obj.estimate( data, mask, fitting, extraData, pars0);``

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Input
     - Description
   * - data
     - 4D DWI [x,y,z,dwi]: full acquisition (with ``extraData.bval``/``bvec``/``ldelta``/``BDELTA``/``te``) or spherical means
   * - mask
     - 3D mask [x,y,z]
   * - fitting
     - fitting options; class-specific: ``isFitCSF`` (default true), ``isFitR2a``, ``isFitk2a``, ``isFitS0`` (default false), ``start`` (default ``'likelihood'``)
   * - extraData
     - acquisition information for full DWI input; ``noiseSigma`` and ``priorLabels`` for ``mcmc_bayes`` (see :ref:`mcmc-bayes`)
   * - pars0
     - (optional) starting points, one field per model parameter

Bayesian priors (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``fitting.solver = 'mcmc'`` and ``fitting.mcmcClass = 'mcmc_bayes'``, ``gpuMEAxCaliberSMT`` can be fitted with a population prior (optionally per tissue/tract label) and a spatial prior. See :ref:`mcmc-bayes`. In vivo, fixed tissue labels (e.g. a tract, other white matter, grey matter and CSF) worked better than a learned two-group prior, which was multimodal.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuMEAxCaliberSMT_NoisePropagation.m
    :language: matlab
