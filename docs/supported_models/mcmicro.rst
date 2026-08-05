.. _supportedmodels-mcmicro:
.. role::  raw-html(raw)
    :format: html

gpumcmicro
==========

mcmicro estimates neurite volume fraction and intrinsic (microscopic) diffusivity from the spherical mean of multi-shell diffusion MRI, using a two-compartment (intra-/extra-neurite) model with no inter-compartment exchange. When data are acquired at more than one echo time, it additionally fits compartmental R2 (R2a: intra-neurite, R2e: extra-neurite), combining diffusion and relaxation contrast in a single spherical-mean fit.

Reference: `Kaden, E., Kelm, N.D., Carson, R.P., Does, M.D., Alexander, D.C., 2016. Multi-compartment microscopic diffusion imaging. NeuroImage 139, 346-359. <https://doi.org/10.1016/j.neuroimage.2016.06.002>`_

Usage
^^^^^

.. code-block::

    obj = gpumcmicro(b, te);
    [out] = obj.estimate( data, mask, fitting, extraData, pars0);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../mcmicro/gpumcmicro.m
    :language: matlab
    :lines: 27-31

I/O overview
^^^^^^^^^^^^

``obj = gpumcmicro(b, te);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| b                         | 1xNshell b-values vector [ms/um2]                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| te                        | 1xNshell echo time [s] (Optional; scalar broadcasts to all shells; omitted = single TE, R2a/R2e not fitted)  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, fitting, extraData, pars0);``

.. note::
   ``estimate()`` takes its arguments in the order ``(data, mask, fitting, extraData, pars0)`` - ``fitting`` before ``extraData`` - which differs from every other GACELLE model class (``data, mask, extraData, fitting``). Double-check argument order if adapting a script from another model page.

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 4D DWI, [x,y,z,dwi], full acquisition or rotationally invariant (l=0) signal                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown; note argument order below)|
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | Solver used for estimation, 'askadam' (default) | 'mcmc'                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'likelihood' (default) | 1xM parameters array                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isFitD            | Intrinsic diffusivity 'D' is a free parameter, true (default) | false. See note below.                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.bval            | 1D b-values [1xdwi], same order as 'data' [ms/um2] (Optional, only if 'data' is full acquisition)            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.bvec            | 2D b-vector [3xdwi], same order as 'data' (Optional, only if 'data' is full acquisition)                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.ldelta          | 1D gradient pulse duration [1xdwi], same order as 'data' [ms] (Optional, only if 'data' is full acquisition) |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.BDELTA          | 1D diffusion time [1xdwi], same order as 'data' [ms] (Optional, only if 'data' is full acquisition)          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.D               | 3D fixed intrinsic diffusivity map, [x,y,z] [um2/ms] (Required if fitting.isFitD = false)                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| pars0                     | Structure array of starting points, one field per model parameter, same spatial size as 'data' (Optional)    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   ``fitting.isFitD = false`` fixes the intrinsic diffusivity to a supplied map (``extraData.D``) rather than estimating it; the model parameter ``D`` is then dropped from ``modelParams`` entirely rather than held at a constant internal value.

.. note::
   Unlike ``gpuNEXI``, this class does not expose a working ``fitting.lmax`` option: the value is hardcoded to ``0`` everywhere it is used (``prepare_dwi_data``, ``compute_optimisation_weights``, ``estimate_prior``), so only the l=0 (orientation-averaged) rotational invariant is ever fitted. A ``fitting.lmax`` line appears in this file's ``fit()`` docstring but has no effect; it looks like it was copied from a related class and not wired up here.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../mcmicro/demo_noise_propagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../mcmicro/demo_invivo.m
    :language: matlab

