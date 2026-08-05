.. _supportedmodels-SANDI:
.. role::  raw-html(raw)
    :format: html

gpuSANDI
========

SANDI (Soma And Neurite Density Imaging) estimates apparent soma radius and signal fraction alongside neurite density and extracellular diffusivity from the spherical mean of multi-shell diffusion MRI. It models the signal as three compartments: restricted diffusion within spherical somas (of fixed, user-supplied intrinsic diffusivity), stick-like diffusion within neurites, and hindered isotropic diffusion in the extracellular space.

Reference: `Palombo, M., Ianus, A., Guerreri, M., Nunes, D., Alexander, D.C., Shemesh, N., Zhang, H., 2020. SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI. NeuroImage 215, 116835. <https://doi.org/10.1016/j.neuroimage.2020.116835>`_

Usage
^^^^^

.. code-block::

    obj = gpuSANDI(b, ldelta, BDelta, Ds, varargin);
    [out] = obj.estimate( data, mask, fitting, extradata, pars0);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../SANDI/gpuSANDI.m
    :language: matlab
    :lines: 25-29

I/O overview
^^^^^^^^^^^^

``obj = gpuSANDI(b, ldelta, BDelta, Ds, varargin);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| b                         | 1xNshell b-values vector [ms/um2]                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| ldelta                    | 1xNshell gradient pulse duration, same size as 'b' [ms]                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| BDelta                    | 1xNshell diffusion time, same size as 'b' [ms]                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Ds                        | Fixed intrinsic diffusivity of soma [um2/ms] (not fitted; a scalar constant, not a per-voxel map)            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| varargin{1}               | Number of gradient directions per shell, same size as 'b' (Optional, default = 1 per shell)                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, fitting, extradata, pars0);``

.. note::
   ``estimate()`` takes its arguments in the order ``(data, mask, fitting, extradata, pars0)`` - ``fitting`` before ``extradata`` - which differs from most other GACELLE model classes (``data, mask, extraData, fitting``). ``gpumcmicro`` shares this same non-standard order; double-check argument order if adapting a script from another model page.

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
| fitting.pulseType         | Gradient pulse approximation for the restricted soma signal, 'wide' (default) | 'narrow'                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lmax              | Forced to 0 regardless of input; setting > 0 raises a warning and is ignored. See note below.                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bval            | 1D b-values [1xdwi], same order as 'data' [ms/um2] (Optional, only if 'data' is full acquisition)            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bvec            | 2D b-vector [3xdwi], same order as 'data' (Optional, only if 'data' is full acquisition)                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.ldelta          | 1D gradient pulse duration [1xdwi], same order as 'data' [ms] (Optional, only if 'data' is full acquisition) |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.BDELTA          | 1D diffusion time [1xdwi], same order as 'data' [ms] (Optional, only if 'data' is full acquisition)          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| pars0                     | Structure array of starting points, one field per model parameter, same spatial size as 'data' (Optional)    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   ``fitting.lmax`` only accepts ``0``: ``check_set_default`` overwrites any input value to ``0`` and issues a warning ("Higher order rotationally invariant model is not yet supported") if you set it above zero. Only the l=0 (orientation-averaged) rotational invariant is fitted; there is currently no l=2/anisotropy term for this model, unlike ``gpuNEXI``.

.. note::
   ``fitting.pulseType`` selects between the wide-pulse and narrow-pulse (Neuman) approximations for restricted diffusion within the spherical soma compartment; it has no effect on the neurite or extracellular compartments, which use the standard stick and isotropic-Gaussian signal forms regardless of this setting.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../SANDI/demo_gpuSANDI_NoisePropagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../SANDI/demo_invivo.m
    :language: matlab

