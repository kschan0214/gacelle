.. _supportedmodels-NEXI:
.. role::  raw-html(raw)
    :format: html

gpuNEXI
========

NEXI (Neurite EXchange Imaging) estimates neurite volume fraction, intra- and extra-neurite diffusivities, and the inter-compartment water exchange rate from multi-shell, multi-diffusion-time dMRI. It extends the Standard Model of white/gray matter diffusion with the anisotropic Kärger model of two exchanging compartments, and is fitted here using the spherical mean (rotationally invariant) signal to remove the confound of fibre orientation dispersion.

Reference: `Chan, K.-S., Ma, Y., Lee, H., Marques, J.P., Olesen, J.L., Coelho, S., Novikov, D.S., Jespersen, S.N., Huang, S.Y., Lee, H.-H., 2025. In vivo human neurite exchange time imaging at 500 mT/m diffusion gradients. Imaging Neuroscience 3, imag_a_00544. <https://doi.org/10.1162/imag_a_00544>`_

The underlying NEXI model was originally proposed by Jelescu, I.O., de Skowronski, A., Geffroy, F., Palombo, M., Novikov, D.S., 2022. Neurite Exchange Imaging (NEXI): A minimal model of diffusion in gray matter with inter-compartment water exchange. NeuroImage 256, 119277.

Usage
^^^^^

.. code-block::

    obj = gpuNEXI(bval, BDELTA, varargin)
    out = obj.estimate(dwi, mask, extradata, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../NEXI/gpuNEXI.m
    :language: matlab
    :lines: 28-32

I/O overview
^^^^^^^^^^^^

``obj = gpuNEXI(bval, BDELTA);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| bval                      | 1xNshell b-values vector [ms/um2]                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| BDELTA                    | 1xNshell diffusion time, same size as 'bval' [ms]                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| varargin{1}               | Number of gradient directions per shell, same size as 'bval' (Optional, default = 1 per shell)               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``out = obj.estimate(dwi, mask, extradata, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| dwi                       | 4D dMRI data, can be either full acquisition or rotationally invariant signal [x,y,z,diffusion]              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z] (volumetric) | [1,Nvertex,Nhemi] (surface, see fitting.dataType below)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bval            | 1D b-values [1xdiffusion], same order as 'dwi' [ms/um2] (Optional, only if 'dwi' is full acquisition)        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bvec            | 2D b-vector [3xdiffusion], same order as 'dwi' (Optional, only if 'dwi' is full acquisition)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.ldelta          | 1D gradient duration [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.BDELTA          | 1D diffusion time [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.sigma           | 3D noise map, [x,y,z] (Optional, only needed for the NEXIrice noise model)                                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.surf_dir        | FreeSurfer surf directory (Required if fitting.dataType = 'surface')                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.hemisphere      | Cell array subset of {'lh','rh'} (Required if fitting.dataType = 'surface'; must match mask dim3)            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.depth           | Cortical depth, [0,1] (Optional, surface mode only, default = 0.5)                                           |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | Solver used for estimation, 'askadam' (default) | 'mcmc'. See note below.                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lmax              | Maximum order of rotational invariant, 0 (default) | 2                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.dataType          | Data geometry, 'volumetric' (default) | 'surface'. See note below.                                           |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.regularisationType| Regularisation dispatch, 'TV' (default) | 'prior'. See note below.                                           |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'likelihood' (default) | 'default' | 1xM parameters array                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   As of v0.5.0, ``gpuNEXI`` dispatches internally on ``fitting.solver``: the same object handles both the askAdam and MCMC solvers, and setting ``fitting.solver = 'mcmc'`` runs the MCMC path without needing a separate object. A standalone ``gpuNEXImcmc`` class, if still present in the repository, is likely a thin or legacy wrapper around this dispatch rather than an independently maintained implementation.

.. note::
   ``fitting.dataType = 'surface'`` expects ``data``/``mask`` in ``[1, Nvertex, Nhemi]`` convention and requires ``extradata.surf_dir`` and ``extradata.hemisphere``. It bypasses ``fitting.TVmode`` entirely: spatial regularisation is dispatched through a mesh-based total variation operator over the cortical surface rather than the volumetric one. Automatic segmentation in surface mode is exact-only, so ``fitting.NSegmentUser`` must be ``1`` or the number of hemispheres.

   ``fitting.regularisationType = 'prior'`` fits a normal-distribution prior on the parameter(s) named in ``fitting.regmap`` instead of total variation, and requires ``extradata.mu`` and ``extradata.sigma`` (one field per name in ``fitting.regmap``). This path is available for either ``dataType``.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.


Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../NEXI/demo_gpuNEXI_NoisePropagation.m
    :language: matlab

Example script for noise propagation with full diffucion table:

.. literalinclude:: ../../NEXI/demo_gpuNEXI_NoisePropagation_advanced.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../NEXI/demo_gpuNEXI_invivo.m
    :language: matlab
