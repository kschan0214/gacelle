.. _supportedmodels-AxCaliberSMT:
.. role::  raw-html(raw)
    :format: html

gpuAxCaliberSMT
===================

AxCaliberSMT estimates a per-voxel, orientation-invariant axon diameter index from multi-shell diffusion MRI. It generalises the original single-fibre AxCaliber model (restricted diffusion in cylindrical axons) under the spherical mean technique (SMT) framework, which removes the confounding effects of fibre orientation dispersion and crossing fibres by averaging the diffusion signal over all sampled directions within each shell prior to model fitting. The signal is modelled as a three-compartment mixture of intra-axonal (cylindrically restricted), extra-axonal (hindered, Gaussian), and CSF (free, isotropic) diffusion.

Reference: Fan, Q., Nummenmaa, A., Witzel, T., Ohringer, N., Tian, Q., Setsompop, K., Klawiter, E.C., Rosen, B.R., Wald, L.L., Huang, S.Y., 2020. Axon diameter index estimation independent of fiber orientation distribution using high-gradient diffusion MRI. NeuroImage 222, 117197.

Usage
^^^^^

.. code-block::

    obj = gpuAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf, varargin)
    out = obj.estimate(s, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../AxCaliberSMT/gpuAxCaliberSMT.m
    :language: matlab
    :lines: 25-29

I/O overview
^^^^^^^^^^^^

``obj = gpuAxCaliberSMT(b, delta, Delta, D0, Da, DeL, Dcsf);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| b                         | 1xNshell b-values vector [ms/um2]                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| delta                     | 1xNshell diffusion gradient pulse width vector, aka little delta, same size as 'bval' [ms]                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Delta                     | 1xNshell diffusion time, aka big delta, same size as 'bval' [ms]                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| D0                        | intra-cellular intrinsic diffusivity [um2/ms]                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Da                        | intra-cellular axial diffusivity [um2/ms]                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| DeL                       | extra-cellular axial diffusivity [um2/ms]                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Dcsf                      | CSF diffusivity [um2/ms]                                                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``out = obj.estimate(dwi, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| dwi                       | 4D dMRI data, can be either full acquisition or SMT signal [x,y,z,diffusion]                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata                 | Structure array with additional data (Optional)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bval            | 1D b-values [1xdiffusion], same order as 'dwi' [ms/um2] (Optional, only if 'dwi' is full acquisition)        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.bvec            | 2D b-vector [3xdiffusion], same order as 'dwi' (Optional, only if 'dwi' is full acquisition)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.ldelta          | 1D gradient duration [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extradata.BDELTA          | 1D diffusion time [1xdiffusion], same order as 'dwi' [ms] (Optional, only if 'dwi' is full acquisition)      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.solver            | Solver used for estimation, 'askadam' (default) | 'mcmc'. See note below.                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.model             | Intracellular restricted-diffusion signal representation, 'vangelderen' (default) | 'neuman'                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 
| fitting.start             | Starting point method, 'likelihood' (default) | 'default' | 1xM parameters array                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+ 

.. note::
   As of v1.1, ``gpuAxCaliberSMT`` dispatches internally on ``fitting.solver``: the same object handles both the askAdam and MCMC solvers, and setting ``fitting.solver = 'mcmc'`` runs the MCMC path described under `gpuAxCaliberSMTmcmc`_ below without needing a separate object.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMT_NoisePropagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../AxCaliberSMT/demo_gpuAxCaliberSMT_invivoData.m
    :language: matlab
