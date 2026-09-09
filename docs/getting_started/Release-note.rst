.. _gettingstarted-releasenote:

Release note
============

v1.1 (current, in development)
-------------------------------

New QSM module
^^^^^^^^^^^^^^^
* New ``qsm/`` module, added as a new category of model class alongside the existing microstructure/relaxometry ones:

  * ``gpumcTFI`` - multi-echo complex Total Field Inversion: jointly reconstructs a preconditioned susceptibility map, R2*, M0 and a phase offset directly from raw complex multi-echo GRE data in a single nonlinear fit, combining mono-exponential R2* decay with the dipole-convolved local field's phase evolution across echoes (see :ref:`supportedmodels-mcTFI`). Spatial regularisation follows the MEDI family (anatomically-weighted TV, optional automatic CSF zero-referencing), implemented via a new internal ``+MEDI_helper`` package.
  * ``gpuPDF`` - GPU implementation of Projection onto Dipole Fields background field removal for QSM/phase processing (see :ref:`supportedmodels-PDF`).
* ``R2starMapping`` (single-compartment mono-exponential R2* decay) is now documented and packaged as its own standalone model rather than only as a component of the joint VFA-R1/R2* fit (see :ref:`supportedmodels-R2starMapping`); its implementation was also rewritten alongside the other model classes described below.

Unified solver architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Every model class now exposes a single ``fitting.solver = 'askadam' | 'mcmc'`` option and one ``estimate(...)`` entry point, replacing the previous pattern of separate ``gpu<Model>`` / ``gpu<Model>mcmc`` classes per solver (e.g. ``gpuNEXI`` + ``gpuNEXImcmc``, ``gpuJointR1R2starMapping`` + ``gpuJointR1R2starMappingmcmc``, ``gpuAxCaliberSMT`` + ``gpuAxCaliberSMTmcmc`` are each now a single class). The old solver-specific classes have been removed from the main model folders (old implementations kept only under each model's ``sandbox/deprecated/`` for reference). ``gpuMEAxCaliberSMT`` and ``gpuNEXIrice`` remain experimental and live under their model's ``sandbox/`` rather than being part of this release's supported model set.
* All demo/example scripts across every model were updated and re-validated against the new interface.

Automatic GPU memory management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New automatic memory manager (``utils.find_optimal_segment_3D``), enabled via ``fitting.autoMemManage = 1``: probes peak GPU memory on small sub-samples of the data, predicts the peak memory a full-volume fit would need, and - only if that would exceed available VRAM - transparently segments the volume into density-balanced, halo-padded chunks that are fitted sequentially and reassembled. Works with both ``askadam.m`` and ``mcmc.m``. See :ref:`automatic_memory_management`.
* Manual override of segmentation via ``fitting.segmentOverlap`` (halo size) and ``fitting.NSegmentUser`` (force a minimum segment count).

askAdam solver improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``askadam.m`` was substantially reorganised and extended (see :ref:`askadam-convergence`):

  * Five independent stopping criteria instead of a single loss threshold: loss threshold, loss convergence (selectable ``fitting.convergenceModel``: ``'linear'`` slope or ``'ema'``), step-norm convergence, gradient-norm convergence, and max iterations, each with its own patience window.
  * New ``fitting.robustConvergence`` option for outlier-robust convergence detection, so a minority of poorly-fitting voxels is less likely to stall or prematurely trigger convergence for the whole volume.
* New default parameter-space transform, ``fitting.parameterTransform = 'sigmoid'`` (unconstrained logit/sigmoid reparameterisation), replacing the previous hard-clamped linear rescaling as the default. This fixes a boundary-sticking failure mode where Adam's adaptive step size could numerically freeze a parameter near its bound even when the true optimum was well inside the feasible region. The old behaviour is still available via ``fitting.parameterTransform = 'linear'``. See :ref:`askadam-parameter-transform`.

MCMC solver improvements
^^^^^^^^^^^^^^^^^^^^^^^^^
* ``mcmc.m`` substantially reworked alongside the memory-manager and unified-solver changes above.
* New fixed-anchor-walkers option for the affine-invariant ensemble sampler.

Toolbox / convenience
^^^^^^^^^^^^^^^^^^^^^^
* New ``addpath_gacelle.m`` - adds the whole toolbox to the MATLAB path in one call, automatically excluding ``docs/``, ``sandbox/``, ``deprecated/`` and ``mpl_training/`` folders.
* New ``utils.GACELLE_version`` helper reporting the installed toolbox version.
* New ``utils.check_dwi_invivo_demo_data`` / ``utils.check_gre_invivo_demo_data`` helpers used by the in-vivo demo scripts to check for (and prompt download of) the required example data.

Documentation
^^^^^^^^^^^^^
* New "Advanced" section: :ref:`automatic_memory_management`, :ref:`askadam-convergence`, :ref:`askadam-parameter-transform`.
* New :doc:`Writing a new model <../tutorial/writing_a_new_model>` tutorial, walking through the shared model-class skeleton using ``gpuR2starMapping`` as a worked example.
* New :doc:`Understanding the output <output_handling>` page describing the ``out`` structure returned by ``askadam.m`` and ``mcmc.m``.
* New reference pages for :ref:`supportedmodels-mcTFI`, :ref:`supportedmodels-PDF`, :ref:`supportedmodels-R2starMapping`, :ref:`supportedmodels-mcmicro` and :ref:`supportedmodels-SANDI`.

v1.0
----

Release date: 16 November 2025

Initial tagged release.
