.. _supportedmodels-mcTFI:
.. role::  raw-html(raw)
    :format: html

gpumcTFI
========

mcTFI (multi-echo complex Total Field Inversion) jointly reconstructs a preconditioned susceptibility map, R2*, M0, and a phase offset directly from raw complex multi-echo GRE data, in a single nonlinear fit. It combines mono-exponential R2* decay with the dipole-convolved local field's phase evolution across echoes, so - unlike :ref:`supportedmodels-PDF`, which requires a pre-processed total field map as input - background field removal and dipole inversion are not separate preprocessing steps here. Spatial regularisation follows the MEDI family: an anatomically-weighted total variation term and, optionally, automatic CSF zero-referencing, both applied to the preconditioned susceptibility chi = P.*y rather than to the raw fitted parameter y.

The source file gives no reference for the joint complex parametrisation itself. Its regularisation machinery calls directly into ``MEDI_helper`` (``fgrad``, ``gradient_mask``, ``extract_CSF``, ``compute_preconditioner``), and ``reg_csf0ref`` matches the CSF zero-reference method published as:

`Liu, Z., Spincemaille, P., Yao, Y., Zhang, Y., Wang, Y., 2018. MEDI+0: Morphology enabled dipole inversion with automatic uniform cerebrospinal fluid zero reference for quantitative susceptibility mapping. Magnetic Resonance in Medicine 79, 2795-2803. <https://doi.org/10.1002/mrm.26946>`_

``reg_tv``'s morphology-masked spatial TV matches the earlier MEDI regulariser:

`Liu, J., Liu, T., de Rochefort, L., Ledoux, J., Khalidov, I., Chen, W., Tsiouris, A.J., Wisnieff, C., Spincemaille, P., Prince, M.R., Wang, Y., 2012. Morphology enabled dipole inversion for quantitative susceptibility mapping using structural consistency between the magnitude image and the susceptibility map. NeuroImage 59, 2560-2568. <https://doi.org/10.1016/j.neuroimage.2011.08.082>`_

These are the closest published methods to what the regularisation code implements, identified from the functions it calls; they are not stated as references anywhere in the source file itself.

Usage
^^^^^

.. code-block::

    obj = gpumcTFI(voxelSize, B0, B0dir, te);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../recon/gpumcTFI.m
    :language: matlab
    :lines: 18-21

``y`` is the preconditioned susceptibility (chi = extraData.P .* y) and ``phi`` is a per-voxel phase offset. There is no ``R1`` or solver-conditional ``noise`` term in this model, and no ``step`` array either, consistent with MCMC not being supported (see note below).

I/O overview
^^^^^^^^^^^^

``obj = gpumcTFI(voxelSize, B0, B0dir, te);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| voxelSize                 | 1x3 voxel size [mm]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| B0                        | Main field strength [T]                                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| B0dir                     | Static field direction, unit vector [x,y,z] in image coordinates                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| te                        | 1xnTE echo time vector [s] (>= 2 echoes; dTE taken as te(2)-te(1)). See note below.                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   ``te`` is technically optional (omitting it sets ``te = dTE = 0``) but is required whenever R2* is actually used - i.e. for ``fitting.precond`` values that depend on R2* (such as ``'auto'``/``'emp+r2s'``), for the CSF zero-reference mask, and for the model itself, since R2* is always fitted. Omitting it does not raise an error; it silently degrades the fit (dTE = 0 makes the phase-evolution term degenerate). Supply at least two echo times in essentially all real use.

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | Complex multi-echo GRE data, [x,y,z,nTE]. See note above - this is NOT a pre-processed field map.            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]. Data-fidelity support only; does not restrict where chi is estimated (whole FOV).          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.weights         | 3D fidelity weights (Optional; defaults to ones, replicated across echoes and the cos/sin channels)          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.R2star          | 3D precomputed R2* map [Hz] (Optional; else computed from 'data' via a trapezoidal fit)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.M0              | 3D precomputed M0 map (Optional; else computed from 'data' via the same trapezoidal fit)                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.MG              | 4D morphology/edge mask for TV, [x,y,z,3], 0 at edges (Optional; else derived from 'data')                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.M2              | 3D zero-reference mask, e.g. ventricular CSF (Optional; else derived from R2* if fitting.lambdaCSF>0)        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.fint            | 3D initial total field estimate [Hz] (Optional; triggers an internal gpuPDF call to derive chi_b)            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.chi_b           | 3D initial susceptibility estimate [ppm] (Optional; used as y0 = chi_b./P. See note below.)                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | 'askadam' only. See note below.                                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'prior' (default, closed-form) | 'default' | 1xM parameters array                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lambdaTV          | Weight of the morphology-masked spatial TV term on chi (default = 0, i.e. off). See note below.              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lambdaCSF         | Weight of the CSF zero-reference term on chi (default = 0, i.e. off). See note below.                        |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.precond           | Preconditioner mode, 'none' (default) | 'auto' | 'emp+r2s' (partial list; see note below)                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   If both ``extraData.fint`` and ``extraData.chi_b`` are supplied, ``fint`` takes precedence: ``prepare_data`` runs an internal ``gpuPDF`` fit on ``fint`` unconditionally whenever it is present and overwrites ``extraData.chi_b`` with the result before ``estimate_prior`` ever runs.

.. note::
   Unlike every other GACELLE model class, ``gpumcTFI`` genuinely only supports the askAdam solver. Requesting ``fitting.solver = 'mcmc'`` triggers a warning ("Only askadam.m is supported. Switched to askadam.") and falls back to askAdam; the class comment gives two reasons: the phasor-stacked fidelity is not a Gaussian likelihood in the measured field, and the whole-FOV parameter count is impractical for MCMC sampling.

.. note::
   ``fitting.lambdaTV`` and ``fitting.lambdaCSF`` both default to ``0`` (both regularisation terms off) via ``check_set_default``. Neither weight matches the published MEDI/MEDI+0 values directly: the regularisation terms here are means rather than sums and the fidelity is in ppm² rather than the original field units, so both need recalibrating on your own data regardless of published settings.

``estimate()`` does **not** run GACELLE's automatic GPU memory manager: like :ref:`supportedmodels-PDF`, the dipole convolution and the TV term are both global operations, so the volume cannot be segmented along any spatial axis without changing the operator. Large volumes are handled only by zero-padding (``this.gapMinMM``), not by spatial segmentation.

Example
^^^^^^^

See example here.
