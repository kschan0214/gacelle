.. _supportedmodels-PDF:
.. role::  raw-html(raw)
    :format: html

gpuPDF
======

PDF (Projection onto Dipole Fields) is a background field removal method for QSM/phase processing. It estimates a background susceptibility distribution supported outside a region-of-interest mask, such that its dipole field best explains the measured total field inside the mask; subtracting this fitted background field from the measured field yields the local (tissue) field used for downstream QSM reconstruction.

Reference: `Liu, T., Khalidov, I., de Rochefort, L., Spincemaille, P., Liu, J., Tsiouris, A.J., Wang, Y., 2011. A novel background field removal method for MRI using projection onto dipole fields (PDF). NMR in Biomedicine 24, 1129-1136. <https://doi.org/10.1002/nbm.1670>`_

Usage
^^^^^

.. code-block::

    obj = gpuPDF(voxelSize, B0, B0dir, dTE);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../recon/gpuPDF.m
    :language: matlab
    :lines: 52-56

I/O overview
^^^^^^^^^^^^

``obj = gpuPDF(voxelSize, B0, B0dir, dTE);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| voxelSize                 | 1x3 voxel size [mm]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| B0                        | Main field strength [T]                                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| B0dir                     | Static field direction, unit vector [x,y,z] in image coordinates                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| dTE                       | Echo spacing / effective TE [s] (Optional for the linear branch; REQUIRED for fitting.isnonlinear = true)    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 3D total field map [Hz], [x,y,z]                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D ROI mask, [x,y,z]. Defines fidelity support and, via its complement, parameter support. See note below.   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data (Optional)                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.weights         | 3D data weighting map (typically magnitude, normalised to max 1 so 'tol' stays dataset-portable)             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | 'askadam' only. See note below.                                                                              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isnonlinear       | Data-fidelity formulation, false (default, linear) | true (nonlinear phasor; requires dTE). See class header.|
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   Unlike every other GACELLE model class, ``gpuPDF`` genuinely only supports the askAdam solver. Requesting ``fitting.solver = 'mcmc'`` triggers a warning ("Only askadam.m is supported. Switched to askadam.") and silently falls back to askAdam; the class header explains why: the phasor-stacked nonlinear fidelity is not a Gaussian likelihood in the measured field, so it isn't compatible with the MCMC framework used elsewhere.

.. note::
   ``mask`` plays two roles simultaneously and they are not separable: the loss (data fidelity) is evaluated only where ``mask`` is true, while the fitted parameter ``chi_b`` is supported only where ``mask`` is false (i.e. its complement). Eroding or dilating the mask moves both the fitting region and the background-source region at once.

.. note::
   Starting points are not configurable here: ``determine_x0`` always initialises every voxel from the fixed ``this.startPoint`` (0 ppm), and there is no ``fitting.start`` option. The source code's own comment flags this as a known limitation for the nonlinear branch specifically, where the objective is periodic and non-convex and a better-informed starting point (e.g. a scaled background estimate) would be preferable but is not yet implemented.

.. note::
   Regularisation is not user-configurable despite ``fitting.lambda``/``regmap``/``TVmode`` existing as generic askAdam fields: ``check_set_default`` forces ``fitting2.lambda = {0}`` unconditionally, with the comment that askAdam's built-in TV regulariser cannot be used here because of the inverted-mask parameter support. Also unlike every other GACELLE model class, ``gpuPDF`` does **not** run the automatic GPU memory manager: the source comments explain that the dipole convolution is a global operation, so the volume cannot be segmented along any spatial axis without changing the forward operator. Large volumes are handled only by zero-padding (``this.gapMinMM``), not by spatial segmentation.

Example
^^^^^^^

Example script for in vivo data:

.. literalinclude:: ../../qsm/gpuPDF.m
    :language: matlab

