.. _supportedmodels-GREMWI:
.. role::  raw-html(raw)
    :format: html

gpuGREMWI
=========

GRE-MWI fits a three-pool (myelin water, intracellular/axonal water, extracellular water) complex-valued signal model to multi-echo gradient-echo data to estimate myelin water fraction (MWF) and the other compartmental relaxation and frequency parameters. Optionally, fibre geometry derived from diffusion MRI (DIMWI: diffusion-informed myelin water imaging) constrains the frequency and R2* of the intra/extra-axonal compartments via a hollow-cylinder fibre model, improving conditioning of the otherwise ill-posed multi-exponential fit.

Reference: `Chan, K.-S., Marques, J.P., 2020. Multi-compartment relaxometry and diffusion informed myelin water imaging - Promises and challenges of new gradient echo myelin water imaging methods. NeuroImage 221, 117159. <https://doi.org/10.1016/j.neuroimage.2020.117159>`_

Usage
^^^^^

.. code-block::

    obj = gpuGREMWI(te,fixed_params);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../MCRMWI/gpuGREMWI.m
    :language: matlab
    :lines: 31-35

I/O overview
^^^^^^^^^^^^

``obj = gpuGREMWI(te,fixed_params);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| te                        | 1xNte echo time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params              | Parameters to be fixed (Optional)                                                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.x_i          | isotropic susceptibility of myelin [ppm], default = -0.1                                                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.x_a          | anisotropic susceptibility of myelin [ppm], default = -0.1                                                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.E            | exchange induced frequency shift [ppm], default = 0.02                                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.rho_mw       | myelin water proton ratio, default = 0.36/0.86                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.B0           | main magnetic field strength [T], default = 3                                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.B0dir        | main magnetic field direction, [x,y,z], default = [0;0;1]                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.thres_R2s    | single-compartment R2* threshold used to refine the brain mask [1/s], default = 2                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 4D multi-echo GRE data, [x,y,z,te]. Complex-valued if fitting.isComplex; magnitude otherwise (see note below)|
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.freqBKG         | 3D initial estimate of total field [Hz] (highly recommended), [x,y,z]; defaults to zero if omitted           |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.pini            | 3D initial estimate of B1 offset [rad] (highly recommended), [x,y,z]; auto-estimated if omitted              |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.ff              | 3D/4D fibre fraction map, [x,y,z,Nfibre] (Required for DIMWI, i.e. any DIMWI.isFit* = false)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.theta           | 3D/4D angle between B0 and fibre orientation, [x,y,z,Nfibre] (DIMWI; derived from extraData.fo if omitted)   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.fo              | 4D/5D fibre orientation vector map, [x,y,z,Nfibre,3] (DIMWI; alternative to extraData.theta)                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.IWF             | 3D volume fraction Intracellular/(Intracellular+extracellular), [x,y,z] (Required if DIMWI.isFitIWF = false) |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | Solver used for estimation, 'askadam' (default) | 'mcmc'. See note below.                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isComplex         | Fit complex-valued data (real+imaginary), true (default) | false. See note below.                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'prior' (default, closed-form) | 1xM parameters array                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isWeighted        | Weight the cost by echo intensity, true (default) | false                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightMethod      | Weighting method, '1stecho' (default) | 'norm'                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightPower       | Power order of the weight (default = 1)                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.DIMWI             | Structure variable for DIMWI (diffusion-informed MWI) options                                                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.DIMWI.isFitIWF    | 'IWF' is a free parameter, default = true                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.DIMWI.isFitFreqMW | Myelin water frequency is a free parameter, default = true                                                   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.DIMWI.isFitFreqIW | Intracellular water frequency is a free parameter, default = true                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.DIMWI.isFitR2sEW  | Extracellular water R2* is a free parameter, default = true                                                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   As of the June 2026 update, ``gpuGREMWI`` dispatches internally on ``fitting.solver``: the same object handles both the askAdam and MCMC solvers, and setting ``fitting.solver = 'mcmc'`` runs the MCMC path without needing a separate object.

.. note::
   ``fitting.isComplex`` controls whether ``dfreqBKG`` and ``dpini`` (the residual background frequency and B1 phase offset parameters) are fitted at all: they are dropped from the model entirely when ``fitting.isComplex = false``, since they have no meaning for magnitude-only data. Note that ``check_set_default`` only downgrades ``isComplex`` to ``false`` automatically when the input ``data`` array is real-valued on a local copy of ``fitting`` that is not the one returned to the caller; if you are passing genuinely magnitude data, set ``fitting.isComplex = false`` explicitly rather than relying on auto-detection.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../MCRMWI/demo_gpuGREMWI_noisePropagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../MCRMWI/demo_gpuGREMWI_invivo.m
    :language: matlab
