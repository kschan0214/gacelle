.. _supportedmodels-MCRMWI:
.. role::  raw-html(raw)
    :format: html

gpuMCRMWI
=========

MCR-MWI (Multi-Compartment Relaxometry Myelin Water Imaging) jointly fits a three-pool (myelin water, intra-, extra-axonal water) complex-valued signal model to variable flip angle, multi-echo GRE data, estimating myelin water fraction (MWF) alongside compartmental R1, R2*, frequency, and the myelin-to-intra/extra-axonal water exchange rate. Exchange is modelled via the Bloch-McConnell equations, evaluated either analytically or through a pretrained multilayer-perceptron (EPG-X) surrogate for speed. As with GRE-MWI, fibre geometry from diffusion MRI can optionally constrain the model through the same DIMWI hollow-cylinder fibre framework.

Reference: `Chan, K.-S., Marques, J.P., 2020. Multi-compartment relaxometry and diffusion informed myelin water imaging - Promises and challenges of new gradient echo myelin water imaging methods. NeuroImage 221, 117159. <https://doi.org/10.1016/j.neuroimage.2020.117159>`_

`Chan, K.-S., Chamberland, M., Marques, J.P., 2023. On the performance of multi-compartment relaxometry for myelin water imaging (MCR-MWI) - test-retest repeatability and inter-protocol reproducibility. NeuroImage 266, 119824. <https://doi.org/10.1016/j.neuroimage.2022.119824>`_

Chan, K.-S., Kim T.H., Bilgic B., Marques J.P., 2022. Semi-supervised learning for fast multi-compartment relaxometry myelin water imaging (MCR-MWI). In: Proceedings 30th Annual Meeting ISMRM, London, UK, 1639. (EPG-X neural-network acceleration used by ``fitting.isEPG``)

Usage
^^^^^

.. code-block::

    obj = gpuMCRMWI(te,tr,fa,fixed_params);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../MCRMWI/gpuMCRMWI.m
    :language: matlab
    :lines: 35-39

I/O overview
^^^^^^^^^^^^

``obj = gpuMCRMWI(te,tr,fa,fixed_params);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| te                        | 1xNte echo time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| tr                        | repetition time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fa                        | 1xNfa flip angle vector [degree]                                                                             |
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
| fixed_params.t1_mw        | myelin (water) T1 [s], default = 234e-3                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.thres_R2star | single-compartment R2* threshold used to refine the brain mask [1/s], default = 2                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fixed_params.thres_T1     | upper bound on fitted T1 used to refine the brain mask [s], default = 3.1                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 5D VFA multi-echo GRE data, [x,y,z,te,fa]. Complex-valued if fitting.isComplex; magnitude otherwise.         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data (Optional unless noted)                                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.b1              | 3D B1+ map [ratio], [x,y,z]                                                                                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.freqBKG         | 3D/4D initial estimate of total field [Hz] (highly recommended), [x,y,z,fa]; defaults to zero                |
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
| fitting.isComplex         | Fit complex-valued data (real+imaginary), true (default) | false. Auto-set to false for real-valued input.   |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'prior' (default, closed-form) | 1xM parameters array                                 |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isWeighted        | Weight the cost by echo intensity, true (default) | false                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightMethod      | Weighting method, '1stecho' (default) | 'norm'                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightPower       | Power order of the weight (default = 2)                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isFitExchange     | Myelin-water exchange rate (kIEWM) is a free parameter, default = true                                       |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isEPG             | Use the pretrained EPG-X network instead of the analytical Bloch-McConnell solution. See note below.         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.epgx_phase_ann    | Path to the pretrained EPG-X phase MLP .mat file. Default ships with GACELLE; see note below.                |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.epgx_mag_ann      | Path to the pretrained EPG-X magnitude MLP .mat file. Default ships with GACELLE; see note below.            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isMultiStep       | default = false. Present in check_set_default but not read elsewhere in this file; see note below.           |
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
   As of the 23 June 2026 update, ``gpuMCRMWI`` dispatches internally on ``fitting.solver``: the same object handles both the askAdam and MCMC solvers, and setting ``fitting.solver = 'mcmc'`` runs the MCMC path without needing a separate object. This contradicts a stale header comment in the source file ("Support askadam.m only!") left over from before the merge; the actual dispatch code in ``estimate()``, ``fit()``, and ``check_set_default()`` fully implements the MCMC branch, so the comment should not be relied on.

.. note::
   ``fitting.isEPG = true`` (default) replaces the analytical Bloch-McConnell exchange solution with a pretrained multilayer-perceptron surrogate (``fitting.epgx_phase_ann`` / ``fitting.epgx_mag_ann``) for speed. Both network paths are loaded unconditionally inside ``fit()`` regardless of ``fitting.isEPG``, so the files must exist even if you set ``isEPG = false``; the GACELLE-shipped defaults point to ``EPGXgen_net/MCRMWI_MLP_EPGX_RFphase50_T1M234_{phase,magn}.mat`` relative to the class file.

.. note::
   ``fitting.isMultiStep`` defaults to ``false`` and is set in ``check_set_default``, but no other code in this file reads it. It currently has no effect; treat it as reserved rather than functional until confirmed otherwise.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../MCRMWI/demo_gpuMCRMWI_noisePropagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../MCRMWI/demo_gpuMCRMWI_invivo.m
    :language: matlab
