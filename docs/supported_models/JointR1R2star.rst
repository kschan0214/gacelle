.. _supportedmodels-JointR1R2star:
.. role::  raw-html(raw)
    :format: html

gpuJointR1R2starMapping
=========================

JointR1R2star jointly estimates R1 (=1/T1) and R2* from a single-compartment relaxometry model combining variable flip angle (VFA) spoiled gradient echo data with multi-echo GRE data. R1 is obtained from the DESPOT1 steady-state VFA signal model; R2* is obtained from the mono-exponential decay across echo times of the same acquisition, fitted jointly with R1 and the proton-density-weighted signal M0 in a single optimisation.

Reference: `Deoni, S.C.L., Rutt, B.K., Peters, T.M., 2003. Rapid combined T1 and T2 mapping using gradient recalled acquisition in the steady state. Magnetic Resonance in Medicine 49, 515-526. <https://doi.org/10.1002/mrm.10407>`_ (VFA/DESPOT1 R1 model)

Usage
^^^^^

.. code-block::

    obj = gpuJointR1R2starMapping(te,tr,fa);
    [out] = obj.estimate( data, mask, extraData, fitting);

Model parameters
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../R1R2s/gpuJointR1R2starMapping.m
    :language: matlab
    :lines: 25-29

I/O overview
^^^^^^^^^^^^

``obj = gpuJointR1R2starMapping(te,tr,fa);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| te                        | 1xNte echo time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| tr                        | repetition time [s]                                                                                          |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fa                        | 1xNfa flip angle vector [degree]                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

``[out] = obj.estimate( data, mask, extraData, fitting);``

+---------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                     | Description                                                                                                  |
+===========================+==============================================================================================================+
| data                      | 5D MRI data, [x,y,z,t,fa]                                                                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                      | 3D mask, [x,y,z]                                                                                             |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData                 | Structure array with additional data                                                                         |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| extraData.b1              | 3D B1+ map [ratio], [x,y,z]                                                                                  |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                   | Structure array for model parameter estimation (only class-specific options shown here)                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.solver            | Solver used for estimation, 'askadam' (default) | 'mcmc'. See note below.                                    |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.start             | Starting point method, 'prior' (default, closed-form) | 'default' | 1xM parameters array                     |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isWeighted        | Weight the cost by echo intensity, true | false (default = false)                                            |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightMethod      | Weighting method, '1stecho' (default) | 'norm'                                                               |
+---------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.weightPower       | Power order of the weight (default = 2)                                                                      |
+---------------------------+--------------------------------------------------------------------------------------------------------------+

.. note::
   As of the June 2026 update, ``gpuJointR1R2starMapping`` dispatches internally on ``fitting.solver``: the same object handles both the askAdam and MCMC solvers, and setting ``fitting.solver = 'mcmc'`` runs the MCMC path without needing a separate object. A standalone ``gpuJointR1R2starMappingmcmc`` class, if still present in the repository, is likely a thin or legacy wrapper around this dispatch rather than an independently maintained implementation.

``estimate()`` also runs GACELLE's automatic GPU memory manager (``utils.find_optimal_segment_3D``) transparently, segmenting large volumes if required. See `Automatic GPU Memory Management <https://gacelle.readthedocs.io/en/latest/advanced/automatic_memory_management.html>`_ for the relevant ``fitting.autoMemManage``, ``fitting.NSegmentUser``, and ``fitting.segmentOverlap`` options.

Example
^^^^^^^

Example script for noise propagation:

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMapping_NoisePropagation.m
    :language: matlab

Example script for in vivo data:

.. literalinclude:: ../../R1R2s/demo_gpuJointR1R2starMapping_invivo.m
    :language: matlab
