.. _gettingstarted-design_model:
.. role::  raw-html(raw)
    :format: html

Designing a new model
=====================

Design a new model for askAdam solver
-------------------------------------

First things first...
#####################

To take advantage of the automatic gradient computation (i.e., ``dlgradient``) from Matlab Deep Learning toolbox (so that we don't need to provide the partial derivatives for the optimisation ourselves), the input data and estimation parameters are stored as the `dlarray <https://www.mathworks.com/help/deeplearning/ref/dlarray.html>`_ in ``askadam.m``. Please check the Matlab documentation to see `the list of functions with dlarrat support <https://www.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html>`_.

.. warning::
    It is tempting to use ``extractdata`` to pull the data out from dlarray for wider function support but this will cause error in automatic gradient computation. Instead, try approximation (e.g., ``trapz`` can be a good approximation of ``integral`` with sufficiently small spacing and supports dlarray).

What input do I need for my model function?
###########################################

In the most basic setting, we will need at least two input variables for the forward model function. The first variable is ALWAYS a structure array that contains all estimation parameters. The second variable could be any measurement parameter.

Here is an example for a simple monoexponential fitting:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam.m
    :language: matlab
    :lines: 20-32

Here, the variable ``pars`` has two fields: (1) ``S0`` corresponds to the initial signal intensity and (2) ``R2star`` is a decay constant. The measurement variable ``t`` is the sampling time of the decay. We can expand the number of input variables if needed.

Note that when using the forward model function with ``askadam.m``, the data in ``pars`` contain only masked voxel by default. In other words, ``pars.S0`` and ``pars.R2star`` in the example above are in size of [1*Nvoxel], where Nvoxel if the number of masked voxels. The motivation of masking prior foward signal generation is to improve GPU memory and computational efficiency. For in vivo neuroimaging, the brain mask mostly accounted for only 60% of the entire volume, which means we can save 40% computational sources.

Body of the forward function
############################

When it comes to the body of the forward function we have a great flexibility in terms of design. The main requirement is that the output signal ``S`` has to be with either the G-D size (i.e., [Nmeas*Nvoxels]) or full N-D size (e.g., [Nx*Ny*Nz*Nmeas]). 

We provide 3 examples here to show how this can be done:

- If you already masked the input data before ``askadam.m``, see :doc:`this tutorial <askadam_basic_tutorial>`;
- If you prefer using the full N-D data, with N-D computation, see :ref:`strategy1`;
- If you prefer using the full N-D data, with the most computationl efficient computation, see :ref:`strategy2`.


Design a new model for MCMC solver
----------------------------------

Since no gradient computation is needed for MCMC, the input data and estimation parameters are stored as `gpuArray <https://www.mathworks.com/help/parallel-computing/gpuarray.html>`_ with single precision. Here is `the list of functions with gpuArray support <https://www.mathworks.com/help/referencelist.html?type=function&capability=gpuarrays>`_.


