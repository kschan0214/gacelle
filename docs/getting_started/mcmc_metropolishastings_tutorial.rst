.. _gettingstarted-mcmc_metropolishastings_tutorial:
.. role::  raw-html(raw)
    :format: html

MCMC Metropolis-Hastings tutorial
=================================

This tutorial demonstrates an example of how to use the Metropolis-Hastings MCMC solver in this package for model parameter estimation. This is a lower level usage in contrast to the mcmc.optimisation function. The main differences are:

- no masking of input measurement data
- only posterior distirbution is available (no summary statistics derived from posterior distribution)

Let's say we have a simple monoexponential decay model:

.. math::

    S = S0 \times e^{-R_{2}^{*}t}

In this model, we have two parameters to be estimated: :math:`S0` and :math:`R_{2}^{*}`.

The first thing is to create a function to generate the forward signal. Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_mcmc.m
    :language: matlab

Note that the design of the forward function is slightly stricter for the MCMC solver. The output signal s must have a dimension of Nmeas by Nvoxel.

We can simulate the measurements using this function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 4-28

.. note::
    Note that 'y' must be a [Nmeas x Nvoxels] matrix here.

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 31-34

2. Set up the model parameters and fitting boundary

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 38-42

3. Set up optimisation setting

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 43-46

4. Define the forward function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 48-49

5. Define fitting weights (optional)

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 51-52

6. Start the optimisation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 54-55

7. Plot the estimation results

.. literalinclude:: ../../examples/Example_monoexponential_estimate_mcmc_MH.m
    :language: matlab
    :lines: 57-67

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_mcmc_MH.m>`_.
    