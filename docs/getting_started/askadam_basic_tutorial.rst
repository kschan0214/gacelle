.. _gettingstarted-askadam_basic_tutorial:
.. role::  raw-html(raw)
    :format: html

askAdam basic tutorial
======================

This tutorial demonstrates an example of how to use the askAdam solver in this package for model parameter estimation. 

Let's say we have a simple monoexponential decay model:

.. math::

    S = S0 \times e^{-R_{2}^{*}t}

In this model, we have two parameters to be estimated: :math:`S0` and :math:`R_{2}^{*}`.

The first thing is to create a function to generate the forward signal. Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam.m
    :language: matlab

We can simulate the measurements using this function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 4-28

.. note::
    The dimenion and arrangement of y must be the same as the output of the forward function.

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 31-32

2. Set up the model parameters and fitting boundary

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 34-39

3. Set up optimisation setting

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 40-47

4. Define the forward function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 49-50

5. Define fitting weights (optional)

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 52-53

6. Start the optimisation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 55-56

7. Plot the estimation results

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam.m
    :language: matlab
    :lines: 58-64

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_askadam.m>`_.
    