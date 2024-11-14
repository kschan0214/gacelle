.. _gettingstarted-askadam_basicnd_tutorial:
.. role::  raw-html(raw)
    :format: html

askAdam basic tutorial for N-D data
===================================

This tutorial demonstrates an example of how to use the askAdam solver in this package for model parameter estimation on N-D data (N>2). 

Let's say we have a simple monoexponential decay model:

.. math::

    S = S0 \times e^{-R_{2}^{*}t}

In this model, we have two parameters to be estimated: :math:`S0` and :math:`R_{2}^{*}`.

.. _strategy1:

Strategy 1
----------

This strategy provides a more straightforward operation in terms of designing the signal model, by making all operations N-D based. The drawback is that more GPU memory and computational power are needed.

The first thing is to create a function to generate the forward signal. To accomodate the first 3 dimensions for spatial information, we will put the time :math:`t` in the 4th dimension.

Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam_3D_Strategy1.m
    :language: matlab

In contrast to :doc:`the basic tutorial <askadam_basic_tutorial>`, we have some extra operations here

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 32-37

This block of code is to handle the input from ``askadam.m``. As discussed in :doc:`designing a new model <designing_model>`, the estimation parameters from askAdam are already masked and reshaped into [1*Nvoxel] arrays. This block essentially has two operations: (1) checks if the estimation parameters are masked, (2) if so then reshape the estimation parameters to their original N-D size using the 3-D mask. GACELLE provides a utility function ``reshape_GD2ND`` to simplify the reshaping operation of an array from G-D size to N-D size.

Note that the final output ``S`` of this forward model function is always in 4D. ``askadam.m`` will take care of the masking operation internally.

We can then simulate the 4D measurements using this function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 4-30

This time we also have a spherical non-zero mask to demonstrate the usage of a mask.

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 17

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 34-35

2. Set up the model parameters and fitting boundary

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 37-42

3. Set up optimisation setting

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 43-50

4. Define the forward function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 52-53

5. Define fitting weights (optional)

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 55-56

6. Start the optimisation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 58-59

7. Plot the estimation results

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m
    :language: matlab
    :lines: 61-74

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_askadam_3D_Strategy1.m>`_.
    
.. _strategy2:

Strategy 2
----------

This strategy provides the most memory and computationally efficient way for the optimisation, but the operation is a bit less intuitive.

The first thing is to create a function to generate the forward signal. To accomodate the first 3 dimensions for spatial information, we will put the time :math:`t` in the 4th dimension.

Here is an example:

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam_3D_Strategy2.m
    :language: matlab

In contrast to :doc:`the basic tutorial <askadam_basic_tutorial>`, we have some extra operations here

.. literalinclude:: ../../examples/Example_monoexponential_FWD_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 38-40

This block of code is to handle the input from ``askadam.m``. As discussed in :doc:`designing a new model <designing_model>`, the estimation parameters from askAdam are already masked and reshaped into [1*Nvoxel] arrays. This block essentially has two operations: (1) checks if the estimation parameters are masked, (2) if so then reshape the output ``S`` to G-D size. GACELLE provides a utility function ``reshape_ND2GD`` to simplify the reshaping operation of an array from N-D size to G-D size.

Note that the idea of the function design here is a bit different from :ref:`strategy1`: In Strategy 1, we make sure the output for the forward function is (4)N-D and let ``askadam.m`` deals with the masking internally. In Strategy 2, we accepted the masked estimation parameters in G-D format from ``askadam.m`` and make sure the output ``S`` is also in G-D format in the optimisation process. 

Note that the size of ``S`` can be G-D or N-D, depending on whether the estimation parameters are masked or not.

We can then simulate the 4D measurements using this function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 4-30

This time we also have a spherical non-zero mask to demonstrate the usage of a mask.

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 17

To estimate :math:`S0` and :math:`R_{2}^{*}` from y, 

1. Set up the starting point for the estimation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 34-35

2. Set up the model parameters and fitting boundary

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 37-42

3. Set up optimisation setting

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 43-50

4. Define the forward function

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 52-53

5. Define fitting weights (optional)

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 55-56

6. Start the optimisation

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 58-59

7. Plot the estimation results

.. literalinclude:: ../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m
    :language: matlab
    :lines: 61-74

The full example script can be found in `here <../../examples/Example_monoexponential_estimate_askadam_3D_Strategy2.m>`_.