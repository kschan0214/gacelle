.. _tipsandtricks-general:
.. role::  raw-html(raw)
    :format: html

Tips and Tricks
===============

1. When designing the forward signal function, try to avoid potential division of zeros which may cause gradient to explode during the optimisation. Oen way to prevent this is to the minimum of a variable to a very small value. For example, assuming the minimum value of a is 0, we can add an extra line

    ``a = min(a,1e-8);``

    to avoid 'a' to be 0 in some scenarios.

2. For MCMC, replacing some of the operations in forward signal simulation using the ``arrayfun`` `function <https://www.mathworks.com/help/parallel-computing/gpuarray.arrayfun.html>`_ can significantly speed up the computation.

3. Use single precision instead of double precision as suggested by `Matlab <https://www.mathworks.com/help/parallel-computing/measure-and-improve-gpu-performance.html#mw_1ffb4887-62d8-40cd-aaff-b694a85ccc62>`_.

4. Check your forward function to ensure there is no division-by-zero operations as this can cause NaN or Inf in the gradients or signals which affects the optimisation. Generally speaking, using operations like ``y = max(y,1e-8)`` will replace the zeros by a very small value.
