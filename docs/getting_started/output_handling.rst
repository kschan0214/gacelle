.. _output_handling:

Understanding the output
========================

Both solvers return a single structure, ``out``. The organisation differs
between ``askadam.m`` and ``mcmc.m``, reflecting the point-estimate nature of
the former and the distributional nature of the latter.

.. important::

   **Parameter maps are returned on the full input grid.** Even when
   ``fitting.isOptimiseMemory = true`` (the default) causes parameters to be
   masked to ``[1 x Nvoxel]`` internally, they are unmasked before being
   returned. Voxels outside the signal mask are set to **zero**. Users do not
   need to unmask the estimated maps themselves.

   **Diagnostic fields are not all image-shaped.** See the tables below:
   ``out.final.residual`` (``askadam.m``) and ``out.posterior``
   (``mcmc.m``) are returned in masked, vectorised form, not on the image grid.

askadam.m
---------

Results are reported at two points: the **final** iteration and the iteration
with the **minimum loss** (which need not be the last one).

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Field
     - Contents
     - Size
   * - ``out.final.<param>``
     - Estimate at the final iteration
     - Input grid (masked voxels = 0)
   * - ``out.final.loss``
     - Total loss (fidelity + regularisation)
     - scalar
   * - ``out.final.loss_fidelity``
     - Data fidelity term
     - scalar
   * - ``out.final.loss_reg``
     - Regularisation term (0 if unused)
     - scalar
   * - ``out.final.resloss``
     - Mean residual per voxel
     - Input grid
   * - ``out.final.residual``
     - Residual per measurement per voxel
     - ``[Nmeas x Nvoxel]``
   * - ``out.final.Niteration``
     - Iterations performed
     - scalar
   * - ``out.final.memoryUsage``
     - Peak GPU memory used (GB)
     - scalar
   * - ``out.min.<...>``
     - As above, at the minimum-loss iteration
     - (no ``memoryUsage``)

``<param>`` are the names given in ``fitting.modelParams``, e.g.
``out.final.S0``, ``out.final.R2star``.

.. note::

   Parameters whose first three dimensions do not match the mask (e.g. global
   or scalar parameters shared across voxels) are returned unmasked, as-is.

mcmc.m
------

``mcmc.m`` always returns the retained posterior samples, plus any summary
metrics requested via ``fitting.metric``.

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Field
     - Contents
     - Size
   * - ``out.posterior.<param>``
     - Retained posterior samples
     - ``[Nvoxel x Nsample x Nrepetition]`` (Metropolis-Hastings);
       ``[Nvoxel x Nwalker x Nsample x Nrepetition]`` (ensemble)
   * - ``out.mean.<param>``
     - Posterior mean
     - Input grid
   * - ``out.std.<param>``
     - Posterior standard deviation
     - Input grid
   * - ``out.median.<param>``
     - Posterior median
     - Input grid
   * - ``out.iqr.<param>``
     - Posterior interquartile range
     - Input grid
   * - ``out.mode.<param>``
     - Posterior mode (histogram, ``fitting.Nbin`` bins)
     - Input grid

.. warning::

   ``out.posterior`` is **not** on the image grid: its first dimension indexes
   only the voxels inside the mask. Summary metrics (``mean``, ``std``, ...)
   *are* on the image grid.

Which metrics are computed is controlled by ``fitting.metric``
(default ``{'mean','std'}``)::

   fitting.metric = {'mean','std','median','iqr'};

Fields not listed in ``fitting.metric`` are absent from ``out``.
