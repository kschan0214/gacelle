.. _api-mcmc_bayes-estimate_hyper_subset:
.. role::  raw-html(raw)
    :format: html

mcmc_bayes.estimate_hyper_subset
================================

*Experimental.* Learns the hierarchical (population) prior from a random subset of voxels, and returns a ``fitting`` structure with the prior fixed, for use on the whole volume. With a fixed prior the voxels are independent, so the whole-volume fit can be segmented by the memory manager. This is a way to use the hierarchical prior on volumes too large for one GPU call.

Usage
-----

.. code-block:: matlab

    obj = mcmc_bayes;
    fitting.prior.hierarchical.subsetFraction = 0.2;       % 20% of the masked voxels
    [muHat, SigmaHat, fittingFixed, outSub, idxSub, piHat] = ...
        obj.estimate_hyper_subset( data, mask, weights, parameters, fitting, FWDfunc, varargin);

    % whole volume with the fixed population prior
    out = obj.optimisation( data, mask, weights, parameters, fittingFixed, FWDfunc, varargin);

I/O overview
------------

Inputs as :ref:`api-mcmc_bayes-optimisation`. ``fitting.prior.hierarchical`` must be set and not fixed; ``.subsetFraction`` in ``(0, 1]`` is the fraction of masked voxels used (at least ``min(Nmask, 10)``). The subset is drawn with the global random stream. With segmentation labels (``prior.hierarchical.labels``) it is drawn per label group, ``min(n_k, max(ceil(subsetFraction*n_k), 10, 2d))`` voxels of each, and ``fittingFixed`` keeps the labels of the whole volume with ``.labelValues`` of the groups. ``varargin`` is passed unchanged, so it must not contain inputs with a voxel dimension.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Output
     - Description
   * - ``muHat``
     - posterior mean of the population mean, ``[d,1]`` (``[d,K]``), transformed space
   * - ``SigmaHat``
     - posterior mean of the population covariance, ``[d,d]`` (``[d,d,K]``)
   * - ``fittingFixed``
     - ``fitting`` with ``prior.hierarchical.fixed = true``, ``.mu``, ``.Sigma`` (``.pi``) set and ``.subsetFraction = 1``
   * - ``outSub``
     - output of the subset run
   * - ``idxSub``
     - linear indices (into ``mask``) of the subset voxels
   * - ``piHat``
     - posterior mean of the group weights ``[K,1]`` (1 for ``K = 1``)
