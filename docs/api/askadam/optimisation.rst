.. _api-askadam-optimisation:
.. role::  raw-html(raw)
    :format: html

askadam.optmisation
===================

Usage
-----

.. code-block::

    obj = askadam;
    out = obj.optimisation( data, mask, weights, parameters, fitting, FWDfunc, varargin);

I/O overview
------------

+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| Input                            | Description                                                                                                  |
+==================================+==============================================================================================================+
| data                             | (Masked) N-D (imaging) data                                                                                  |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| mask                             | (1-3)D signal mask applied on FWDfunc, **NOTE this mask does NOT apply on data**                             |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| weights                          | N-D weights, same dimension as 'data' (optional)                                                             |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| parameters                       | structure variable containing starting points of all model parameters to be estimated (optional)             |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting                          | structure contains fitting algorithm parameters                                                              |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.optimiser                | Algorithm for parameter update, 'adam' (default) | 'sgdm' | 'rmsprop'                                        |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.model_params             | 1xM cell variable, name of the model parameters, e.g. {'S0','R2star'}                                        |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lb                       | 1xM numeric variable, fitting lower bound, same order as field 'model_params', e.g. [0.5, 0]                 |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.ub                       | 1xM numeric variable, fitting upper bound, same order as field 'model_params', e.g. [2, 1]                   |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.isDisplay                | boolean, display optimisation process in graphic plot                                                        |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.initialLearnRate         | (initial) learn rate of Adam optimiser, default = 0.001                                                      |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.iteration                | maximum number of optimisation iterations, default = 4000                                                    |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.tol                      | stop if total loss < tol, default = 1e-3                                                                     |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lambda                   | regularisation parameter(s), default = 0 (no regularisation)                                                 |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.regmap                   | model parameter(s) to which regularisation is applied                                                        |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.TVmode                   | mode for total variation (TV) regularisation, '2D' (default) | '3D'                                          |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.lossFunction             | loss function for data fidelity term, 'L1' (default) | 'L2' | 'huber' | 'mse'                                |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.randomness               | randomness of starting point; 0 = fixed (default), 1 = fully random                                          |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| fitting.debug                    | display extra messages and enable GPU memory tracking, default = false                                       |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| FWDfunc                          | function handle for forward signal generation; output size must match size of 'data'                         |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+
| varargin                         | additional input for FWDfunc other than 'parameter' and 'mask'                                               |
+----------------------------------+--------------------------------------------------------------------------------------------------------------+

+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| Output                        | Description                                                                                                  |
+===============================+==============================================================================================================+
| out                           | structure contains optimisation result                                                                       |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final                     | output structure at final iteration                                                                          |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final.loss                | total loss = loss_fidelity + loss_reg                                                                        |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final.loss_fidelity       | loss of data consistency term                                                                                |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final.loss_reg            | loss of regularisation term                                                                                  |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final.(model_params{k})   | estimated model parameter(s)                                                                                 |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.min                       | output structure at minimum loss iteration                                                                   |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.min.(model_params{k})     | estimated model parameter(s) at minimum loss iteration                                                       |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+
| out.final.memoryUsage         | estimated GPU memory usage in GB (requires fitting.debug = true for full tracking)                           |
+-------------------------------+--------------------------------------------------------------------------------------------------------------+

Stopping criteria
-----------------

``askadam`` supports multiple stopping criteria that can be used independently or in combination.
The optimisation terminates when **any** active criterion is satisfied.

Basic stopping criteria
~~~~~~~~~~~~~~~~~~~~~~~

These are always active.

+-----------------------------+----------+------------------------------------------------------------------------------------------+
| Option                      | Default  | Description                                                                              |
+=============================+==========+==========================================================================================+
| fitting.iteration           | 4000     | Stop when the maximum number of iterations is reached                                    |
+-----------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.tol                 | 1e-3     | Stop when total loss falls below this threshold                                          |
+-----------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.convergenceValue    | 1e-8     | Stop when the convergence signal falls below this threshold for ``patienceConvergence``  |
|                             |          | consecutive checks                                                                       |
+-----------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.patienceConvergence | 5        | Number of consecutive checks below ``convergenceValue`` required before stopping         |
+-----------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.patience            | 5        | Shared default for all patience counters; individual patience values override this       |
+-----------------------------+----------+------------------------------------------------------------------------------------------+

Convergence model
~~~~~~~~~~~~~~~~~

Controls how the convergence signal is computed from the loss. Applies to the loss-based stopping criterion above.

+---------------------------+----------+------------------------------------------------------------------------------------------+
| Option                    | Default  | Description                                                                              |
+===========================+==========+==========================================================================================+
| fitting.convergenceModel  | 'ema'    | Method for computing convergence signal from loss history.                               |
|                           |          | ``'linear'``: slope of loss over last ``convergenceWindow`` iterations.                  |
|                           |          | ``'ema'``: relative change in exponential moving average (EMA) of loss —                 |
|                           |          | more robust to short-term oscillations.                                                  |
+---------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.convergenceWindow | 20       | Number of iterations used to compute slope (``'linear'`` model only)                     |
+---------------------------+----------+------------------------------------------------------------------------------------------+
| fitting.emaDecay          | 0.95     | EMA decay factor (``'ema'`` model only); higher values smooth more aggressively          |
+---------------------------+----------+------------------------------------------------------------------------------------------+

Robust convergence (v1.1)
~~~~~~~~~~~~~~~~~~~~~~~~~

When enabled, detects voxels that are not improving relative to the rest of the population
and downweights their contribution to the gradient computation. The convergence signal is
then computed on the main (non-outlier) population only, preventing a small number of
stuck voxels from masking genuine convergence of the majority.

Outlier classification is based on two independent criteria, both of which must be satisfied
for a voxel to be flagged:

- **Criterion A**: the voxel has improved by less than ``outlierVoxelThres`` over the last
  ``outlierCheckWindow`` checks, while the median voxel has improved by more than ``outlierPopThres``.
- **Criterion B**: the voxel has improved by less than ``outlierInitThres`` relative to its
  own loss at initialisation, while the median voxel has improved by more than ``outlierInitPopThres``.

Once flagged, a voxel remains downweighted for at least ``outlierMinFlagDuration`` checks
before it can be reinstated, giving the downweighting time to take effect.

.. note::

    Outlier downweighting applies to the data fidelity gradient only. TV regularisation
    gradients are unaffected. The outlier classification lags by one ``weightUpdateInterval``
    because ``extractdata`` breaks the autodiff graph — this is intentional.

+--------------------------------+----------+--------------------------------------------------------------------------------------+
| Option                         | Default  | Description                                                                          |
+================================+==========+======================================================================================+
| fitting.robustConvergence      | false    | Enable robust convergence mode                                                       |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierWeight          | 0.1      | Gradient contribution of outlier voxels relative to main population (0-1)            |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.weightUpdateInterval   | 5        | Number of iterations between outlier mask and weight updates                         |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierCheckWindow     | 5        | Number of checks used to assess improvement in criterion A                           |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierMinFlagDuration | 5        | Minimum number of checks a voxel remains flagged before reassessment                 |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierVoxelThres      | 0.01     | Criterion A: minimum fractional improvement required per voxel (1%)                  |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierPopThres        | 0.05     | Criterion A: minimum fractional improvement required for median voxel (5%)           |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierInitThres       | 0.05     | Criterion B: minimum fractional improvement from initialisation per voxel (5%)       |
+--------------------------------+----------+--------------------------------------------------------------------------------------+
| fitting.outlierInitPopThres    | 0.20     | Criterion B: minimum fractional improvement from initialisation for median (20%)     |
+--------------------------------+----------+--------------------------------------------------------------------------------------+

Additional convergence signals (v1.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These optional signals provide additional stopping criteria independent of ``robustConvergence``.
Each is disabled by default (value = 0) and activates when set to a positive value.
Each uses the same ``patience`` mechanism as the loss-based criterion.

**Step norm** (analogous to ``StepTolerance`` in ``lsqnonlin``):

Stops when the relative norm of the parameter update step falls below threshold,
indicating that parameters have effectively stopped moving:

.. math::

    \frac{\| \Delta\theta \|_2}{1 + \| \theta \|_2} < \texttt{convergenceStepTol}

+------------------------------+----------+-------------------------------------------------------------------------------------+
| Option                       | Default  | Description                                                                         |
+==============================+==========+=====================================================================================+
| fitting.convergenceStepTol   | 0        | Relative step norm threshold; 0 = disabled                                          |
+------------------------------+----------+-------------------------------------------------------------------------------------+
| fitting.patienceStep         | 5        | Consecutive checks below threshold required before stopping                         |
+------------------------------+----------+-------------------------------------------------------------------------------------+

**Gradient norm**:

Stops when the raw gradient norm (before Adam correction) falls below threshold,
indicating that the loss landscape is genuinely flat:

+------------------------------+----------+-------------------------------------------------------------------------------------+
| Option                       | Default  | Description                                                                         |
+==============================+==========+=====================================================================================+
| fitting.convergenceGradTol   | 0        | Gradient norm threshold; 0 = disabled                                               |
+------------------------------+----------+-------------------------------------------------------------------------------------+
| fitting.patienceGrad         | 5        | Consecutive checks below threshold required before stopping                         |
+------------------------------+----------+-------------------------------------------------------------------------------------+

.. note::

    The step norm and gradient norm signals are complementary. The step norm catches
    parameter stagnation; the gradient norm catches loss landscape flatness. With Adam,
    a small step norm does not necessarily imply a small gradient norm since Adam
    normalises gradients via its second moment estimate.

Example: enabling robust convergence with EMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: matlab

    fitting.convergenceModel    = 'ema';       % use EMA-smoothed convergence signal
    fitting.robustConvergence   = true;        % enable outlier-aware convergence
    fitting.outlierWeight       = 0.1;         % outlier voxels contribute 10% gradient weight
    fitting.weightUpdateInterval = 5;          % update outlier mask every 5 iterations

    obj = askadam;
    out = obj.optimisation(data, mask, weights, parameters, fitting, FWDfunc, varargin);

See also :ref:`gettingstarted-askadam_basic_tutorial`.