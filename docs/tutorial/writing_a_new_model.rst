.. _tutorial-writing_a_new_model:
.. role::  raw-html(raw)
    :format: html

Writing a new fitting class
=============================

Every GPU model class in GACELLE (``gpuNEXI``, ``gpuAxCaliberSMT``, ``gpuGREMWI``, ...) follows the same skeleton, so that it plugs into ``askadam``/``mcmc``, GACELLE's automatic GPU memory manager, and the standard ``estimate(data, mask, ...)`` calling convention shared across the toolbox. This tutorial builds that skeleton step by step, using the simplest built-in model, ``gpuR2starMapping``, as a complete worked example.

``gpuR2starMapping`` fits a single-compartment mono-exponential decay,

.. math::

    S(t) = M_0 \, e^{-R_2^* t},

to magnitude multi-echo GRE data. It has no diffusion terms, no fibre geometry, no MCMC-specific machinery beyond the minimum required to be MCMC-*compatible* - which makes it the clearest possible illustration of the required structure, without a more complex model's biophysics obscuring it.

For style conventions specific to writing ``FWD()`` functions (dlarray/dlaccelerate compatibility, avoiding in-place indexing, etc.), see :doc:`designing_model`. For how your finished class participates in segmentation for large volumes, see :doc:`../advanced/automatic_memory_management`.

What a model class needs to provide
-------------------------------------

At minimum, a GACELLE model class needs:

- A **model parameter contract**: four (or five, if MCMC is supported) index-aligned properties naming the parameters being fitted and their bounds/starting points.
- A **constructor** that stores acquisition parameters (echo times, b-values, flip angles, ...) that are fixed for the whole dataset, as opposed to being fitted per voxel.
- ``check_set_default(fitting)``: fills in model-specific defaults on top of ``askadam.check_set_default_basic``/``mcmc.check_set_default_basic``.
- ``updateProperty(fitting)``: prunes the parameter contract for the current solver/options (e.g. drops a solver-conditional ``noise`` parameter when not using MCMC).
- ``FWD(this, pars, ...)``: the forward signal model, differentiable and dlaccelerate-compatible.
- ``determine_x0``/an equivalent starting-point routine.
- ``fit(data, mask, fitting, ...)``: the per-segment worker that assembles everything above and calls ``askadam().optimisation(...)`` or ``mcmc().optimisation(...)``.
- ``estimate(data, mask, fitting, ...)``: the public entry point - validates/normalises input, drives the memory-manager segmentation loop, calls ``fit()`` per segment, and saves the output.

The rest of this page builds each of these in order.

Step 1: the model parameter contract
--------------------------------------

Every model class declares five index-aligned arrays as protected properties: the parameter names, their upper/lower bounds, their default starting points, and (for MCMC) a per-parameter proposal step size.

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 8-29

Two conventions matter here, and both are load-bearing rather than stylistic:

- **The five arrays must stay the same length and index-aligned.** ``modelParams{k}``, ``ub(k)``, ``lb(k)``, ``startPoint(k)``, and ``step(k)`` all describe the same, k-th parameter. Never assign into a single element of one array from outside the class - always mutate the set together, which is exactly what ``updateProperty()`` (Step 3) does.
- **A solver-conditional parameter goes last.** ``noise`` is only meaningful for the MCMC solver (it isn't a free parameter of the forward model itself - it's the likelihood's noise term). Keeping it last lets ``updateProperty()`` strip it by name (``ismember(this.modelParams,'noise')``) without hardcoding an index that would desync the moment a parameter is added or reordered.

Step 2: the constructor
--------------------------

The constructor stores whatever acquisition information is fixed across the whole dataset - here, just the echo times:

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 54-59

Keep the constructor minimal. Anything that varies per voxel (the actual data, a mask, per-voxel priors) is an argument to ``estimate()``/``fit()``, not something the constructor should ever see. Acquisition properties are typically declared ``GetAccess = public, SetAccess = protected`` (as ``te`` is above), so callers can inspect them but not mutate them after construction.

Step 3: pruning the parameter contract - ``updateProperty()``
-----------------------------------------------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 77-90

This is where the "solver-conditional parameter goes last" convention from Step 1 pays off: stripping ``noise`` from all five arrays at once, by name, keeps them aligned regardless of how many parameters precede it. If your model has other conditionally-fitted parameters (e.g. a parameter that's only free when a particular ``fitting`` flag is set), prune them here the same way, always as a matched set across all the arrays you mutate.

.. warning::
   ``updateProperty()`` only has an effect if something actually calls it. It must be invoked from ``fit()``, after ``check_set_default()`` has resolved ``fitting.solver`` and before ``fitting.modelParams`` is read off ``this.modelParams`` - see the callout in Step 7 below, which walks through a real instance of this being forgotten.

Step 4: fitting defaults - ``check_set_default()``
------------------------------------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 462-486

The pattern is always the same: default ``fitting.solver`` to ``'askadam'`` if absent, call the corresponding base-class defaulter (``askadam.check_set_default_basic`` or ``mcmc.check_set_default_basic`` - see :ref:`api-askadam-optimisation`) to fill in every generic optimisation option, and then layer your model-specific options on top with plain ``if ~isfield(fitting,'X'); fitting2.X = default; end`` guards. Keep this layering order: generic-first, model-specific-second, so a model's custom default can freely reference or override anything the base defaulter set.

Step 5: the forward model - ``FWD()``
----------------------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 320-363

And the actual signal equation it calls:

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 426-434

``FWD()`` is the one function that must work under automatic differentiation (``dlgradient``) and, for the askAdam path, under ``dlaccelerate``. In practice that means:

- Read parameters out of the ``pars`` struct by field name (``pars.M0``, ``pars.R2star``), never by iterating/indexing into it generically.
- Prefer elementwise array operations (as ``model_R2s`` does) over control flow that branches on parameter *values* - branching on ``solver``/``fitting`` (both fixed for the whole call, not learnable) is fine, as seen above.
- Reshape the output to match the measurement layout the solver expects: ``utils.reshape_ND2GD(s,[])`` here, for both the askAdam and MCMC branches, once the signal has been generated in N-D form.
- Whatever the true output units are, they must match what ``fit()`` hands to ``askadam().optimisation()``/``mcmc().optimisation()`` as the measurement - see :doc:`designing_model` for the full set of conventions (avoiding in-place indexing that breaks the autodiff graph, boolean-mask compaction pitfalls with ``dlaccelerate``, and so on).

Step 6: starting points
--------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 256-316

``fitting.start`` dispatches between three sources of starting points, and every model class should support the same three, for consistency across GACELLE:

- ``'prior'``: a closed-form or otherwise cheap per-voxel estimate (here, a trapezoidal-integration R2* fit) computed in ``estimate_prior()``.
- ``'default'``: the fixed ``startPoint`` values from Step 1, broadcast to every voxel.
- a user-supplied array: one value (or map) per parameter, in the order given by ``this.modelParams``.

Whichever source is used, the result is clamped to ``[fitting.lb, fitting.ub]`` via ``askadam.set_boundary`` before being handed off - starting points from a closed-form estimate are not guaranteed to respect the fitting bounds, and this is the one place that's checked.

Step 7: the per-segment worker - ``fit()``
----------------------------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 181-245

This function assembles everything from Steps 1-6 and hands off to the solver. The sequence that matters:

1. Resolve defaults (``check_set_default``).
2. **Prune the parameter contract for the resolved solver** - call ``this = this.updateProperty(fitting);`` here, then read ``fitting.modelParams = this.modelParams;`` off the *pruned* result.
3. Fall back to the class's own ``ub``/``lb`` if the caller didn't supply them.
4. Determine starting points (Step 6).
5. Compute optimisation weights, if your model supports weighting (optional - see the utility methods in Step 9).
6. Dispatch on ``fitting.solver``: ``askadam().optimisation(data, mask, w, pars0, fitting, @this.FWD, ...)`` or the equivalent ``mcmc()`` call, passing through whatever extra positional arguments your ``FWD()`` needs after ``pars`` (here, ``fitting.solver`` itself, so ``FWD`` can branch on it).

.. note::
   ``gpuR2starMapping.m`` now includes the ``this = this.updateProperty(fitting);`` call shown as step 2 above, immediately after ``check_set_default`` and before ``fitting.modelParams`` is read - this is exactly the sequence a new class should follow. An earlier revision of this file omitted it, which meant ``noise`` stayed in the parameter set on the askAdam path even though ``FWD()`` never reads it - fitted as a free parameter that did nothing, at the cost of GPU memory and iteration time, and surfaced as a meaningless value in ``out.final.noise``. Worth keeping in mind precisely because it's easy to define ``updateProperty()`` (Step 3) and forget to wire it in here.

Step 8: the public entry point - ``estimate()``
---------------------------------------------------

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 93-179

``estimate()`` is what a user (or another model class, for a composite model) actually calls. Its responsibilities are distinct from ``fit()``'s and shouldn't be blurred together:

1. Display basic info, resolve defaults.
2. **Validate and normalise the data** (``prepare_data`` - Step 9). Rescaling to a sensible numeric range here, and undoing it on the output afterwards (see the ``scaleFactor`` handling below), keeps the optimiser's step sizes and tolerances meaningful regardless of the input data's raw intensity scale.
3. **Drive the memory-manager segmentation loop**: ``utils.find_optimal_segment_3D`` decides how many segments the volume needs to fit in GPU memory, then a loop over segments calls ``this.slice_segment`` to extract each chunk, ``this.fit`` to run the actual optimisation on it, and ``utils.restore_segment_structure`` to stitch results back into a single output. This is the entire mechanism by which automatic memory management works - your model class doesn't need to know anything about GPU memory limits itself, as long as ``fit()`` operates on whatever-sized chunk it's handed. See :doc:`../advanced/automatic_memory_management` for the mechanism in full.
4. **Undo any normalisation and save**, dispatching on solver the same way ``fit()`` does, since the askAdam and MCMC output structures differ (``out.final``/``out.min`` vs ``out.posterior``/``out.(fitting.metric{k})``).

Step 9: small supporting utilities
--------------------------------------

A few more methods round out the class. None of these are unique to this model - most classes need equivalents of all three:

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 367-421

- ``validate_input``: fail fast and loudly on shape mismatches (echo count vs data, mask vs data) rather than letting a cryptic error surface deep inside the optimiser. Falls back to a simple intensity-threshold mask if none is supplied.
- ``prepare_data``: normalises the data to a numerically sensible range (98th-percentile scaling here) and removes NaN/Inf voxels from the mask. The corresponding *undo* step lives in ``estimate()`` (Step 8), not here - keep the two symmetric.
- ``slice_segment``: extracts one segment's worth of data/mask given the index ranges the memory manager computed. This must match whatever N-D layout your ``data`` actually has - note the trailing ``,:,:,:,:,:,:,:`` here, which tolerates extra trailing dimensions beyond the third (spatial) one.

Optional but common: a weighting scheme for the data-fidelity term, and a matching ``display_algorithm_info`` so the console output documents what's actually happening for the current run:

.. literalinclude:: ../../R2star/gpuR2starMapping.m
    :language: matlab
    :lines: 488-529

Checklist
-----------

Before considering a new model class finished:

- [ ] ``modelParams``/``ub``/``lb``/``startPoint``/``step`` are the same length and index-aligned; any solver-conditional parameter is last.
- [ ] ``updateProperty()`` is both defined **and called** from ``fit()``, before ``fitting.modelParams`` is read off ``this.modelParams`` (see the note in Step 7 for what happens if this is skipped).
- [ ] ``check_set_default()`` calls the appropriate base-class defaulter first, then layers model-specific defaults on top.
- [ ] ``FWD()`` only reads ``pars`` by field name, avoids in-place/logical indexing that breaks the autodiff graph, and its output shape matches what the solver expects for both the askAdam and MCMC branches if both are supported.
- [ ] ``fitting.start`` supports at least ``'prior'``, ``'default'``, and a user-supplied array, for consistency with the rest of GACELLE.
- [ ] Starting points are clamped to ``[lb,ub]`` before being handed to the solver.
- [ ] ``estimate()`` uses ``utils.find_optimal_segment_3D`` + ``slice_segment`` + ``restore_segment_structure`` rather than assuming the whole volume fits on the GPU at once.
- [ ] Any data normalisation done in ``prepare_data`` is undone symmetrically on the output in ``estimate()``.
- [ ] In-source docstrings actually describe this class's arguments and behaviour, not a different class's - copy-pasting a docstring from another model file and forgetting to update the argument list/shapes is a recurring, easy-to-miss source of documentation bugs across this codebase.

Documenting the finished class
----------------------------------

Once the class itself is working, the documentation page follows a standard shape used across every model in :ref:`supportedmodels-R2starMapping` and its neighbours: a short intro paragraph plus reference, a ``Usage`` code block, a ``Model parameters`` section built with ``literalinclude`` against the raw ``modelParams``/``ub``/``lb``/``startPoint`` arrays (so the docs can't drift out of sync with the code the way a hand-copied table can), an ``I/O overview`` table for the constructor and ``estimate()`` arguments, any notes on solver support or non-obvious behaviour, and an example script. Use an existing page as a template rather than starting from a blank file.

See also :ref:`supportedmodels-R2starMapping`, :doc:`designing_model`, and :doc:`../advanced/automatic_memory_management`.
