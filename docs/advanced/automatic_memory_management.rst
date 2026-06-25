.. _automatic_memory_management:

Automatic GPU Memory Management
================================

When processing large datasets, GPU memory can become a bottleneck. GACELLE includes an automatic memory manager that detects whether the full dataset fits in available GPU memory and, if not, transparently segments the data into smaller chunks that are processed sequentially. This page explains how the mechanism works, what its limitations are, and what users should expect.

.. note::
   Automatic memory management is currently supported for the built-in model classes (e.g. ``gpuMCRMWI``, ``gpuJointR1R2starMapping``) and applies to both the ``askadam.m`` and ``mcmc.m`` solvers. It is activated by setting ``fitting.autoMemManage = 1`` before calling the model's ``fit`` method. Two further options, ``fitting.segmentOverlap`` and ``fitting.NSegmentUser``, give manual control over segment overlap and segment count; see :ref:`memmanage-manual-control` below.

Why Segmentation Is Needed
----------------------------

GACELLE's ``askadam.m`` solver optimises all masked voxels simultaneously within a single objective function, and ``mcmc.m`` runs all chains in parallel across voxels. Both approaches are what enable GPU parallelism, but they also mean that GPU memory scales with the number of voxels being processed. For ``askadam.m``, MATLAB must additionally materialise the full autodiff computation graph during the backward pass, which can require several times the memory of the forward pass alone. For ``mcmc.m``, the parallel chain storage and likelihood evaluations across all voxels similarly grow with dataset size.

For typical in vivo datasets at 1–2 mm isotropic resolution, this is usually not a problem. For high-resolution or high-dimensional datasets (e.g. many diffusion shells, many echo times), or on GPUs with limited VRAM, the full volume can exceed available memory. Naively reducing the matrix size or disabling regularisation are poor solutions. The automatic memory manager instead determines the largest safe workload empirically and splits the data accordingly, without any change to the user's fitting script.

How It Works
-------------

The memory manager is implemented in ``utils.find_optimal_segment_3D``, which runs before the main optimisation whenever segmentation might be needed: either because ``fitting.autoMemManage = 1`` (the probe-based path described below), or because ``fitting.NSegmentUser`` is set to force a minimum segment count regardless of ``autoMemManage`` (see :ref:`memmanage-manual-control`). This section describes the three-stage probe path; the manual override is covered separately.

**Stage 1: Probe fitting**

Two small *probe fits* are run on sub-samples of the brain mask: one on a minimum of 100 voxels and one on up to 10% of the total masked voxels (capped at 100,000). For each probe, a background ``nvidia-smi`` process logs GPU memory use at 5 ms intervals into a temporary CSV file. The MATLAB process ID is used to separate MATLAB's memory footprint from that of other GPU processes already running on the system (e.g. a display server or another user's job). The peak MATLAB-only memory for each probe is computed as::

   matlabPeak_MB = totalPeak_MB - max(otherMem_before_MB, otherMem_after_MB)

where ``totalPeak_MB`` is read from the nvidia-smi log (discarding the first 5% of samples to avoid transient artefacts), and the other-process contribution is taken as the more conservative of the snapshots before and after the probe fit.

.. note::
   ``nvidia-smi`` polls at 5 ms intervals, so the peak estimate is an approximation. It captures the gradient materialisation spike during the backward pass, which ``gpuDevice().AvailableMemory`` cannot reliably measure because it only reflects instantaneous free memory rather than the true peak.

**Stage 2: Memory prediction**

A linear model is fitted to the two (probe size, peak memory) data points::

   mem_matlab_peak ≈ slope × Nvoxels + intercept

where ``slope`` captures the per-voxel memory cost (including autodiff overhead) and ``intercept`` captures the fixed MATLAB/CUDA overhead that is independent of data size. This linear model is then extrapolated to the full voxel count to predict the peak memory the full fit would require.

The available VRAM is queried via ``utils.get_available_vram()``, which reads the current free memory from nvidia-smi (not from MATLAB) so that memory already occupied by other processes is properly accounted for.

**Stage 3: Segmentation decision**

- If the predicted peak is within the available VRAM budget, no segmentation is applied and the full volume is processed in a single pass.
- If the predicted peak exceeds the budget, the maximum number of voxels that can safely fit in one segment is computed by inverting the linear model::

     NvoxPerSegFit = floor((memAvail_MB - intercept) / slope)

  This budget is for the segment as actually *fitted*, which includes any halo slices (see :ref:`memmanage-manual-control`). The halo cost is subtracted first, using the mean masked-voxel density per slice and the worst case of two halo faces, to give a smaller, conservative target for the voxels each segment *owns*::

     NvoxPerSegOwned = max(1, NvoxPerSegFit - 2 * segmentOverlap * meanSliceVox)

  The brain mask is then divided into **density-balanced owned slice groups** using ``utils.build_balanced_boundaries``, which partitions the volume along the slice (z) dimension so each segment owns at most ``NvoxPerSegOwned`` masked voxels, with boundaries chosen to keep segment sizes as equal as possible. ``utils.expand_segments_with_halo`` then pads each segment's owned range with up to ``segmentOverlap`` halo slices on its internal faces (never past the true volume boundary) to get the range actually fitted.

The function returns a struct array ``seg``, one entry per segment, with three fields:

- ``seg(k).owned`` — global slice indices this segment is responsible for. Disjoint across segments and exactly partitions the full volume; used when writing results back.
- ``seg(k).fit`` — global slice indices actually extracted and fitted, i.e. ``.owned`` padded with halo slices.
- ``seg(k).local`` — position of ``.owned`` within ``.fit``, precomputed so calling code never has to re-derive the offset.

With no overlap (``segmentOverlap = 0``, the default), ``.fit`` and ``.owned`` are identical and this reduces exactly to the previous behaviour. The model's ``fit`` method loops over ``seg``, fits each segment's ``.fit`` range, crops the output down to ``.owned`` via ``.local``, and reassembles the full parameter maps.

Enabling Automatic Memory Management
--------------------------------------

Set the flag in your fitting options before calling ``fit``::

   fitting.autoMemManage = 1;
   out = modelObj.fit(data, mask, fitting);

When the flag is active, GACELLE will print a brief report to the console::

   Checking GPU memory requirements...
     Probe 1/2 (N= 100 voxels): MATLAB peak = 312 MiB (total=1847, other=1535)
     Probe 2/2 (N=5000 voxels): MATLAB peak = 489 MiB (total=2024, other=1535)
   Memory prediction:
     Predicted MATLAB peak : 18432 MB
     Available VRAM (smi)  : 12288 MB
     Budget (100%)         : 12288 MB
   Data divided into 4 segments (target 62500 owned voxels/segment, halo=0 slices)
   The estimation may not be exactly the same as 1 segment.

If the full volume fits, you will instead see::

   Full data fits in GPU memory (predicted 9.2 GB / available 12.0 GB)

and a single-pass fit proceeds as normal.

.. _memmanage-manual-control:

Manual Control: Overlap and Forced Segment Count
---------------------------------------------------

Automatic detection is a prediction, not a guarantee, and the linear memory model can be wrong for some forward functions (see :ref:`memmanage-limitations` below). Two ``fitting`` fields let you override or assist it directly, and — importantly — **both work whether or not the automatic probe runs at all**, including with ``fitting.autoMemManage = 0``:

``fitting.segmentOverlap`` (default ``0``)
   Number of halo slices added on each *internal* segment boundary. When a spatial regulariser couples neighbouring slices (e.g. 3D TV), each segment normally loses correct neighbour information at its cut faces, producing mild discontinuities in the reconstructed map at segment boundaries (see :ref:`memmanage-limitations`). Setting ``segmentOverlap`` to a small positive integer (e.g. ``3``) gives every internal segment face that many slices of context borrowed from its neighbour; those halo slices are fitted but discarded on reassembly, so only the ``owned`` core of each segment is written to the output. The true volume boundary (the very first and last slice of the whole dataset) never gets a halo, since there is nothing to borrow there. The default of ``0`` reproduces the original no-halo behaviour exactly. If the halo width approaches half the thickness of the thinnest segment, GACELLE issues a ``GACELLE:haloVsOwnedThickness`` warning, since at that point a segment is spending most of its compute refitting its neighbours' slices rather than its own.

``fitting.NSegmentUser`` (default ``[]``)
   A user-requested **minimum** number of segments. This is a floor, not a fixed override: the final segment count is ``max(NSegmentUser, memoryRequiredSegments)``, so it can only ever push the segment count up, never down past what the memory model determined was necessary. Setting this cannot, by construction, cause an out-of-memory error that the automatic logic would otherwise have prevented.

   This is the option to reach for if automatic detection is unavailable or untrustworthy for your situation:

   - If the probe itself raises ``GACELLE:memoryError`` (predicted fixed overhead alone exceeds available VRAM) or you hit an out-of-memory error mid-run despite a "fits in memory" prediction, set ``fitting.autoMemManage = 0`` and ``fitting.NSegmentUser`` to a value you know is safe (e.g. from a previous successful run, or by estimating from your GPU's VRAM and the dataset's voxel count) and re-run.
   - ``NSegmentUser`` is honoured even when ``autoMemManage`` is off, or when the probe is skipped because the mask is too small to probe reliably (see below). In both of those cases the resulting segments are equal-thickness slabs rather than density-balanced, since there is no memory or voxel-density information available to balance against.
   - It is also useful on its own for validating that segmentation itself is not the source of an unexpected result: force a known segment count on a dataset that normally fits in one pass, and compare against the single-pass output.

Example — forcing two segments with a 3-slice halo, bypassing automatic detection entirely::

   fitting.autoMemManage  = 0;
   fitting.NSegmentUser   = 2;
   fitting.segmentOverlap = 3;
   out = modelObj.fit(data, mask, fitting);

Example — letting automatic detection decide segment count, but adding a halo in case 3D TV is active::

   fitting.autoMemManage  = 1;
   fitting.segmentOverlap = 3;
   out = modelObj.fit(data, mask, fitting);

When autoMemManage Is Skipped
-------------------------------

The probe stage disables itself (i.e., falls back to single-pass sizing) in two situations:

- The total number of masked voxels is smaller than the minimum probe size (100 voxels). There is nothing meaningful to segment.
- The larger probe size does not exceed the smaller one (e.g. 10% of the mask is fewer than 100 voxels). The linear fit would be degenerate.

In both cases, ``fitting.autoMemManage`` is silently set to 0 internally. This does **not** mean segmentation is impossible in this situation: ``fitting.NSegmentUser`` is still honoured as a floor even when the probe is skipped or ``autoMemManage`` is off, so if you need a forced segment count on a small mask (e.g. for testing), set ``NSegmentUser`` directly rather than relying on the probe. With no memory information available, the resulting segments are equal-thickness slabs rather than density-balanced.

.. _memmanage-limitations:

Limitations and Caveats
-------------------------

**Segmentation changes results slightly.** When the data is split into segments, each segment is optimised independently. For ``askadam.m``, because gradients are accumulated across all voxels in a single loss, splitting the volume changes the loss landscape: voxels at segment boundaries lose their neighbours during that segment's optimisation, and any spatial regularisation (e.g. TV) is applied within, not across, segment boundaries unless a halo is used. For ``mcmc.m``, chains are independent per voxel so the effect is more limited, but initialisation and any shared hyperparameters may differ slightly across segments. In practice, differences are small for typical segmentation counts, but results will not be numerically identical to a single-pass fit. This is noted explicitly in the console output.

**3D TV regularisation does not cross segment boundaries unless a halo is set.** If you are using 3D total variation regularisation with ``fitting.segmentOverlap = 0`` (the default), the regulariser is computed independently within each segment's owned slice group, which can introduce mild discontinuities at boundaries, particularly when segment counts are high. Setting ``fitting.segmentOverlap`` to a few slices (see :ref:`memmanage-manual-control`) gives each segment correct neighbour information across its cut faces and removes this discontinuity in practice, at the cost of refitting a few extra slices per internal boundary. 2D TV (applied within each axial slice) is unaffected by segmentation either way.

**Whole-image operations (e.g. Fourier transforms) need further testing.** Segmentation assumes the forward model's dependence on neighbouring voxels is local and slice-limited, which is what a halo of a few slices can correct for. Operations that act on the *entire* image volume at once, such as a 3D FFT or other transforms with global support, do not fit this assumption: a halo of any finite width cannot reproduce what a whole-volume transform would see, since every voxel in principle depends on every other voxel. Models or regularisers that include such whole-image operations have not yet been thoroughly validated under segmentation, with or without halo, and should be treated with caution until this is tested explicitly.

**The linear memory model is an approximation.** The probe-based extrapolation assumes that peak memory scales linearly with voxel count. This is a reasonable first-order model for most GACELLE applications, but it can underestimate memory for models with strongly nonlinear forward functions or when MATLAB's internal allocator fragments GPU memory. A conservative ``safetyFactor < 1`` can be applied to reduce the effective budget if you observe out-of-memory errors despite the manager predicting a fit.

**Memory prediction depends on the GPU state at probe time.** If GPU memory is heavily fragmented or occupied by other processes at the time of probing, the estimate of other-process overhead may be inaccurate, leading to a more conservative (more segments) or, less commonly, an overoptimistic (fewer segments) segmentation decision. Closing other GPU-intensive applications before running is recommended for reproducibility.

**Forward model internal allocations are not captured.** The probe measures peak memory during a zero-iteration fit (``fitting.iteration = 0``). Allocations made inside the forward model during a real optimisation run (e.g. temporary arrays in EPG-X or ANN inference) may add memory beyond what the probe captures. If you find that the predicted-safe segment count still causes OOM errors, set ``fitting.NSegmentUser`` to a higher value yourself (see :ref:`memmanage-manual-control`) rather than relying on the probe, or contact the developers.

**nvidia-smi must be available.** The memory manager relies on ``nvidia-smi`` being on the system path. On most Linux HPC systems this is the case; on Windows it may require adding the CUDA bin directory to the PATH.

Related Utilities
------------------

The memory manager calls the following internal utilities, which are documented in the API reference:

- ``utils.get_other_process_memory(pid)`` — returns the current GPU memory (MB) used by all processes except the MATLAB process with the given PID.
- ``utils.get_available_vram()`` — returns current free GPU VRAM (MB) as reported by nvidia-smi.
- ``utils.read_absolute_peak_from_log(logFile)`` — reads a nvidia-smi CSV log and returns the peak total GPU memory usage (MB), discarding the first 5% of samples as a warm-up period.
- ``utils.build_balanced_boundaries(mask, NvoxPerSeg, NSegmentMin)`` — partitions a 3D binary mask into owned slice groups such that each group contains at most ``NvoxPerSeg`` masked voxels, with group sizes as equal as possible. ``NSegmentMin`` (optional, default 1) is a floor on the number of segments, used to implement ``fitting.NSegmentUser``.
- ``utils.expand_segments_with_halo(ownedBoundaries, h, dims3)`` — takes the owned slice ranges from ``build_balanced_boundaries`` and pads each internal boundary with ``h`` halo slices to produce the ``seg`` struct array (``.owned``/``.fit``/``.local``) used by ``find_optimal_segment_3D``.