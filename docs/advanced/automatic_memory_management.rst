.. _automatic_memory_management:

Automatic GPU Memory Management
================================

When processing large datasets, GPU memory can become a bottleneck. GACELLE includes an automatic memory manager that detects whether the full dataset fits in available GPU memory and, if not, transparently segments the data into smaller chunks that are processed sequentially. This page explains how the mechanism works, what its limitations are, and what users should expect.

.. note::
   Automatic memory management is currently supported for the built-in model classes (e.g. ``gpuMCRMWI``, ``gpuJointR1R2starMapping``) and applies to both the ``askadam.m`` and ``mcmc.m`` solvers. It is activated by setting ``fitting.autoMemManage = 1`` before calling the model's ``fit`` method.

Why Segmentation Is Needed
----------------------------

GACELLE's ``askadam.m`` solver optimises all masked voxels simultaneously within a single objective function, and ``mcmc.m`` runs all chains in parallel across voxels. Both approaches are what enable GPU parallelism, but they also mean that GPU memory scales with the number of voxels being processed. For ``askadam.m``, MATLAB must additionally materialise the full autodiff computation graph during the backward pass, which can require several times the memory of the forward pass alone. For ``mcmc.m``, the parallel chain storage and likelihood evaluations across all voxels similarly grow with dataset size.

For typical in vivo datasets at 1–2 mm isotropic resolution, this is usually not a problem. For high-resolution or high-dimensional datasets (e.g. many diffusion shells, many echo times), or on GPUs with limited VRAM, the full volume can exceed available memory. Naively reducing the matrix size or disabling regularisation are poor solutions. The automatic memory manager instead determines the largest safe workload empirically and splits the data accordingly, without any change to the user's fitting script.

How It Works
-------------

The memory manager is implemented in ``utils.find_optimal_segment_3D`` and runs automatically before the main optimisation when ``fitting.autoMemManage = 1``. It proceeds in three stages.

**Stage 1: Probe fitting**

Two small *probe fits* are run on sub-samples of the brain mask: one on a minimum of 100 voxels and one on up to 10% of the total masked voxels (capped at 100,000). For each probe, a background ``nvidia-smi`` process logs GPU memory use at 5 ms intervals into a temporary CSV file. The MATLAB process ID is used to separate MATLAB's memory footprint from that of other GPU processes already running on the system (e.g. a display server or another user's job). The peak MATLAB-only memory for each probe is computed as::

   matlabPeak_MB = totalPeak_MB - max(otherMem_before_MB, otherMem_after_MB)

where ``totalPeak_MB`` is read from the nvidia-smi log (discarding the first 15% of samples to avoid transient artefacts), and the other-process contribution is taken as the more conservative of the snapshots before and after the probe fit.

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

     NvoxPerSeg = floor((memAvail_MB - intercept) / slope)

  The brain mask is then divided into **density-balanced slice groups** using ``utils.build_balanced_boundaries``. This partitions the 3D volume along the slice (z) dimension such that each segment contains at most ``NvoxPerSeg`` masked voxels, with boundaries chosen to keep segment sizes as equal as possible rather than splitting at fixed slice intervals.

The resulting slice boundaries are returned to the model's ``fit`` method, which loops over segments, fits each independently, and reassembles the output parameter maps.

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
   Data divided into 4 segments (max 62500 voxels/segment)
   The estimation will not be exactly the same as 1 segment.

If the full volume fits, you will instead see::

   Full data fits in GPU memory (predicted 9.2 GB / available 12.0 GB)

and a single-pass fit proceeds as normal.

When autoMemManage Is Skipped
-------------------------------

The manager disables itself (i.e., falls back to single-pass) in two situations:

- The total number of masked voxels is smaller than the minimum probe size (100 voxels). There is nothing meaningful to segment.
- The larger probe size does not exceed the smaller one (e.g. 10% of the mask is fewer than 100 voxels). The linear fit would be degenerate.

In both cases, ``fitting.autoMemManage`` is silently set to 0 internally and the full data is processed in one pass.

Limitations and Caveats
-------------------------

**Segmentation changes results slightly.** When the data is split into segments, each segment is optimised independently. For ``askadam.m``, because gradients are accumulated across all voxels in a single loss, splitting the volume changes the loss landscape: voxels at segment boundaries lose their neighbours during that segment's optimisation, and any spatial regularisation (e.g. TV) is applied within, not across, segment boundaries. For ``mcmc.m``, chains are independent per voxel so the effect is more limited, but initialisation and any shared hyperparameters may differ slightly across segments. In practice, differences are small for typical segmentation counts, but results will not be numerically identical to a single-pass fit. This is noted explicitly in the console output.

**3D TV regularisation does not cross segment boundaries.** If you are using 3D total variation regularisation, the regulariser is computed independently within each segment's slice group. This can introduce mild discontinuities at boundaries, particularly when segment counts are high. 2D TV (applied within each axial slice) is unaffected.

**The linear memory model is an approximation.** The probe-based extrapolation assumes that peak memory scales linearly with voxel count. This is a reasonable first-order model for most GACELLE applications, but it can underestimate memory for models with strongly nonlinear forward functions or when MATLAB's internal allocator fragments GPU memory. A conservative ``safetyFactor < 1`` can be applied to reduce the effective budget if you observe out-of-memory errors despite the manager predicting a fit.

**Memory prediction depends on the GPU state at probe time.** If GPU memory is heavily fragmented or occupied by other processes at the time of probing, the estimate of other-process overhead may be inaccurate, leading to a more conservative (more segments) or, less commonly, an overoptimistic (fewer segments) segmentation decision. Closing other GPU-intensive applications before running is recommended for reproducibility.

**Forward model internal allocations are not captured.** The probe measures peak memory during a zero-iteration fit (``fitting.iteration = 0``). Allocations made inside the forward model during a real optimisation run (e.g. temporary arrays in EPG-X or ANN inference) may add memory beyond what the probe captures. If you find that the predicted-safe segment count still causes OOM errors, reduce the number of segments manually or contact the developers.

**nvidia-smi must be available.** The memory manager relies on ``nvidia-smi`` being on the system path. On most Linux HPC systems this is the case; on Windows it may require adding the CUDA bin directory to the PATH.

Related Utilities
------------------

The memory manager calls the following internal utilities, which are documented in the API reference:

- ``utils.get_other_process_memory(pid)`` — returns the current GPU memory (MB) used by all processes except the MATLAB process with the given PID.
- ``utils.get_available_vram()`` — returns current free GPU VRAM (MB) as reported by nvidia-smi.
- ``utils.read_absolute_peak_from_log(logFile)`` — reads a nvidia-smi CSV log and returns the peak total GPU memory usage (MB), discarding the first 15% of samples as a warm-up period.
- ``utils.build_balanced_boundaries(mask, NvoxPerSeg)`` — partitions a 3D binary mask into slice groups such that each group contains at most ``NvoxPerSeg`` masked voxels, with group sizes as equal as possible.