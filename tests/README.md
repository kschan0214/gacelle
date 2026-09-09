# GACELLE regression tests

A lightweight `matlab.unittest` suite - no extra dependencies, ships with
every MATLAB install. Two tiers:

- **`ClassAvailabilityTest.m`** (Tier 1, no GPU needed) - checks that
  every supported model class, `askadam`/`mcmc`, and `addpath_gacelle`
  itself actually resolve on the path after calling `addpath_gacelle()`,
  and that `addpath_gacelle()` correctly excludes `docs/`, `sandbox/`,
  `deprecated/` and `mpl_training/`. This exists because two files
  required by already-committed code were once left untracked in the
  working tree, silently breaking a fresh checkout - this test is
  specifically designed to catch that class of bug.
- **`SmokeFit_*Test.m`** (Tier 2, GPU required) - one per model
  (`R2starMapping`, `mcmicro`, `NEXI`, `AxCaliberSMT`, `GREMWI`,
  `JointR1R2starMapping`, `SANDI`). Each forward-simulates a *tiny*
  synthetic dataset (a handful of voxels) with known ground truth, fits
  it with `fitting.solver = 'askadam'` at a drastically reduced iteration
  count, and checks the fit runs without error and produces finite
  (non-NaN/Inf) output. This is **not** a scientific-accuracy test - a
  handful of voxels/iterations can't reproduce the real
  `demo_*NoisePropagation*.m` scripts' validation - it only checks "did
  this still run and produce sane output."

Not covered yet: `gpuMCRMWI` (heaviest setup - EPG-X ANN + DIMWI +
complex data; a real reduction of it is a good follow-up once this
pattern is established), `gpumcTFI` and `gpuPDF` (no `NoisePropagation`
demo to base a test on; both need a synthetic brain-like complex GRE
volume with a dipole-convolved ground truth field rather than a flat
voxel array).

## Running

```matlab
cd tests
run_tests
```

or, from anywhere:

```matlab
addpath('/path/to/gacelle'); addpath_gacelle();
runtests('/path/to/gacelle/tests', 'IncludeSubfolders', true)
```

On a machine/CI runner with no GPU, every `SmokeFit_*Test` reports as
**Incomplete/filtered**, not failed - that's expected
(`gacelletest.assumeGPU`, called at the start of each Tier-2 test, uses
MATLAB's `assumeGreaterThan` so a missing GPU skips the test rather than
failing it). `ClassAvailabilityTest` always actually runs.

## Adding a new model's smoke test

Copy the pattern from an existing `SmokeFit_*Test.m` most similar to your
model (e.g. `SmokeFit_NEXITest.m` for a DWI model with `extraData=[]`,
`SmokeFit_GREMWITest.m` for a complex multi-echo GRE model with nontrivial
`extraData`). Base the constructor args / `FWD()` call shape / `fitting`
struct / `estimate()` call directly on that model's own
`demo_*NoisePropagation*.m` script, and shrink `Nsample` and
`fitting.iteration` down from the demo's defaults (typically 1e3 voxels,
1e4 askadam / 2e5 mcmc iterations) to a handful of each.
