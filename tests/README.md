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
  `JointR1R2starMapping`, `SANDI`, `MCRMWI`, `gpumcTFI`, `gpuPDF`). Each
  forward-simulates a *tiny* synthetic dataset (a handful of voxels) with
  known ground truth, fits it with `fitting.solver = 'askadam'` at a
  drastically reduced iteration count, and checks the fit runs without
  error and produces finite (non-NaN/Inf) output. This is **not** a
  scientific-accuracy test - a handful of voxels/iterations can't
  reproduce the real `demo_*NoisePropagation*.m` scripts' validation - it
  only checks "did this still run and produce sane output."
  `SmokeFit_MCRMWITest.m` is the heaviest of the demo-based ones - it
  loads the bundled pretrained EPG-X MLP weights
  (`MCRMWI/EPGXgen_net/*.mat`) to forward-simulate ground truth, same as
  the real demo does.

  `SmokeFit_gpuPDFTest.m` and `SmokeFit_gpumcTFITest.m` are different from
  the rest: neither model has a `NoisePropagation` demo to base a test on,
  so these instead follow the synthetic-phantom design SEPIA's own test
  suite uses for its background-field-removal/QSM tests
  (`sepia/test/phantom/generate_synthetic_phantom.m` - not the code
  directly, since SEPIA writes NIfTI files for its own I/O rather than
  in-memory arrays): a couple of point susceptibility sources
  dipole-kernel-convolved (via `gpuPDF`'s own static `dipole_kernel`
  method) to a field, combined with a mono-exponential decay for
  `gpumcTFI`'s multi-echo case. Both also override the class's 20 mm
  `gapMinMM` default down to 4 mm so the internal zero-padding these two
  models do (to give the dipole convolution's background support room)
  stays small and the test stays fast.

## Running

```matlab
cd tests
run_tests
```

or, from anywhere:

```matlab
addpath('/path/to/gacelle'); addpath_gacelle();
run('/path/to/gacelle/tests/run_tests.m')
```

`run_tests` collects every test under `tests/` including subfolders, but
**excludes `tests/validation/`** (see below). Calling
`runtests(testsDir, 'IncludeSubfolders', true)` directly would also pick
up the validation scripts as script-based tests, so prefer `run_tests`.

On a machine/CI runner with no GPU, every `SmokeFit_*Test` reports as
**Incomplete/filtered**, not failed - that's expected
(`gacelletest.assumeGPU`, called at the start of each Tier-2 test, uses
MATLAB's `assumeGreaterThan` so a missing GPU skips the test rather than
failing it). `ClassAvailabilityTest` always actually runs.

## mcmc_bayes (EXPERIMENTAL) tests

Tests for the experimental `utils/mcmc_bayes.m` subclass of `mcmc`:

- **`McmcBayesUnitTest.m`** (mostly Tier 1, no GPU) - option detection
  (`isLegacy`), the not-implemented guard for later-phase options, the
  parameter transforms (round trip, log-Jacobian vs finite differences,
  stability at large |u|, per-parameter cell parsing) and the R-hat/ESS
  diagnostics on synthetic iid and AR(1) chains. A few dispatch tests
  that actually start the sampler call `gacelletest.assumeGPU`.
- **`McmcBayesLegacyTest.m`** (Tier 2, GPU) - with all new options absent
  or at their defaults, `mcmc_bayes` output must be bitwise identical to
  `mcmc` for the same seeds.
- **`RicianUtilTest.m`** (Tier 1, one GPU test) - the Rician mean in
  `utils/rician.m` (`rician_mean`, `rician_mean_gacelle`, `L12_gacelle`)
  against numerical integration of the Rician density at low, moderate and
  high SNR, the high-SNR limit, and `dlarray` gradients.

### Validation scripts (hand-run, not part of `run_tests`)

`tests/validation/mcmc_bayes/` holds long statistical validation scripts
(one per test ID, e.g. `test_T1_2_transform_equivalence.m`). Each states
its criterion up front, prints PASS/FAIL and the seeds used. They take
minutes each on a GPU and are **excluded from `run_tests`**; run them by
hand from the repository root:

```matlab
addpath(pwd); addpath_gacelle(pwd);
addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
test_T1_2_transform_equivalence
```

### MATLAB version note (MGH linen/virtuoso hosts)

On the MGH hosts used for development, the default `/usr/local/bin/matlab`
(R2026a) cannot see the GPU, so every Tier-2 test is skipped. Use R2024b:

```bash
/usr/pubsw/common/matlab/24.2/bin/matlab -batch "cd tests; run_tests"
```

## Adding a new model's smoke test

Copy the pattern from an existing `SmokeFit_*Test.m` most similar to your
model (e.g. `SmokeFit_NEXITest.m` for a DWI model with `extraData=[]`,
`SmokeFit_GREMWITest.m` for a complex multi-echo GRE model with nontrivial
`extraData`). Base the constructor args / `FWD()` call shape / `fitting`
struct / `estimate()` call directly on that model's own
`demo_*NoisePropagation*.m` script, and shrink `Nsample` and
`fitting.iteration` down from the demo's defaults (typically 1e3 voxels,
1e4 askadam / 2e5 mcmc iterations) to a handful of each.
