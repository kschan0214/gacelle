# mcmc_bayes validation scripts

Long statistical validation scripts for the EXPERIMENTAL `utils/mcmc_bayes.m`
subclass. They are run by hand on a GPU machine, not by `tests/run_tests.m`.

- One script per test ID (`test_T1_2_transform_equivalence.m`, ...).
- Each script states its criterion up front, prints `PASS` or `FAIL`, and
  records the seeds it used (`rng` and `parallel.gpu.rng`).
- GPU only, single precision (the sampler has no CPU path).
- Run from the repository root with MATLAB R2024b on the MGH hosts (the
  default R2026a cannot see the GPU):

  ```matlab
  addpath(pwd); addpath_gacelle(pwd);
  addpath(fullfile(pwd,'tests','validation','mcmc_bayes'));
  test_T1_2_transform_equivalence
  ```

Phase 1 scripts (a few minutes each on an A40):

- `test_T1_2_transform_equivalence.m`: 'sigmoid' vs 'linear' under the
  same flat native prior; per-voxel two-sample KS rejection count must be
  consistent with 5%.
- `test_T1_3_adaptation.m`: post-burn-in acceptance within target +/- 0.05
  in >= 95% of voxels (joint and componentwise), and posterior mean/SD
  agree with a long non-adaptive legacy run within MC error.
- `test_T1_neutral_vs_legacy.m`: the neutral new path (linear, joint, no
  adaptation, forced with the test-only `fitting.forceNewPath = true`)
  agrees statistically with legacy `mcmc`; also reports same-seed bitwise
  identity and wall time (information only).

Phase 2 scripts (marginal likelihoods; minutes each on an A40):

- `test_T2_1_grid_reference.m`: `marginal_S0noise` posterior quantiles of the
  non-linear parameters vs a brute-force double-precision grid reference
  (monoexponential R2star, 1D grid; IVIM D/F/Dstar, refined 3D grid), SNR 20
  and 50, transforms 'linear' and 'sigmoid'; z-scores from the quantile MC
  standard error (ESS of the indicator).
- `test_T2_2_nuisance_recovery.m`: the post-hoc sigma and S0 draws vs a
  reference (monoexponential: 3D joint grid over R2star, S0, log sigma^2 from
  the priors only; IVIM: grid posterior x exact conditionals).
- `test_T2_3_rician_bias.m`: characterisation only (no pass/fail): bias of
  the posterior mean/median under Rician vs Gaussian noise at SNR 5 and 10.

Shared helpers:

- `ivim_fwd.m`: minimal IVIM forward model,
  `S = S0*[F*exp(-b*Dstar) + (1-F)*exp(-b*D)]`, returning `[Nb, Nv]` in the
  mcmc FWD convention.
- `r2star_sim.m`: monoexponential (gpuR2starMapping) simulation and mcmc
  fitting structure.
- `ks2_asymptotic.m`: two-sample KS statistic and asymptotic p-value (no
  Statistics toolbox).
- `binom_bounds.m`: exact central interval of a binomial count.
- `thin_by_ess.m`: thin one chain by 2*tau (tau = N/ESS).
- `mc_compare.m`: MC z-scores of posterior mean/SD differences between two
  runs.
- `ivim_t2_config.m`, `ivim_sim.m`: Phase 2 IVIM truth, b-values, box and
  Gaussian/Rician simulation.
- `run_chains.m`: several independent chains per voxel as voxel copies in one
  mcmc_bayes call.
- `grid3_refine.m`: coarse-then-refined 3D grid posterior (double, GPU) with a
  second, coarser refined grid for the discretisation error.
- `grid_quantiles.m`: quantiles and density of a 1D marginal on grid nodes.
- `quantile_z.m`: z-scores of sampler quantiles vs reference quantiles.

Fast unit tests and the bitwise legacy-identity test live in
`tests/McmcBayesUnitTest.m` and `tests/McmcBayesLegacyTest.m`. This folder
is excluded from `tests/run_tests.m`.
