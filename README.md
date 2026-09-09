# GACELLE (GPU-AcCELerated toolbox for high-throughput multi-dimensionaL quantitative parameter Estimation)

![gacelle logo](docs/getting_started/_images/20250505_gacelle_1e429d%20blue.png)

## Introduction  

**GACELLE** is a MATLAB parameter estimation tool optimised for high-throughput data processing.

GACELLE provides two types of solvers to handle parameter estimation prolem:
1. Determinisic, minimisation-based solver: *askAdam.m* 
2. Stochastic, Markov chain Monte Carlo solver: *mcmc.m*

**Full documentation can be found on http://gacelle.readthedocs.io**

## Supported models

| Model (class) | Estimates | Data |
|---|---|---|
| [AxCaliberSMT](https://gacelle.readthedocs.io/en/latest/supported_models/AxCaliberSMT.html) (`gpuAxCaliberSMT`) | Orientation-invariant axon diameter index | Multi-shell diffusion MRI |
| [SANDI](https://gacelle.readthedocs.io/en/latest/supported_models/SANDI.html) (`gpuSANDI`) | Soma radius/fraction, neurite density, extracellular diffusivity | Multi-shell diffusion MRI |
| [NEXI](https://gacelle.readthedocs.io/en/latest/supported_models/NEXI.html) (`gpuNEXI`) | Neurite volume fraction, compartment diffusivities, inter-compartment exchange rate | Multi-shell, multi-diffusion-time diffusion MRI |
| [mcmicro](https://gacelle.readthedocs.io/en/latest/supported_models/mcmicro.html) (`gpumcmicro`) | Neurite volume fraction and intrinsic diffusivity (± compartmental R2) | Multi-shell (± multi-echo) diffusion MRI |
| [GRE-MWI](https://gacelle.readthedocs.io/en/latest/supported_models/GREMWI.html) (`gpuGREMWI`) | Myelin water fraction and compartmental relaxation/frequency parameters | Multi-echo GRE |
| [MCR-MWI](https://gacelle.readthedocs.io/en/latest/supported_models/MCRMWI.html) (`gpuMCRMWI`) | Myelin water fraction, compartmental R1/R2*/frequency, myelin water exchange rate | Variable-flip-angle, multi-echo GRE |
| [JointR1R2star](https://gacelle.readthedocs.io/en/latest/supported_models/JointR1R2star.html) (`gpuJointR1R2starMapping`) | R1 and R2* jointly | Variable-flip-angle, multi-echo GRE |
| [R2starMapping](https://gacelle.readthedocs.io/en/latest/supported_models/R2starMapping.html) (`gpuR2starMapping`) | M0 and R2* (mono-exponential decay) | Multi-echo GRE (magnitude) |
| [PDF](https://gacelle.readthedocs.io/en/latest/supported_models/PDF.html) (`gpuPDF`) | Background field removal for QSM/phase processing | Total field map |
| [mcTFI](https://gacelle.readthedocs.io/en/latest/supported_models/mcTFI.html) (`gpumcTFI`) | Susceptibility, R2*, M0 and phase offset jointly (no separate background-removal/dipole-inversion steps) | Raw complex multi-echo GRE |

## What's new in v1.1 (since v1.0)

- **Unified askadam/mcmc solver interface.** Every model class now exposes
  a single `fitting.solver = 'askadam' | 'mcmc'` option and one
  `estimate(...)` entry point, replacing the old pattern of separate
  `gpu<Model>` / `gpu<Model>mcmc` classes (e.g. `gpuJointR1R2starMapping` +
  `gpuJointR1R2starMappingmcmc` are now just `gpuJointR1R2starMapping`).
- **New QSM module (`qsm/`):**
  - `gpumcTFI` - multi-echo complex Total Field Inversion: jointly
    reconstructs susceptibility, R2*, M0 and a phase offset directly from
    raw complex multi-echo GRE in a single nonlinear fit (MEDI-style
    anatomically-weighted TV regularisation, optional CSF
    zero-referencing).
  - `gpuPDF` - GPU Projection onto Dipole Fields background field removal.
- **Automatic GPU memory management** (`fitting.autoMemManage = 1`): a
  probe-based memory predictor (`utils.find_optimal_segment_3D`)
  transparently segments large volumes into density-balanced, halo-padded
  chunks processed sequentially when the full dataset won't fit in
  available VRAM, for both `askadam.m` and `mcmc.m`, with manual override
  via `fitting.segmentOverlap` / `fitting.NSegmentUser`.
- **Reworked `askadam.m` convergence handling:** five independent stopping
  criteria (loss threshold, robust loss convergence via
  `fitting.robustConvergence`, step-norm, gradient-norm, max iterations),
  selectable linear/EMA convergence models, and per-criterion patience
  windows.
- **New default parameter-space transform for `askadam.m`:**
  `fitting.parameterTransform = 'sigmoid'` (unconstrained logit/sigmoid
  reparameterisation) is now the default, fixing a boundary-sticking
  numerical failure mode present in the old hard-clamped linear rescaling
  (`'linear'`, the v1.0 behaviour, is still available).
- **MCMC ensemble improvements**, including a fixed-anchor-walkers option.
- **`addpath_gacelle.m`** - adds the whole toolbox to the MATLAB path in
  one call, automatically excluding `docs/`, `sandbox/`, `deprecated/`
  and `mpl_training/`.
- Expanded documentation: new "Advanced" pages (convergence, parameter
  transform, automatic memory management), a "Writing a new model"
  tutorial, an "Understanding the output" page, and reference pages for
  the new QSM classes plus R2starMapping/mcmicro/SANDI.

## Terms of use
Please check [the license file](https://github.com/kschan0214/gacelle/blob/master/LICENSE) for more information. 

If you use GACELLE in your research, please cite the following article:

[Chan, K.-S., Lee, H., Ma, Y., Bilgic, B., Huang, S.Y., Lee, H.-H., Marques, J.P., 2025. GACELLE: GPU-accelerated tools for model parameter estimation and image reconstruction. arXiv:2511.22094.](https://arxiv.org/abs/2511.22094)

If you use GACELLE's implementation of NEXI, please also cite:

[Chan, K.-S., Ma, Y., Lee, H., Marques, J.P., Olesen, J.L., Coelho, S., Novikov, D.S., Jespersen, S.N., Huang, S.Y., Lee, H.-H., 2025. In vivo human neurite exchange time imaging at 500 mT/m diffusion gradients. Imaging Neurosci. 3, imag_a_00544.](https://doi.org/10.1162/imag_a_00544)

If you use GACELLE's implementation of MCR-MWI, please also cite:

[Chan, K.-S., Kim T.H., Bilgic B., Marques J.P. Semi-supervised learning for fast multi-compartment relaxometry myelin water imaging (MCR-MWI). In:
Proceedings 30. Annual Meeting International Society for Magnetic Resonance in Medicine. Vol 30. London, United Kingdom; 2022:1639.](https://doi:10.58530/2022/1639),

[Chan, K.-S., Marques, J.P., 2020. Multi-compartment relaxometry and diffusion informed myelin water imaging – Promises and challenges of new gradient echo myelin water imaging methods. Neuroimage 221, 117159.](https://doi.org/10.1016/j.neuroimage.2020.117159), and

[Chan, K.-S., Chamberland, M., Marques, J.P., 2023. On the performance of multi-compartment relaxometry for myelin water imaging (MCR-MWI) – test-retest repeatability and inter-protocol reproducibility. Neuroimage 266, 119824.](https://doi.org/10.1016/j.neuroimage.2022.119824)

Please report any bugs on the [Issue page](https://github.com/kschan0214/gacelle/issues). 

If you have a more general question regarding the usage of GACELLE, please make use of the [Discussion board](https://github.com/kschan0214/gacelle/discussions).

