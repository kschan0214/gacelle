# Contributing to GACELLE

Thanks for considering a contribution. This file covers what's useful to
know when *modifying* the toolbox, as opposed to just using it - see
[README.md](README.md) and http://gacelle.readthedocs.io for user-facing
documentation.

## Development setup

```matlab
addpath('/path/to/gacelle');
addpath_gacelle();
```

`addpath_gacelle()` adds the toolbox and its subfolders to the path while
excluding `docs/`, `sandbox/`, `deprecated/` and `mpl_training/` - use it
instead of a plain `addpath(genpath(...))`, which would pull those in too.

Requirements: a recent MATLAB (Deep Learning Toolbox for the `askadam.m`
solver's autodiff), and an NVIDIA GPU with the Parallel Computing Toolbox
for anything that actually fits a model (`gpuDevice`).

## Running the test suite

```matlab
cd tests
run_tests
```

See [tests/README.md](tests/README.md) for what's covered (a Tier-1 suite
that checks every model class actually resolves on the path with no GPU
needed, and Tier-2 smoke fits per model that need one) and how to add a
test for a new model. Tier-2 tests skip cleanly (not fail) with no GPU
available. CI (`.github/workflows/tests.yml`) runs the suite on every
push/PR; Tier 1 always runs there, Tier 2 shows as filtered since
GitHub-hosted runners have no GPU.

Run the test suite before opening a PR. There's no formal coverage
requirement, but if you're adding a new model class, adding its Tier-2
smoke test alongside it is expected.

## Conventions to follow

- **Every model class's `estimate()` takes arguments in the same order:
  `(data, mask, extraData, fitting[, pars0])`.** Two models
  (`gpuSANDI`, `gpumcmicro`) once reversed `fitting`/`extraData` and had
  that documented as intentional - it wasn't, and it's exactly the kind
  of inconsistency that produces silently wrong results (an argument
  bound into the wrong slot) instead of an error. Don't reintroduce a
  one-off ordering for a new model.
- **One class per model, dispatching on `fitting.solver = 'askadam' |
  'mcmc'`.** Don't add a separate `gpu<Model>mcmc` class - see any
  existing model (e.g. `NEXI/gpuNEXI.m`) for the pattern, and
  `docs/tutorial/writing_a_new_model.rst` for a full worked example built
  around `gpuR2starMapping`.
- Superseded/experimental code goes in that model's `sandbox/` (or
  `sandbox/deprecated/` if it's actively been replaced), never left
  sitting in the main model folder - `addpath_gacelle()` excludes both by
  name, and `tests/ClassAvailabilityTest.m` checks that exclusion holds.
- Hardcoded absolute paths (e.g. `/autofs/...`) don't belong in anything
  meant to run outside this lab's cluster - demo/example scripts should
  default to a portable path (see the DWI demos' `fullfile('~/Downloads',
  ...)` pattern) and downloadable data should go through
  `utils.check_dwi_invivo_demo_data` / `utils.check_gre_invivo_demo_data`
  rather than being assumed already present.

## Reporting bugs / asking questions

Bugs: [Issue page](https://github.com/kschan0214/gacelle/issues).
General usage questions: [Discussion board](https://github.com/kschan0214/gacelle/discussions).

## License

GACELLE is licensed under GPLv3 - see [LICENSE](LICENSE). Contributions
are accepted under the same license.
