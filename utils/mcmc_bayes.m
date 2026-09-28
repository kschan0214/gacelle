classdef mcmc_bayes < mcmc
% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
%
% *** EXPERIMENTAL ***
% This is a subclass of mcmc for Bayesian extensions of the MCMC sampler
% (parameter transforms, marginal likelihoods, hierarchical and MRF priors).
% Interface and behaviour may change without notice.
%
% With all new options at their legacy defaults (or absent), optimisation
% passes through to mcmc.optimisation unchanged, so the output is bitwise
% identical to mcmc.
%
% The sampler options parameterTransform, updateScheme, adaptStepSize, adaptInterval,
% adaptTarget, adaptCovariance and overdisp are also opt-in options of mcmc itself
% (mcmc.metropolis_hastings_adaptive, the same chain for the same seed). mcmc_bayes
% keeps running its own loop (metropolis_hastings_bayes) for them.
%
% New fitting options (legacy defaults)
%   .parameterTransform : 'linear'      'linear'|'sigmoid'|'log', or a cell with one entry per modelParams
%   .likelihood         : 'gaussian'    'gaussian'|'marginal_noise'|'marginal_S0noise'|'marginal_S0noise_flat'
%   .S0Param            : ''            name of the amplitude parameter in modelParams, required by
%                                       (and only allowed with) 'marginal_S0noise(_flat)'
%   .updateScheme       : 'joint'       'joint'|'componentwise'
%   .adaptStepSize      : false         adapt proposal scale during burn-in (frozen afterwards)
%   .adaptInterval      : 50            # iterations between two adaptation steps
%   .adaptTarget        : []            target acceptance rate, [] -> 0.234 (joint) | 0.44 (componentwise)
%   .adaptCovariance    : false         adaptive-covariance (Haario-style adaptive Metropolis) joint
%                                       proposal, learnt per voxel during burn-in and frozen afterwards.
%                                       Requires updateScheme = 'joint' and adaptStepSize = true (else error
%                                       mcmc:adaptCovariance). See "Adaptive covariance" below
%   .overdisp           : 0             relative over-dispersion of the start point for repetition > 1
%   .prior              : []            structure with the fields
%       .hierarchical   : []            hierarchical Normal prior N(u | mu, Sigma) on the transformed
%                                       parameters (Phase 3), a structure (struct() or true for all
%                                       defaults) with the fields
%           .hyperprior     : 'niw'         'niw' | 'jeffreys_half'
%           .m0             : []            NIW mean, [] -> mean of the starting u over voxels
%           .kappa0         : 1e-3          NIW mean precision factor
%           .nu0            : []            NIW degrees of freedom, [] -> d + 2
%           .Psi0           : []            NIW scale matrix, [] -> max(nu0-d-1,1) * diag(v0), see below
%           .fixed          : false         true: skip the Gibbs block and use .mu/.Sigma
%           .mu, .Sigma     : []            fixed hyperparameters (required with fixed = true)
%           .params         : []            cellstr, parameters under the hierarchy (they form u),
%                                           [] -> all sampled parameters except 'noise'
%           .subsetFraction : 1             fraction of voxels for stage 1 (estimate_hyper_subset only)
%           .maxGPUMemory   : []            bytes available for the coupled free-hyperparameter run,
%                                           [] -> gpuDevice().AvailableMemory
%           .K              : 1             # Gaussian-mixture groups (Phase 7), positive integer; K = 1 is the
%                                           single-Normal prior above (bitwise identical to before Phase 7)
%           .alpha          : 1             symmetric Dirichlet concentration of the group weights pi (K > 1)
%           .init           : 'kmeans'      initial z, mu_k, Sigma_k, pi of the free mode (K > 1), see Phase 7
%           .pi             : []            fixed group weights [K,1] (fixed = true and K > 1; then .mu is
%                                           [d,K] and .Sigma is [d,d,K])
%       .mrf            : []            MRF prior on the hierarchical parameters (Phase 4), a structure
%                                       (struct() or true for all defaults) with the fields below.
%                                       Requires .hierarchical with fixed = true (use run_two_stage
%                                       for the free-hyperparameter case)
%           .potential      : 'l1'          'l1' | 'huber' | 'quadratic'
%           .tau            : 1             temperature, > 0 (coupling strength 1/tau)
%           .W              : []            [d,1] per-parameter weights W_p > 0, [] -> 1./sqrt(diag(Sigma))
%                                           of the fixed Sigma (fixed for the whole run); K > 1: Sigma is
%                                           replaced by the mixture's marginal covariance (Phase 7)
%           .huberDelta     : 1             Huber threshold in units of sqrt(Sigma_pp): delta_p =
%                                           huberDelta * sqrt(Sigma_pp) (scalar or [d,1]; 'huber' only;
%                                           K > 1: the same mixture scale as W)
%           .edgeWeights    : []            [K,Nv] fixed symmetric edge weights w_ij >= 0 in the layout of
%                                           build_neighbours (entries of absent neighbours ignored), [] -> 1
%           .mode           : '3d'          '3d' | '2d' (in-plane only: dims 1-2 of the same slice)
%           .radius         : 1             neighbourhood radius r (positive integer)
%           .connectivity   : []            'face' (r = 1 only) | 'full' (cube/square of radius r),
%                                           [] -> 'face' for '3d' with r = 1, else 'full'
%           .maxGPUMemory   : []            bytes for the memory guard, [] -> prior.hierarchical.maxGPUMemory,
%                                           else gpuDevice().AvailableMemory
%           .subsetForward  : true          evaluate FWDfunc and the likelihood on the voxels of the active
%                                           colour only (Phase 4b, performance only, same target). Needs a
%                                           column-separable FWDfunc (column v of the output depends on
%                                           column v of the parameters only, for any number of columns);
%                                           varargin entries with a voxel dimension are NOT subset. Checked
%                                           automatically at setup, with a fallback to the full evaluation
%                                           (warning mcmc_bayes:subsetForwardFallback), see Phase 4b below
%
% Test-only options (not part of the user interface)
%   .forceNewPath       : false         run the new sampling loop even if all new options are at their
%                                       legacy defaults. Used to compare the neutral new path against
%                                       the legacy mcmc in tests/validation/mcmc_bayes.
%   .fixedParams        : []            structure of scalar values, e.g. struct('noise',0.02): these
%                                       parameters are removed from the sampled set and their value is
%                                       injected before each FWDfunc/likelihood call (as S0Param = 1 is);
%                                       they are not in the output. Used for 'known noise' in T3.1.
%   .checkCache         : false         after every sweep, recompute the log-likelihood, log-prior and
%                                       log-Jacobian of the current state from scratch and record the
%                                       largest difference to the cached values (diagnostics.cacheCheck);
%                                       with the MRF also the largest change of any voxel outside the
%                                       active colour during a colour step (cacheCheck.inactiveMoved, must be 0)
%   .mrfUpdate          : 'chromatic'   *** TEST ONLY *** 'simultaneous' proposes and accepts ALL voxels
%                                       at once, each with its neighbours taken from the pre-sweep state.
%                                       This targets the WRONG distribution on purpose: it is the negative
%                                       control of test_T4_3 (it must fail test_T4_2). Never use it for
%                                       inference.
%   .prior.mrf.bayesivimWeights : false *** TEST ONLY *** replicate the spatial term of the BayesIVIM
%                                       code (Spinner et al. 2021, ivim_bayes.m) to show that mcmc_bayes
%                                       reproduces it under identical conditions: for voxel i and parameter
%                                       p the local term is (1/tau) [sum_{j in N(i)} |u_i^p - u_j^p| +
%                                       |u_i^p - u_i^p,curr|] / |u_i^p,curr| (the centre voxel of their
%                                       block is included, and the weight comes from the CURRENT state).
%                                       Requires potential 'l1'; W is not used. The weight depends on the
%                                       units of u, so run in BayesIVIM's units. Combine with mrfUpdate =
%                                       'simultaneous', updateScheme = 'componentwise', mode '2d', radius 2,
%                                       connectivity 'full' for their exact scheme. The target distribution
%                                       is undefined; never use it for inference.
%
% Phase 1 (sampler infrastructure) design notes
% ---------------------------------------------
% (Phase 8: the transforms, adaptation, adaptive covariance, over-dispersion and R-hat/ESS helpers
%  are static methods of mcmc, which also offers them as opt-in options of its own MH sampler.)
% Target. Still the legacy target: Gaussian likelihood (logP_Gaussian with
%   the sampled 'noise') times a uniform box prior on [lb,ub] in native
%   space. Sampling happens in transformed space u = T(x), so the log-target
%   in u is loglik(x(u)) + sum_p log|dx_p/du_p|.
%
% Transforms (per parameter)
%   'linear'  : x = u.                          Box enforced by rejection (as legacy).
%   'sigmoid' : x = lb + (ub-lb)*sigmoid(u).    Box never violated, no rejection.
%   'log'     : x = exp(u), requires lb >= 0.   ub (and lb if lb > 0) enforced by rejection.
%   Native start points are clamped to [lb+eps, ub-eps], eps = 1e-4*(ub-lb),
%   before the forward transform of 'sigmoid'/'log' (as askadam.rescale_parameters).
%   log-sigmoid is computed via softplus for numerical stability.
%
% Proposal scale. Per voxel per parameter, in u space. It is initialised
%   from fitting.xStepSize (a native-space step) mapped through the local
%   derivative at the start point: sigma_u = xStepSize / |dx/du|(u0). For
%   'linear' this is exactly xStepSize. For non-linear transforms sigma_u
%   is capped at the u-width of the eps-clamped box.
%
% Update schemes
%   'joint'         : all parameters of a voxel proposed together (1 forward evaluation per iteration)
%   'componentwise' : one parameter at a time, sequentially within the sweep,
%                     the cached loglik is updated after each step (Nvar forward evaluations per iteration)
%   All voxels are independent and accepted in parallel.
%
% Adaptation (burn-in only, frozen afterwards). Every adaptInterval iterations
%   (as long as the window ends within the burn-in), Robbins-Monro on the log scale:
%       log sigma <- log sigma + gamma_j * (acc_j - adaptTarget),  gamma_j = 2 * j^(-0.6)
%   where acc_j is the acceptance rate of the j-th window. For 'joint', one
%   update per voxel scales all parameters together (ratios kept); for
%   'componentwise', one update per voxel per parameter.
%
% Over-dispersed starts. For repetition ii > 1 (and overdisp > 0):
%       u0_ii = u0 + overdisp * W .* randn,   W_p = T_p(ub_p) - T_p(lb_p)
%   i.e. overdisp is a fraction of the u-width of the (eps-clamped) box; the
%   result is clamped back into that box. For 'linear', W = ub-lb, the same
%   scale that mcmc uses for 'startRange'. overdisp = 0 gives the legacy
%   behaviour (all repetitions start from the same point).
%
% Diagnostics (computed on native-space samples [Nv, Ns, Nchains])
%   rhat : split-R-hat (Gelman et al., BDA3, 2013), not rank-normalised
%   ess  : multi-chain ESS on split chains with Geyer's initial positive and
%          monotone sequence estimator, as in Stan / Vehtari et al. (2021)
%          but without rank normalisation
%
% Phase 2 (marginal likelihoods) derivation
% -----------------------------------------
% Per voxel: y [m x 1] measurements, W = diag(w) the fitting weights, theta
%   the sampled (non-linear) parameters, g = g(theta) the forward model.
%   Noise model as in legacy logP_Gaussian: y_i ~ N(S0*g_i, sigma^2/w_i), i.e.
%       p(y|theta,S0,sigma^2) = (2 pi sigma^2)^(-m/2) exp(-Q/(2 sigma^2)),  Q = (y-S0 g)'W(y-S0 g)
%   (up to the theta-independent factor prod(w_i)^(1/2), which legacy drops too).
%   A measurement with w_i = 0 has infinite variance: it carries no information
%   and contributes no factor sigma^(-1). So m is the number of measurements with
%   NON-ZERO weight, per voxel (m_v = #{i : w_iv ~= 0}), in all marginal
%   likelihoods, the InvGamma shapes and the post-hoc draws below. (The legacy
%   'gaussian' path, logP_Gaussian, keeps m = # rows of y.) Gamma integral used throughout:
%       int_0^inf (s2)^(-a-1) exp(-b/s2) ds2 = Gamma(a) b^(-a)            (a,b > 0)
%
% 'marginal_noise' (no amplitude; g is the full model), p(sigma^2) ∝ 1/sigma^2:
%       R          = r'Wr,  r = y - g
%       L(theta)   = int (2 pi s2)^(-m/2) exp(-R/(2 s2)) s2^(-1) ds2
%                  = pi^(-m/2) Gamma(m/2) R^(-m/2)
%       log L      = -(m/2) log R + [log Gamma(m/2) - (m/2) log pi]         (exponent m/2 confirmed)
%       sigma^2 | theta,y ~ InvGamma(shape m/2, scale R/2)
%
% 'marginal_S0noise' (amplitude S0 linear, S0Param is removed from the sampled set):
%   With c = g'Wg, Shat = y'Wg / c and the weighted residual sum of squares
%       RSS = (y - Shat g)'W(y - Shat g) = y'Wy - (y'Wg)^2/(g'Wg)          (>= 0),
%   Q(S0) = RSS + c (S0 - Shat)^2.
%   Zellner g-prior S0 | sigma^2,theta ~ N(0, k sigma^2/c) plus p(sigma^2) ∝ 1/sigma^2
%   (Orton et al. 2014; Spinner et al. 2021 Eq. 5). Integrating S0 gives the
%   factor (1+k)^(-1/2) with NO sigma dependence (the g-prior's sigma/sqrt(c) scale
%   cancels the Gaussian integral's), and RSS_k = y'Wy - k/(1+k) (y'Wg)^2/c:
%       L_k(theta) = (1+k)^(-1/2) pi^(-m/2) Gamma(m/2) RSS_k^(-m/2)
%   Broad limit k -> inf (the (1+k)^(-1/2) factor is theta-independent):
%       log L      = -(m/2) log RSS + [log Gamma(m/2) - (m/2) log pi]       (DEFAULT)
%   This is the reference's (y'y - (y'g)^2/(g'g))^(-m/2) with W = I: the -m/2
%   exponent is exactly the Zellner broad limit. Conditionals (broad limit):
%       sigma^2 | theta,y      ~ InvGamma(shape m/2, scale RSS/2)
%       S0 | sigma^2,theta,y   ~ N(Shat, sigma^2/c)        (on the real line, lb/ub of S0 not applied)
%   (finite k: InvGamma(m/2, RSS_k/2) and N(k/(1+k) Shat, k/(1+k) sigma^2/c)).
%
% 'marginal_S0noise_flat' (alternative, implemented): flat p(S0) on the real line
%   plus p(sigma^2) ∝ 1/sigma^2. Integrating S0 gives (2 pi s2/c)^(1/2), so
%       L(theta)   = (2 pi)^(-(m-1)/2) Gamma((m-1)/2) 2^((m-1)/2) c^(-1/2) RSS^(-(m-1)/2)
%       log L      = -(1/2) log(g'Wg) - ((m-1)/2) log RSS + const           (needs m >= 2)
%       sigma^2 | theta,y      ~ InvGamma(shape (m-1)/2, scale RSS/2)
%       S0 | sigma^2,theta,y   ~ N(Shat, sigma^2/c)
%   It differs from the Zellner broad limit by the factor c^(-1/2) RSS^(1/2):
%   the broad Zellner prior is p(S0|sigma^2,theta) ∝ sqrt(g'Wg)/sigma, i.e. it is
%   not flat in S0 and depends on theta through g'Wg.
%
% Implementation
%   * The model parameters are given in full (modelParams/lb/ub/xStepSize/
%     per-parameter parameterTransform, as the model wrapper defines them).
%     Marginal likelihoods drop 'noise', and 'marginal_S0noise(_flat)' also drops
%     S0Param, from the sampled set internally; the output keeps the full
%     modelParams order ('noise' is appended if the model has none).
%   * Before each FWDfunc call, S0Param is set to 1 ([1,Nv], single gpuArray), so
%     FWDfunc returns g without the amplitude, and 'noise' is set to a dummy 1
%     (no FWD in the repository reads pars.noise; it is injected for safety).
%   * One forward evaluation per iteration (joint) as before. The loop evaluates
%     only the theta-dependent part of log L (constants in [] above dropped).
%     RSS is computed from the residuals y - Shat*g (no y'Wy - (y'Wg)^2/c
%     cancellation in single precision).
%   * Degenerate states are rejected (log L = -Inf), never NaN-accepted:
%       R or RSS <= rssFloor = max(10*eps('single')^2 * y'Wy, realmin('single')) (per voxel), or NaN;
%       g'Wg <= realmin('single')/eps('single') (~1e-31), or not finite.
%   * Nuisance recovery: the sampler caches [R] or [RSS; Shat; g'Wg] of the
%     current state (updated on acceptance, no extra FWD evaluation) and
%     stores them at every retained (thinned) iteration. After sampling, each
%     retained sample gets an exact conditional draw:
%       sigma^2 = (RSS/2) / G, G ~ Gamma(a,1) (randg on the GPU), a = m/2 or (m-1)/2
%       S0      = Shat + sqrt(sigma^2/c) * z,  z ~ N(0,1)
%     stored as posterior fields 'noise' (= sigma, not sigma^2) and S0Param, so
%     out.posterior/mean/... look like the legacy output. out.settings.nuisance
%     records that they are post-hoc conditional draws.
%
% Phase 3 (hierarchical Normal prior, K = 1)
% ------------------------------------------
% Target. With u_i the [d,1] vector of the hierarchical parameters of voxel i
%   (prior.hierarchical.params, in transformed space), n the number of voxels:
%       p(u_1..n, mu, Sigma | y) ∝ prod_i L(y_i | x(u_i)) N(u_i | mu, Sigma) * p(mu, Sigma)
%   The Normal prior is a density in u: the hierarchical parameters get NO
%   log-Jacobian and NO bound rejection (a truncated Normal would make the
%   Gibbs conditionals below inexact). Allowed transforms for them:
%       'sigmoid' (finite lb < ub), 'log' with lb = 0 and ub = Inf,
%       'linear' with lb = -Inf and ub = Inf   (else error mcmc_bayes:hierarchicalTransform).
%   Sampled parameters outside the hierarchy (e.g. 'noise' under 'gaussian')
%   keep the Phase 1 treatment (flat native box prior, Jacobian, rejection).
%
% Per-voxel MH. The cached per-voxel log-prior lp_i = -(u_i-mu)' Sigma^-1 (u_i-mu)/2
%   (kept separately from the log-likelihood cache) enters the acceptance ratio
%   of joint and componentwise updates; it is recomputed for all voxels after
%   every hyperparameter update. Adaptation is unchanged (burn-in only).
%
% Gibbs block (after every MH sweep, free mode). ubar = sum_i u_i / n and
%   S = sum_i (u_i - ubar)(u_i - ubar)' are accumulated on the GPU in double;
%   the d x d draws are done on the host (double, rng stream).
%   'niw' : NIW(m0, kappa0, Psi0, nu0) hyperprior,
%       p(mu,Sigma) = N(mu | m0, Sigma/kappa0) IW(Sigma | Psi0, nu0). Conjugate update:
%       kappa_n = kappa0 + n,  nu_n = nu0 + n,  m_n = (kappa0 m0 + n ubar)/kappa_n,
%       Psi_n   = Psi0 + S + (kappa0 n/kappa_n)(ubar - m0)(ubar - m0)',
%       Sigma | u ~ IW(Psi_n, nu_n),  mu | Sigma,u ~ N(m_n, Sigma/kappa_n)   (exact joint draw).
%   'jeffreys_half' : p(mu) flat, p(Sigma) ∝ |Sigma|^(-1/2) (Spinner et al. 2021),
%       two exact full conditionals, each drawn at the current value of the other:
%       Sigma | u,mu ~ IW(S_mu, n - d),  S_mu = sum_i (u_i-mu)(u_i-mu)' = S + n (ubar-mu)(ubar-mu)'
%         (from |Sigma|^(-(n+1)/2) exp(-tr(Sigma^-1 S_mu)/2); needs n > 2d - 1),
%       mu | Sigma,u   ~ N(ubar, Sigma/n).
%       (The reference draws mu from N(ubar, cov(u)/n), which is not a conditional.)
%   IW(Psi, nu) (density ∝ |Sigma|^(-(nu+d+1)/2) exp(-tr(Psi Sigma^-1)/2), mean
%   Psi/(nu-d-1)) is drawn with the Bartlett decomposition (no Statistics toolbox):
%       Psi = U'U (chol), A lower triangular, A_jj = sqrt(2 G_j), G_j ~ Gamma((nu-j+1)/2) (randg),
%       A_jk ~ N(0,1) (j > k); then Sigma^-1 = U^-1 A A' U^-T ~ W(Psi^-1, nu), i.e.
%       Sigma = T'T with T = A \ U.
%   Defaults (NIW, proper for any n, needed for SBC): kappa0 = 1e-3; m0 = [] -> mean of
%   the starting u; nu0 = [] -> d + 2 (smallest integer with a finite prior mean of Sigma);
%   Psi0 = [] -> max(nu0-d-1,1) * diag(v0), so that E[Sigma] = diag(v0) for nu0 = d+2, with
%   v0_p = max(var_i(u_start,p), (10 s_p)^2) and s_p the median initial u-space proposal
%   scale of parameter p (xStepSize/|dx/du| at the start; a unit-aware floor for when all
%   voxels start at the same point). These data-dependent defaults are a weak, empirical
%   choice; pass m0/Psi0/nu0 explicitly for a fully specified hyperprior (e.g. SBC).
%   Initial hyperparameters of every repetition: mu = mean_i u_i, Sigma = diag(max(var_i u_i, (10 s_p)^2)).
%
% Fixed mode (fixed = true, user mu/Sigma): no Gibbs block; voxels are
%   conditionally independent. estimate_hyper_subset runs the free mode on a
%   random voxel subset (subsetFraction) and returns posterior-mean mu, Sigma
%   and a fitting structure for fixed mode (stage 1 of the two-stage scheme).
%
% Memory. The free mode couples all voxels, so it cannot be split into
%   segments. Before sampling, the GPU memory is estimated as
%   2 x 4 bytes x Nv x (8 Nm + 24 Nvar + 16) (heuristic, factor 2 for forward-model
%   temporaries); if it exceeds the available memory, mcmc_bayes errors
%   (mcmc_bayes:hierarchicalMemory) and suggests fixed + subset mode.
%
% Output (hierarchical): out.hyper with .params, .transform (u in terms of x),
%   .posterior.mu [d,Ns,Nrep] and .posterior.Sigma [d,d,Ns,Nrep] (free mode, thinned
%   like the voxel samples), .mean/.median of mu and Sigma (u space), .ess and
%   (repetition > 1) .rhat of every mu and Sigma entry; out.settings.prior holds
%   the resolved hyperprior.
%
% Phase 4 (MRF prior with chromatic updates)
% ------------------------------------------
% Target (fixed mu, Sigma). With E the neighbour edges, each counted once:
%       p(u_1..n | y, mu, Sigma) ∝ prod_i L(y_i | x(u_i)) N(u_i | mu, Sigma) * exp(-Phi_MRF(u))
%       Phi_MRF(u) = (1/tau) sum_p W_p sum_{(i,j) in E} w_ij rho(u_i^p - u_j^p)
%   over the hierarchical parameters p only (u space). rho:
%       'l1'        : |x|
%       'huber'     : x^2/(2 delta_p) for |x| <= delta_p, |x| - delta_p/2 otherwise. This is the Huber
%                     function divided by delta_p (the "smoothed L1"), so that it has slope 1 for large |x|
%                     and tau means the same as for 'l1' at large differences. delta_p = huberDelta *
%                     sqrt(Sigma_pp), default huberDelta = 1 (provisional; Phase 5 decides).
%       'quadratic' : x^2/2 (Gaussian MRF, used for the exact tests)
%   W_p (default 1/sqrt(Sigma_pp)) and w_ij (default 1) are fixed before sampling and never depend
%   on the chain state (unlike the reference's 1/|u_i|). The MRF alone is improper (shift
%   invariant), hence it requires the hierarchical prior.
%   Gaussian MRF scaling: for one parameter, sum_{(i,j) in E} w_ij (u_i - u_j)^2 = u' L_w u with
%   L_w = D - A_w the weighted graph Laplacian (A_w(i,j) = w_ij, D = diag(sum_j w_ij)), so with
%   rho = x^2/2 each edge counted once, exp(-Phi) = exp(-u' [(1/tau) L_w (x) diag(W)] u / 2): the
%   MRF adds exactly (1/tau)(L_w (x) diag(W)) to the precision (voxel-major ordering of u).
%
% Local term. Only the edges at voxel i involve u_i, and each such edge appears once in
%       Phi_i(u_i; u_N(i)) = (1/tau) sum_p W_p sum_{j in N(i)} w_ij rho(u_i^p - u_j^p),
%   so changing u_i alone changes Phi_MRF by exactly the change of Phi_i (needs a symmetric
%   neighbour table and symmetric w_ij; both are asserted).
%
% Geometry (build_neighbours). nbr [K,Nv] int32: position of each neighbour in the masked-voxel
%   index (order of find(mask)), 0 if outside the mask or the volume. Offsets are sorted
%   lexicographically, so offset K+1-k is the negative of offset k.
%       '3d', 'face', r = 1 : 6 face neighbours
%       '3d', 'full', r     : cube, (2r+1)^3 - 1 neighbours
%       '2d', 'face', r = 1 : 4 in-plane neighbours (dims 1 and 2, same slice)
%       '2d', 'full', r     : square, (2r+1)^2 - 1 (r = 2 is the reference's 5 x 5)
% Colouring (build_colours), (i,j,k) the 1-based voxel subscripts:
%       'face'      : parity of i+j+k ('3d') or i+j ('2d'), 2 colours (a face step changes one
%                     coordinate by 1)
%       'full', r   : (mod(i,r+1), mod(j,r+1)[, mod(k,r+1)]), (r+1)^3 ('3d') or (r+1)^2 ('2d') colours
%                     (two voxels within Chebyshev distance r differ by 1..r in some coordinate,
%                     which is non-zero mod r+1)
%   Properness (no neighbour shares a colour) and the symmetry of nbr are asserted at construction.
%
% Chromatic sweep. One sweep loops over the (non-empty) colours in a fixed order. For colour c,
%   every voxel of c is proposed (joint or componentwise) and accepted with
%       log r = dlogL + dlog N(u_i | mu, Sigma) [+ dlog|J| of non-hierarchical rows] - dPhi_i,
%   conditioning on the CURRENT values of all other voxels. The voxels of c are conditionally
%   independent given the others (no two are neighbours), so the colour step is an exact MH
%   update of their joint conditional; voxels outside c never change in that step. The log-
%   likelihood (and marginal statistics) and hierarchical log-prior caches are updated for the
%   accepted voxels after each colour step; Phi_i is always recomputed from the current
%   neighbours and never cached. Each voxel is updated once per sweep, so the per-voxel
%   acceptance counts and the burn-in adaptation are unchanged.
%   Cost with prior.mrf.subsetForward = false (the Phase 4 first version, and the fallback of
%   Phase 4b): the forward model is evaluated on ALL voxels and the result is used for the active
%   colour only, i.e. C forward evaluations per sweep (C x Nvar componentwise).
% Negative control (fitting.mrfUpdate = 'simultaneous', TEST ONLY): one class with all voxels,
%   each voxel's Phi_i uses the pre-sweep neighbours while the neighbours move at the same time.
%   This is NOT a valid MH kernel for the joint target.
%
% Two-stage empirical Bayes (run_two_stage). With the MRF on, the normalising constant of the
%   joint prior on u depends on Sigma, so the IW draw is no longer the exact conditional and the
%   free-hyperparameter mode is not allowed with the MRF. run_two_stage: stage 1 samples
%   (u, mu, Sigma) under the hierarchical prior without the MRF (optionally on a voxel subset,
%   estimate_hyper_subset); stage 2 fixes mu, Sigma at their stage-1 posterior means and samples u
%   under hierarchical + MRF with the same likelihood, starting from the stage-1 posterior means
%   of u. This is NOT the full joint posterior (hyperparameter uncertainty is not propagated).
%
% Memory. The MRF couples all voxels, so it needs a single call (no segmentation). The guard of
%   the free hierarchical mode is extended by 4 x K x Nv x 6 bytes (neighbour table, edge weights,
%   neighbour-value temporaries), plus 8 x Nm x Nv bytes with subsetForward (per-colour copies of
%   y and weights); mcmc_bayes errors (mcmc_bayes:mrfMemory) if it does not fit.
%
% Output (MRF): out.settings.mrf holds the resolved potential, tau, W, delta, mode, radius,
%   connectivity, # neighbours, # colours, # edges and the subsetForward outcome.
%
% Phase 4b (forward model on the active colour only, prior.mrf.subsetForward = true)
% ----------------------------------------------------------------------------------
% Performance only; the target and the MH kernel are unchanged. In each colour step (chromatic
%   update only) the parameter structure is built for the Na active voxels only (with S0Param,
%   'noise' and test-only fixedParams injected as [1,Na]), y, weights, rssFloor and the per-voxel m
%   (mNZ) are taken from per-colour copies prepared once on the GPU, FWDfunc and the likelihood
%   are evaluated on these Na columns, and the hierarchical log-prior, log-Jacobian, bound check,
%   MRF term and acceptance are computed for them only. Accepted values are scattered back into
%   the full-size state and caches (u, x, log-likelihood, marginal statistics, log-prior,
%   log-Jacobian) through the GPU index arrays of the colour. Forward cost per sweep: ~1 full
%   evaluation (Nvar componentwise) instead of C.
%   Random numbers: the proposal normals and the acceptance uniforms are still drawn for all Nv
%   voxels (and the active columns used), so the random stream is exactly that of the full
%   evaluation; for a forward model whose columns are computed identically for any number of
%   columns, the chain is then the same as with subsetForward = false for the same seed.
%   Without the MRF (one class) and with the TEST ONLY mrfUpdate = 'simultaneous' nothing changes.
% Requirement: FWDfunc must be column-separable (output column v depends on parameter column v
%   only, [Nm, Na] output for Na input columns, for any Na). varargin is passed unchanged (NOT
%   subset), so a varargin entry with a voxel dimension (e.g. a per-voxel map) breaks it.
% Safety check (setup, no random numbers): FWDfunc is evaluated at the starting state once on all
%   voxels (gF) and once on every colour subset (gS). The subset evaluation is used only if every
%   gS has the size of gF(:, act) and
%       |gS - gF(:, act)| <= 1e-6 * max_rows |gF(:, v)|     for every element (column v),
%   with equal Inf/NaN positions. Bitwise equality is expected for elementwise models; 1e-6
%   (~8 single-precision ulps of the column scale) allows a different reduction/GEMM order.
%   Otherwise (a difference, a size mismatch or an error on a subset) the sampler falls back to
%   the full evaluation with the warning mcmc_bayes:subsetForwardFallback. The outcome is recorded
%   in out.settings.mrf.subsetForward (.requested, .used, .reason, .tolerance, .maxRelDiff, .bitwise).
%   The check cannot prove separability for all states; it catches the common failures
%   (voxel-dimensioned varargin, fixed-size models).
%
% Adaptive covariance (fitting.adaptCovariance = true; joint update with adaptStepSize only)
% -----------------------------------------------------------------------------------------
% Proposal. Per voxel i, over all d = Nvar sampled rows of u (all rows the joint proposal moves):
%       u' = u + lambda_i L_i eps,   eps ~ N(0, I_d),   L_i = chol(C_i, 'lower')
%   Only the proposal changes: acceptance, caches, priors, out-of-bound rejection of 'linear'/'log'
%   rows, the MRF colour steps (full and subsetForward) and the Gibbs block are as before. The
%   random stream is the same as the diagonal proposal (one [Nvar,Nv] randn per colour step).
% Phases (all rules are deterministic in k, the same for every voxel):
%   1. k <= switch iteration kS: the diagonal proposal (L_i = diag(sigma_i), lambda = 1), with the
%      usual per-voxel Robbins-Monro adaptation of log sigma (see Adaptation).
%   2. Accumulation: from iteration 2*adaptInterval + 1 (warm-up of 2 adaptation windows discarded)
%      to the end of burn-in, the state u_i after every iteration (accepted or not) updates the
%      running mean and covariance of each voxel (Welford, double precision on the GPU).
%   3. Switch: kS = the first adaptation step (k multiple of adaptInterval, within burn-in) with
%      n >= 10 d accumulated states. There, lambda_i = 2.38/sqrt(d) (Roberts & Rosenthal scaling:
%      proposal covariance (2.38^2/d) C_i) and the Robbins-Monro step continues on log lambda_i:
%       log lambda_i <- log lambda_i + gamma_j (acc_j - adaptTarget)   (same gain sequence j).
%   4. Refresh: at kS and at every later adaptation step within burn-in, C_i = sample covariance
%      (n-1 normalisation, symmetrised) regularised as
%       C_i <- C_i + 1e-6 diag(C_i) + 1e-12 max_p(C_i,pp) I    (+ realmin if C_i = 0),
%      L_i = batched per-voxel Cholesky (double, chol_batch). A voxel keeps its previous L_i (at kS:
%      diag(sigma_i) sqrt(d)/2.38, i.e. exactly its diagonal step) if the factorisation fails, a
%      diagonal entry is not positive/finite, or it accepted fewer than d+1 moves since the start of
%      the accumulation (C_i would be rank deficient).
%   5. Freeze: lambda_i and L_i are fixed at the end of burn-in (no adaptation step after it). The
%      kept chain is then a standard MH chain with a fixed symmetric Gaussian proposal per voxel, so
%      the target is unchanged. Without enough burn-in for the switch (kS does not exist) the
%      diagonal proposal is used throughout (warning mcmc_bayes:adaptCovarianceNoSwitch).
%   The sigmoid cap sigma <= u-width is not applied to lambda_i L_i.
% Output: out.diagnostics.adaptCovariance with .proposalCov [x,y,z,d,d,Nrep] (final lambda^2 L L'),
%   .lambda, .condition (of proposalCov), .valid (sample covariance used) [x,y,z,Nrep], .params and
%   .switchIteration; out.diagnostics.stepSize is sqrt(diag(proposalCov)); out.settings.adaptCovariance
%   (logical) and .adaptCovarianceRule.
%
% Phase 7 (Gaussian-mixture hierarchical prior, prior.hierarchical.K > 1)
% -----------------------------------------------------------------------
% Target. Per voxel i, u_i the [d,1] hierarchical parameters, z_i in {1..K} its group:
%       z_i ~ Categorical(pi),  u_i | z_i = k ~ N(mu_k, Sigma_k),  pi ~ Dirichlet(alpha,...,alpha),
%       (mu_k, Sigma_k) ~ NIW(m0, kappa0, Psi0, nu0) independently, the same resolved hyperprior for every k.
%   K = 1 is exactly Phase 3 (and runs the Phase 3 code unchanged). 'jeffreys_half' is not allowed
%   with K > 1 (improper per group; error mcmc_bayes:hierarchicalMixture).
% One sweep (free mode), a partially collapsed Gibbs sampler (van Dyk & Park 2008):
%   1. MH block on u | theta with z COLLAPSED, unchanged except the prior term, which is the full mixture
%      density (u-dependent part, the (2 pi)^(-d/2) constant dropped):
%          lp_i = log sum_k exp(log pi_k - log|Sigma_k|/2 - (u_i-mu_k)' Sigma_k^-1 (u_i-mu_k)/2),
%      a stable log-sum-exp (max over k subtracted) on the GPU in single precision; it does not depend on
%      z. All update paths (joint, componentwise, adaptive covariance, MRF colour steps, subsetForward)
%      and the cache logic are as before.
%   2. z-step, exact Gibbs, independent over voxels: log w_ik = log pi_k - log|Sigma_k|/2
%      - (u_i-mu_k)' Sigma_k^-1 (u_i-mu_k)/2 (GPU, double); normalised stably (max subtracted); one uniform
%      r_i per voxel (GPU stream): z_i = 1 + #{k < K : cumsum_k(w_i) < r_i * sum(w_i)}.
%   3. Per-group NIW: n_k, ubar_k, S_k of the voxels with z = k (GPU, double), then the exact conjugate
%      draw of Phase 3 per group, k = 1..K in order (host rng). An empty group (n_k = 0) draws from the
%      hyperprior NIW(m0, kappa0, Psi0, nu0) (the conjugate update with n = 0); with the default vague
%      kappa0 its mu is far from the data, so an emptied group is rarely re-populated.
%   4. pi ~ Dirichlet(alpha + n_1, ..., alpha + n_K): G_k ~ Gamma(alpha + n_k) (randg, host), pi = G/sum(G).
%   5. The per-voxel log-prior cache is recomputed.
%   Why the order matters: step 1 draws u from p(u | theta, y) (z integrated out) and step 2 then draws
%   z from p(z | u, theta), so steps 1-2 together are a blocked draw of (u, z) | theta, y; steps 3-4 are the
%   ordinary conditionals of theta given (u, z). Moving the z draw elsewhere (e.g. after theta, or skipping
%   it before the NIW step) would break the stationary distribution of a partially collapsed sampler.
% Fixed mode (fixed = true, .mu [d,K], .Sigma [d,d,K], .pi [K,1]; also stage 2 of run_two_stage): the
%   u updates use the collapsed mixture log-prior and z is never drawn (steps 2-4 skipped); voxels are
%   conditionally independent (they may be segmented).
% Membership (free and fixed mode): Rao-Blackwellised, the average over the kept samples of the exact
%   responsibilities P(z_i = k | u_i, theta) (theta of that sample, rows permuted to the ordered labels),
%   which has lower variance than counting z draws.
% Initialisation of the free mode (.init = 'kmeans', every repetition): k-means on the starting u of the
%   hierarchical parameters, each dimension standardised (mean 0, SD 1 over voxels). k-means++ seeding
%   (host rand, so deterministic given the rng seed), then Lloyd iterations (at most 100, until the labels
%   do not change; an emptied cluster is reseeded at the point farthest from its centre). Then (the
%   k-means labels are not kept: z is first drawn in step 2 of the first sweep) mu_k = mean of cluster k, Sigma_k = diag(max(var_k(u), floorVar)) (the Phase 3
%   variance floor, diagonal as for K = 1), pi_k = max(n_k,1) / sum_j max(n_j,1). No Statistics toolbox.
%   Needs K <= # voxels (error mcmc_bayes:hierarchicalMixture).
% Label switching. The kept samples of (mu_k, Sigma_k, pi_k) are stored after ordering the groups of each
%   kept sample by mu_k of the FIRST hierarchical parameter, ascending (sort, ties keep the lower label).
%   The same permutation reorders the membership rows of that sample. Summaries (mean/median), ESS and R-hat use the ordered
%   samples. Fixed mode keeps the user's labels (no switching possible).
% Output (K > 1): out.hyper.K, .posterior.mu [d,K,Ns,Nrep], .posterior.Sigma [d,d,K,Ns,Nrep],
%   .posterior.pi [K,Ns,Nrep]; .mean/.median .mu [d,K], .Sigma [d,d,K], .pi [K,1]; .ess/.rhat of the same
%   shapes; .membership [x,y,z,K] = Rao-Blackwellised P(z_i = k | y), the mean over the kept samples (all
%   repetitions) of P(z_i = k | u_i, theta) in the ordered labelling (fixed mode: user labels; final state
%   if there is no kept sample); .mapLabel [x,y,z] = argmax_k membership (0 outside the mask).
% Two-stage (run_two_stage, estimate_hyper_subset) with K > 1: stage 2 fixes the ordered stage-1 posterior
%   means of mu_k, Sigma_k and pi (fixed mode: collapsed log-prior, no z draws). The MRF weight default is W_p = 1/sqrt(V_pp), with
%   V the mixture's marginal covariance
%       V = sum_k pi_k (Sigma_k + mu_k mu_k') - mubar mubar',  mubar = sum_k pi_k mu_k,
%   and the Huber threshold is delta_p = huberDelta * sqrt(V_pp) (K = 1: V = Sigma, as before).
%
% Date created: 26 September 2026
% Date modified: 26 September 2026 (Phase 1: transforms, update schemes, adaptation, overdisp, diagnostics)
% Date modified: 26 September 2026 (Phase 2: marginal likelihoods, S0Param injection, nuisance recovery)
% Date modified: 26 September 2026 (Phase 3: hierarchical Normal prior; m counts non-zero weights only)
% Date modified: 26 September 2026 (Phase 4: MRF prior, chromatic updates, two-stage empirical Bayes)
% Date modified: 26 September 2026 (Phase 4b: forward model on the active colour only, subsetForward)
% Date modified: 27 September 2026 (adaptive-covariance joint proposal, adaptCovariance)
% Date modified: 27 September 2026 (Phase 7: Gaussian-mixture hierarchical prior, K > 1)
% Date modified: 28 September 2026 (Phase 8: transforms, adaptation, adaptive covariance, R-hat/ESS and the forward-size
%                                   check moved to mcmc as opt-in options/static helpers; location-shift move removed)
%

    methods
        function out = optimisation(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Input
        % ----------
        % Same as mcmc.optimisation, plus the new fitting options listed in
        % the class header.
        %

            isLegacy     = mcmc_bayes.isLegacy(fitting);
            forceNewPath = isstruct(fitting) && isfield(fitting,'forceNewPath') && ~isempty(fitting.forceNewPath) && fitting.forceNewPath;

            % legacy path: identical to mcmc
            if isLegacy && ~forceNewPath
                out = optimisation@mcmc(this, data, mask, weights, pars0, fitting, FWDfunc, varargin{:});
                return
            end

            % free hierarchical prior + MRF: two-stage empirical Bayes (run_two_stage), so that
            % model wrappers can reach it through optimisation
            if mcmc_bayes.is_two_stage(fitting)
                out = this.run_two_stage(data, mask, weights, pars0, fitting, FWDfunc, varargin{:});
                return
            end

            fitting = this.check_set_default_bayes(fitting);

            % only Metropolis-Hastings on the new path
            if ~strcmpi(fitting.algorithm,'mh')
                error('mcmc_bayes:unsupportedAlgorithm', ...
                    'mcmc_bayes: the new sampling path supports fitting.algorithm = ''MH'' only (got ''%s'').', fitting.algorithm);
            end

            % likelihood, sampled/marginalised parameter sets and prior (validated before any GPU work)
            [fittingS, lik] = this.setup_likelihood(fitting);
            hier = this.setup_hierarchical(fittingS);
            this.setup_mrf(fittingS, hier);         % MRF options (errors only)
            methodS  = this.parse_transform(fittingS.parameterTransform, numel(fittingS.modelParams));
            isHierS  = false(1, numel(fittingS.modelParams)); isHierS(hier.idx) = true;
            this.check_transform_bounds(methodS(~isHierS), fittingS.lb(~isHierS), fittingS.ub(~isHierS), fittingS.modelParams(~isHierS));
            if hier.on && hier.subsetFraction < 1
                error('mcmc_bayes:subsetFraction', ...
                    ['mcmc_bayes: prior.hierarchical.subsetFraction < 1 is used by estimate_hyper_subset (stage 1) only; ' ...
                     'call mcmc_bayes().estimate_hyper_subset(...) and then optimisation with the returned fixed-mode fitting.']);
            end

            % Step 0: display basic messages
            this.display_basic_algorithm_parameters(fitting);
            this.display_bayes_algorithm_parameters(fitting);

            % mask data to reduce memory load, same as mcmc; keep mask
            % geometry for the neighbour table of the MRF prior
            mask_idx = find(mask>0);
            geom     = struct('mask_idx', mask_idx, 'dims', size(mask));
            if ~ismatrix(data);     data    = utils.reshape_ND2GD(data,      mask_idx); else; data = data(:,mask_idx);     end
            if ~ismatrix(weights);  weights = utils.reshape_ND2GD(weights,   mask_idx); elseif ~isempty(weights); weights = weights(:,mask_idx);  end
            pars0 = utils.reshape_ND2GD_struct(pars0,mask);

            % MCMC
            [xPosterior, diagnostics] = this.metropolis_hastings_bayes(data, pars0, weights, fitting, geom, FWDfunc, varargin{:});

            % finish up, with the full (output) parameter list so that e.g. 'mode' finds lb/ub by name
            out = this.res2out(xPosterior,lik.fittingOut,mask,diagnostics);

        end

        function [muHat, SigmaHat, fittingFixed, outSub, idxSub, piHat] = estimate_hyper_subset(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Stage 1 of the two-stage scheme: run the free-hyperparameter sampler
        % on a random subset of the masked voxels and return the posterior
        % means of mu and Sigma, for use in fixed mode on the whole volume.
        %
        % Input
        % -----
        % Same as optimisation. fitting.prior.hierarchical must be set (not
        % fixed); .subsetFraction in (0,1] is the fraction of masked voxels
        % used (at least min(Nmask, 10) voxels). The subset is drawn with the
        % global rng (randperm). Limitation: FWDfunc's varargin is passed
        % unchanged, so it must not contain voxel-dimensioned inputs.
        %
        % Output
        % ------
        % muHat         : [d,1] posterior mean of mu (u space); K > 1: [d,K], ordered groups (Phase 7)
        % SigmaHat      : [d,d] posterior mean of Sigma (u space); K > 1: [d,d,K]
        % fittingFixed  : fitting with prior.hierarchical.fixed = true, .mu = muHat,
        %                 .Sigma = SigmaHat, .subsetFraction = 1 (stage 2 input); K > 1: also .pi = piHat
        % outSub        : mcmc_bayes output of the subset run (image dims [Nsub,1,1])
        % idxSub        : linear indices (into mask) of the subset voxels
        % piHat         : posterior mean of the group weights [K,1] (1 for K = 1)
        %
            if ~isstruct(fitting) || ~isfield(fitting,'prior') || ~isstruct(fitting.prior) || ...
                    ~isfield(fitting.prior,'hierarchical') || isempty(fitting.prior.hierarchical) || ...
                    (islogical(fitting.prior.hierarchical) && ~fitting.prior.hierarchical)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes.estimate_hyper_subset: fitting.prior.hierarchical is required.');
            end
            h = fitting.prior.hierarchical;
            if islogical(h); h = struct(); end
            if isfield(h,'fixed') && ~isempty(h.fixed) && h.fixed
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes.estimate_hyper_subset: prior.hierarchical.fixed must be false (stage 1 estimates mu and Sigma).');
            end
            frac = 1;
            if isfield(h,'subsetFraction') && ~isempty(h.subsetFraction); frac = h.subsetFraction; end
            if ~(isscalar(frac) && frac > 0 && frac <= 1)
                error('mcmc_bayes:subsetFraction', 'mcmc_bayes: prior.hierarchical.subsetFraction must be in (0,1].');
            end

            % random voxel subset
            mask_idx = find(mask>0);
            N        = numel(mask_idx);
            k        = min(N, max(ceil(frac*N), min(N,10)));
            idxSub   = mask_idx(sort(randperm(N, k)));

            % subset data as [k,1,1,...] images (or [Nm,k] if data is given as a matrix, as in mcmc)
            maskSub = true(k,1);
            data    = this.subset_voxels(data, mask, idxSub);
            if ~isempty(weights); weights = this.subset_voxels(weights, mask, idxSub); end
            fn = fieldnames(pars0);
            for kf = 1:numel(fn)
                v = pars0.(fn{kf});
                if ~isscalar(v); pars0.(fn{kf}) = reshape(v(idxSub), k, 1); end
            end

            f1 = fitting;
            h.fixed = false; h.subsetFraction = 1;
            f1.prior.hierarchical = h;
            outSub = this.optimisation(data, maskSub, weights, pars0, f1, FWDfunc, varargin{:});

            muHat    = outSub.hyper.mean.mu;
            SigmaHat = outSub.hyper.mean.Sigma;
            if isfield(outSub.hyper.mean, 'pi'); piHat = outSub.hyper.mean.pi; else; piHat = 1; end

            fittingFixed = fitting;
            h.fixed = true; h.mu = muHat; h.Sigma = SigmaHat; h.subsetFraction = 1;
            if numel(piHat) > 1; h.pi = piHat; end
            fittingFixed.prior.hierarchical = h;
        end

        function out = run_two_stage(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Two-stage empirical Bayes for the MRF prior (see the class header).
        %   Stage 1: free hierarchical prior (Gibbs block), NO MRF, on all voxels or, with
        %            prior.hierarchical.subsetFraction < 1, on a random voxel subset
        %            (estimate_hyper_subset).
        %   Stage 2: mu, Sigma (K > 1: mu_k, Sigma_k, pi of the ordered groups; z is still sampled)
        %            fixed at their stage-1 posterior means, hierarchical + MRF prior,
        %            same likelihood, on all voxels (single call). Start: per voxel, the stage-1
        %            posterior mean of u of every sampled parameter (mapped back to native space);
        %            voxels outside the stage-1 subset start from pars0.
        % This is NOT the full joint posterior p(u, mu, Sigma | y): the hyperparameter
        % uncertainty is not propagated to stage 2.
        %
        % Input
        % -----
        % Same as optimisation. fitting.prior.hierarchical (not fixed) and fitting.prior.mrf are
        % required. fitting.repetition etc. apply to both stages.
        %
        % Output
        % ------
        % out   : stage-2 output (as optimisation), plus
        %   .stage1                 : compact stage-1 summary: .hyper, .diagnostics, .settings,
        %                             .voxelIndex (linear indices into mask), .subsetFraction
        %   .settings.empiricalBayes: description of the two-stage approximation, muHat, SigmaHat
        %
            if ~isstruct(fitting) || ~isfield(fitting,'prior') || ~isstruct(fitting.prior) || ...
                    ~isfield(fitting.prior,'hierarchical') || isempty(fitting.prior.hierarchical) || ...
                    (islogical(fitting.prior.hierarchical) && ~fitting.prior.hierarchical)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes.run_two_stage: fitting.prior.hierarchical is required.');
            end
            if ~isfield(fitting.prior,'mrf') || isempty(fitting.prior.mrf) || (islogical(fitting.prior.mrf) && ~fitting.prior.mrf)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes.run_two_stage: fitting.prior.mrf is required (use optimisation without an MRF).');
            end
            h = fitting.prior.hierarchical;
            if islogical(h); h = struct(); end
            if isfield(h,'fixed') && ~isempty(h.fixed) && h.fixed
                error('mcmc_bayes:invalidPrior', ['mcmc_bayes.run_two_stage: prior.hierarchical.fixed must be false (stage 1 ' ...
                    'estimates mu and Sigma); with known mu/Sigma call optimisation directly.']);
            end
            frac = field_or_default(h, 'subsetFraction', 1);

            % stage 1: hierarchical prior only (the test-only mrfUpdate is a stage-2 option)
            f1 = fitting;
            f1.prior = rmfield(fitting.prior, 'mrf');
            if isfield(f1,'mrfUpdate'); f1 = rmfield(f1, 'mrfUpdate'); end
            mask_idx = find(mask>0);
            if frac < 1
                [muHat, SigmaHat, ~, out1, idxSub, piHat] = this.estimate_hyper_subset(data, mask, weights, pars0, f1, FWDfunc, varargin{:});
            else
                h1 = h; h1.fixed = false; h1.subsetFraction = 1;
                f1.prior.hierarchical = h1;
                out1    = this.optimisation(data, mask, weights, pars0, f1, FWDfunc, varargin{:});
                muHat   = out1.hyper.mean.mu;
                SigmaHat= out1.hyper.mean.Sigma;
                if isfield(out1.hyper.mean, 'pi'); piHat = out1.hyper.mean.pi; else; piHat = 1; end
                idxSub  = mask_idx;
            end

            % stage-2 start: per voxel, posterior mean of u of every sampled parameter
            [fS, ~] = this.setup_likelihood(this.check_set_default_bayes(fitting));
            method  = this.parse_transform(fS.parameterTransform, numel(fS.modelParams));
            pars2   = pars0;
            for k = 1:numel(fS.modelParams)
                p       = fS.modelParams{k};
                xs      = double(out1.posterior.(p));                        % [Nsub, Ns, Nrep]
                Nsub    = size(xs, 1);
                us      = this.transform_forward(reshape(xs, 1, []), method(k), fS.lb(k), fS.ub(k));
                uMean   = mean(reshape(us, Nsub, []), 2);
                xMean   = this.transform_inverse(uMean.', method(k), fS.lb(k), fS.ub(k));
                img     = double(pars0.(p));
                if isscalar(img); img = img .* ones(size(mask)); end
                img     = reshape(img, size(mask));
                img(idxSub) = xMean;
                pars2.(p) = img;
            end

            % stage 2: fixed mu, Sigma at the stage-1 posterior means, plus the MRF
            f2 = fitting;
            h2 = h; h2.fixed = true; h2.mu = muHat; h2.Sigma = SigmaHat; h2.subsetFraction = 1;
            isMixEB = numel(piHat) > 1;
            if isMixEB; h2.pi = piHat; end
            h2 = rmfield(h2, intersect(fieldnames(h2), {'hyperprior','m0','kappa0','Psi0','nu0','alpha','init'}));
            f2.prior.hierarchical = h2;
            out = this.optimisation(data, mask, weights, pars2, f2, FWDfunc, varargin{:});

            out.stage1 = struct('hyper', out1.hyper, 'diagnostics', out1.diagnostics, 'settings', out1.settings, ...
                                'voxelIndex', idxSub, 'subsetFraction', frac);
            out.settings.empiricalBayes = struct( ...
                'scheme',       'two-stage empirical Bayes (NOT the full joint posterior)', ...
                'stage1',       'free hierarchical prior (Gibbs block), no MRF; on a random voxel subset if subsetFraction < 1', ...
                'stage2',       'mu, Sigma fixed at the stage-1 posterior means; hierarchical + MRF prior; same likelihood', ...
                'init',         'stage-1 posterior mean of u per voxel (voxels outside the stage-1 subset: pars0)', ...
                'reason',       ['with the MRF the normalising constant of the joint prior on u depends on Sigma, so the ' ...
                                 'inverse-Wishart draw is not the exact conditional; the hyperparameter uncertainty is not propagated'], ...
                'muHat',        muHat, ...
                'SigmaHat',     SigmaHat, ...
                'Nstage1',      numel(idxSub), ...
                'subsetFraction', frac);
            if isMixEB
                out.settings.empiricalBayes.piHat  = piHat;
                out.settings.empiricalBayes.stage2 = ['mu_k, Sigma_k, pi fixed at the stage-1 posterior means of the ordered groups ' ...
                                                      '(z still sampled); hierarchical mixture + MRF prior; same likelihood'];
            end
        end

        function [xPosterior, diagnostics] = metropolis_hastings_bayes(this,y,x0,weights,fitting,geom,FWDfunc,varargin)
        % Input
        % ------
        % y         : measurements, [Nmeas,Nvoxels]
        % x0        : structure array, starting points, N fields, each field 1xNvoxel
        % weights   : weighting for non-linear least square fitting, same dimension as y
        % fitting   : Structure variable containing all fitting algorithm setting, see mcmc.metropolis_hastings
        %             and the class header for the new options
        % geom      : structure, mask geometry (.mask_idx, .dims), unused before Phase 4
        % FWDfunc   : function handle for forward signal model
        % varargin  : other input required for @FWDfunc
        %
        % Output
        % ------
        % xPosterior    : structure, native-space posterior samples, each field [Nvoxel, Nsample, Nrepetition] (as mcmc)
        % diagnostics   : structure
        %   .acceptance     : post-burn-in acceptance rate, [Nvoxel, Nblock, Nrepetition], Nblock = 1 (joint) or Nvar (componentwise)
        %   .stepSize       : final u-space proposal scale, [Nvoxel, Nvar, Nrepetition]
        %   .hyper          : hierarchical prior samples/settings ([] without hierarchical prior)
        %   .settings       : resolved sampler settings and RNG states
        %
            fitting = this.check_set_default_bayes(fitting);
            if isempty(weights); weights = ones(size(y), 'like', y); end

            % likelihood; from here on fitting.modelParams/lb/ub/xStepSize/parameterTransform
            % hold the SAMPLED parameters only (noise and S0Param removed under marginal
            % likelihoods, test-only fixedParams removed)
            [fitting, lik]  = this.setup_likelihood(fitting);
            isMarginal      = lik.isMarginal;
            % hierarchical prior on the sampled parameters
            hier            = this.setup_hierarchical(fitting);
            isHier          = hier.on;
            isGibbs         = isHier && ~hier.fixed;
            % Gaussian-mixture prior (Phase 7), K > 1 only; K = 1 runs the Phase 3 code unchanged
            isMix           = isHier && hier.K > 1;
            % MRF prior on the hierarchical parameters (fixed mu/Sigma only)
            mrf             = this.setup_mrf(fitting, hier);
            isMRF           = mrf.on;

            % record RNG states before any random number is drawn
            rngState    = rng;
            gpuRngState = parallel.gpu.rng;

            % Nm: # measurements; Nv: # voxels
            [Nm, Nv]    = size(y);
            % Nvar: # estimation parameters
            Nvar        = numel(fitting.modelParams);
            Nburnin     = this.get_number_burnin(fitting);
            % Ns: # samples in posterior distribution
            Ns          = numel(Nburnin+1:fitting.thinning:fitting.iteration);

            % marginal likelihoods: m = # measurements with non-zero weight, per voxel
            if isMarginal
                mNZ = sum(double(gather(weights)) ~= 0, 1);
                if min(mNZ) < 1 + lik.shapeOffset
                    error('mcmc_bayes:tooFewMeasurements', ...
                        'mcmc_bayes: likelihood ''%s'' needs at least %d measurements with non-zero weight per voxel (%d voxel(s) have fewer).', ...
                        lik.name, 1 + lik.shapeOffset, nnz(mNZ < 1 + lik.shapeOffset));
                end
                % scalar when all voxels have the same m (the common case)
                if all(mNZ == mNZ(1)); mNZ = mNZ(1); end
            end

            % free hyperparameters couple all voxels into this call: memory guard
            if isGibbs
                this.check_hierarchical_memory(Nm, Nv, Nvar, hier);
                if strcmp(hier.hyperprior,'jeffreys_half') && Nv <= 2*hier.d - 1
                    error('mcmc_bayes:hierarchicalTooFewVoxels', ...
                        'mcmc_bayes: hyperprior ''jeffreys_half'' needs more than 2d-1 = %d voxels (got %d).', 2*hier.d-1, Nv);
                end
                if isMix && hier.K > Nv
                    error('mcmc_bayes:hierarchicalMixture', ...
                        'mcmc_bayes: prior.hierarchical.K = %d groups need at least K voxels (got %d).', hier.K, Nv);
                end
            end

            % MRF: neighbour table, colour classes and edge weights (host), memory guard
            if isMRF
                if isempty(geom) || ~isfield(geom,'mask_idx') || numel(geom.mask_idx) ~= Nv
                    error('mcmc_bayes:mrfGeometry', 'mcmc_bayes: the MRF prior needs the mask geometry of all %d voxels.', Nv);
                end
                nbr     = this.build_neighbours(geom.mask_idx, geom.dims, mrf.mode, mrf.radius, mrf.connectivity);
                Knb     = size(nbr, 1);
                this.check_mrf_memory(Nm, Nv, Nvar, Knb, mrf);
                if strcmp(mrf.update, 'simultaneous')
                    % TEST ONLY negative control: one class with all voxels (wrong target)
                    colours = ones(1, Nv); NcolNominal = 1;
                else
                    [colours, NcolNominal] = this.build_colours(geom.mask_idx, geom.dims, mrf.mode, mrf.radius, mrf.connectivity, nbr);
                end
                if isempty(mrf.edgeWeights)
                    wEdge = double(nbr > 0);
                else
                    wEdge = this.check_edge_weights(mrf.edgeWeights, nbr);
                end
                Nedges  = nnz(nbr) / 2;
            end

            % transforms
            method      = this.parse_transform(fitting.parameterTransform, Nvar);
            isHierRow   = false(1,Nvar); if isHier; isHierRow(hier.idx) = true; end
            % bounds of the hierarchical parameters are checked in setup_hierarchical
            this.check_transform_bounds(method(~isHierRow), fitting.lb(~isHierRow), fitting.ub(~isHierRow), fitting.modelParams(~isHierRow));
            isLinear    = strcmp(method,'linear');
            hasJac      = ~all(isLinear);                           % false -> x == u, neutral path, skip all Jacobian terms
            useJac      = ~isLinear & ~isHierRow;                   % rows whose log-Jacobian enters the target
            dropJac     = any(~isLinear & isHierRow);               % hierarchical rows: log-Jacobian set to 0
            jacRow      = gpuArray(single(useJac(:)));
            isRejectH   = ~strcmp(method,'sigmoid') & ~isHierRow;   % rows whose box is enforced by rejection
            isRejectRow = gpuArray(isRejectH(:));
            code        = gpuArray(single(this.transform_code(method)));  % [Nvar,1] for the fused GPU kernel
            allReject   = all(isRejectH);                           % true -> same bound check as legacy

            % update scheme and adaptation
            isComponent = strcmpi(fitting.updateScheme,'componentwise');
            if isComponent; Nblock = Nvar; else; Nblock = 1; end
            isAdapt     = logical(fitting.adaptStepSize);
            Nadapt      = floor(Nburnin/fitting.adaptInterval);     % # adaptation windows, all within burn-in
            if isAdapt && Nburnin < 2*fitting.adaptInterval
                warning('mcmc_bayes:shortBurnin', ...
                    'adaptStepSize is on but Nburnin (%d) < 2*adaptInterval (%d); %d adaptation step(s) only.', ...
                    Nburnin, 2*fitting.adaptInterval, Nadapt);
            end
            % adaptive covariance (joint + adaptStepSize, validated in check_set_default_bayes)
            isACov      = logical(fitting.adaptCovariance);
            if isACov
                acWarm  = 2*fitting.adaptInterval;              % warm-up iterations not accumulated
                acMinN  = 10*Nvar;                              % accumulated states needed for the switch
                acScale = 2.38/sqrt(Nvar);                      % initial lambda at the switch
                acEps   = 1e-6;                                 % relative diagonal loading
                acTiny  = 1e-12;                                % ridge, relative to max_p C_pp
                kAd     = (1:Nadapt) .* fitting.adaptInterval;  % adaptation steps
                kSwitch = kAd(find(kAd - acWarm >= acMinN, 1));
                if isempty(kSwitch)
                    kSwitch = NaN;
                    warning('mcmc_bayes:adaptCovarianceNoSwitch', ...
                        ['adaptCovariance is on but the burn-in (%d) is too short to accumulate %d states after the ' ...
                         'warm-up of %d iterations at an adaptation step; the diagonal proposal is used throughout.'], ...
                        Nburnin, acMinN, acWarm);
                end
            end

            % convert data into single datatype for better performance and put them into GPU
            y       = gpuArray( single(y) );
            weights = gpuArray( single(weights) );
            for km = 1:Nvar; x0.(fitting.modelParams{km}) = gpuArray(single( x0.(fitting.modelParams{km}) ));end
            xStepsize = gpuArray(single(fitting.xStepSize(:)));
            % setup boundary variables, [Nvar,1], broadcast over voxels
            lb      = gpuArray( single(fitting.lb(:)));
            ub      = gpuArray( single(fitting.ub(:)));
            % u-space bounds of the eps-clamped box (for overdisp and step-size cap);
            % hierarchical 'log'/'linear' rows are unbounded in u
            uLo     = this.transform_forward(lb, method, lb, ub);
            uHi     = this.transform_forward(ub, method, lb, ub);
            isUnbounded = isHierRow & ~strcmp(method,'sigmoid');
            if any(isUnbounded); uLo(isUnbounded) = -Inf; uHi(isUnbounded) = Inf; end
            uWidth  = uHi - uLo;
            % initialize array to store all the samples
            xPosterior  = zeros(Nvar, Nv, Ns, fitting.repetition,'single');
            acceptance  = zeros(Nv, Nblock, fitting.repetition,'single');
            stepSize    = zeros(Nv, Nvar, fitting.repetition,'single');
            if isACov
                acPropCov   = zeros(Nv, Nvar, Nvar, fitting.repetition, 'single');
                acLambda    = zeros(Nv, fitting.repetition, 'single');
                acCond      = zeros(Nv, fitting.repetition, 'single');
                acValidOut  = false(Nv, fitting.repetition);
            end

            % test-only fixed parameters, [1,Nv] each
            userFixed = struct();
            ufNames   = fieldnames(lik.userFixed);
            for k = 1:numel(ufNames); userFixed.(ufNames{k}) = lik.userFixed.(ufNames{k}) .* ones(1, Nv, 'like', y); end
            hasUserFixed = ~isempty(ufNames);

            % log-likelihood of a native-space parameter array [Nvar,Nv]
            if isMarginal
                % constant amplitude S0Param = 1 and dummy noise = 1 injected before FWDfunc;
                % the second output is the cached sufficient statistics of the state
                fixedVal    = ones(1, Nv, 'like', y);
                rssFloor    = max(10*eps('single')^2 .* sum(weights.*y.^2, 1), realmin('single'));
                if isscalar(mNZ); mArg = mNZ; else; mArg = gpuArray(single(mNZ)); end
                loglik      = @(x) mcmc_bayes.loglik_marginal( ...
                                FWDfunc(mcmc_bayes.inject_values(mcmc_bayes.inject_fixed(this.array2struct(x,fitting.modelParams), lik.fixedParams, fixedVal), userFixed), varargin{:}), ...
                                y, weights, lik.name, rssFloor, mArg);
                Nstat       = lik.Nstat;
                statsPost   = zeros(Nstat, Nv, Ns, fitting.repetition, 'single');
            elseif hasUserFixed
                loglik      = @(x) mcmc_bayes.loglik_gaussian(mcmc_bayes.inject_values(this.array2struct(x,fitting.modelParams), userFixed), y, weights, Nm, FWDfunc, varargin{:});
            else
                loglik      = @(x) mcmc_bayes.loglik_gaussian(this.array2struct(x,fitting.modelParams), y, weights, Nm, FWDfunc, varargin{:});
            end

            % starting point in native space (same as mcmc), then in u space
            xStart  = this.struct2array(x0,fitting.modelParams);      % extract parameter structure to numeric array for faster computation
            xStart  = max(xStart,lb); xStart = min(xStart,ub);         % set boundary
            uStart  = this.transform_forward(xStart, method, lb, ub);

            % the forward model must return one row per measurement and one column per voxel
            parsChk = this.array2struct(xStart, fitting.modelParams);
            if isMarginal; parsChk = mcmc_bayes.inject_fixed(parsChk, lik.fixedParams, fixedVal); end
            if hasUserFixed; parsChk = mcmc_bayes.inject_values(parsChk, userFixed); end
            mcmc_bayes.check_forward_size(FWDfunc(parsChk, varargin{:}), Nm, Nv);
            clear parsChk

            % initial u-space proposal scale at the start point (used for overdisp of
            % unbounded rows and for the hierarchical variance floor)
            if isHier || any(isUnbounded)
                sigmaStart = repmat(xStepsize, 1, Nv);
                if hasJac
                    logJs = this.transform_logjac(uStart, method, lb, ub);
                    sigmaStart(~isLinear,:) = min( sigmaStart(~isLinear,:) ./ exp(logJs(~isLinear,:)), uWidth(~isLinear) );
                end
            end
            % over-dispersion scale: the u-width, or 100 x the initial step for unbounded rows
            if any(isUnbounded) && fitting.overdisp > 0
                Wod = repmat(uWidth, 1, Nv);
                Wod(isUnbounded,:) = 100 .* sigmaStart(isUnbounded,:);
            else
                Wod = uWidth;
            end

            % hierarchical prior: resolve the hyperprior defaults and allocate the hyper samples
            if isHier
                hIdx    = hier.idx;
                d       = hier.d;
                floorVar = (10 .* median(double(gather(sigmaStart(hIdx,:))), 2)).^2;   % [d,1]
                hp      = this.resolve_hyperprior(hier, double(gather(uStart(hIdx,:))), floorVar);
                if isMix
                    % mixture (Phase 7): ordered kept samples, final state, membership counts
                    Kmix        = hier.K;
                    if isGibbs
                        muPost      = zeros(d, Kmix, Ns, fitting.repetition);
                        SigmaPost   = zeros(d, d, Kmix, Ns, fitting.repetition);
                        piPost      = zeros(Kmix, Ns, fitting.repetition);
                        muLast      = zeros(d, Kmix, fitting.repetition);
                        SigmaLast   = zeros(d, d, Kmix, fitting.repetition);
                        piLast      = zeros(Kmix, fitting.repetition);
                    end
                    memCount    = zeros(Kmix, Nv, 'gpuArray');      % sum over kept samples of P(z_i = k | u_i, theta) (ordered labels)
                    memLast     = zeros(Kmix, Nv, 'gpuArray');      % same at the final state (fallback without kept samples)
                elseif isGibbs
                    muPost      = zeros(d, Ns, fitting.repetition);
                    SigmaPost   = zeros(d, d, Ns, fitting.repetition);
                    muLast      = zeros(d, fitting.repetition);         % final Gibbs state per chain
                    SigmaLast   = zeros(d, d, fitting.repetition);
                end
            end
            lpC   = [];     % mixture log-prior constants log pi_k - log|Sigma_k|/2 (K > 1 only; [] selects the single-Normal log-prior)
            if fitting.checkCache
                cacheErr = struct('loglik', 0, 'logprior', 0, 'logjac', 0, 'Ncheck', 0);
                if isMRF; cacheErr.inactiveMoved = 0; end
            end

            % MRF on the GPU, per colour class: active voxels, their neighbours and edge weights.
            % An absent neighbour points to the voxel itself with weight 0 (contributes 0).
            if isMRF
                colList = unique(colours);                  % non-empty classes only
                Ncol    = numel(colList);
                selfIdx = repmat(int32(1:Nv), Knb, 1);
                nbrSelf = nbr; nbrSelf(nbr == 0) = selfIdx(nbr == 0);
                mrfAct  = cell(1, Ncol); mrfMask = cell(1, Ncol); mrfNbr = cell(1, Ncol); mrfW = cell(1, Ncol);
                mrfActH = cell(1, Ncol);
                for kc = 1:Ncol
                    isC         = colours == colList(kc);
                    mrfActH{kc} = find(isC);
                    mrfAct{kc}  = gpuArray(uint32(mrfActH{kc}));
                    mrfMask{kc} = gpuArray(isC);
                    mrfNbr{kc}  = gpuArray(uint32(nbrSelf(:, isC)));
                    mrfW{kc}    = gpuArray(single(wEdge(:, isC)));
                end
                clear selfIdx nbrSelf
                mrfCoef  = gpuArray(single(mrf.W(:) ./ mrf.tau));  % [d,1] W_p/tau
                mrfDelta = gpuArray(single(mrf.delta(:)));          % [d,1] Huber thresholds (u space)

                % Phase 4b: forward model and likelihood on the active colour only (see the header)
                sfTol  = 1e-6;
                sfInfo = struct('requested', logical(mrf.subsetForward), 'used', false, 'reason', '', ...
                                'tolerance', sfTol, 'maxRelDiff', [], 'bitwise', []);
                if ~strcmp(mrf.update, 'chromatic')
                    sfInfo.reason = 'mrfUpdate = ''simultaneous'' (TEST ONLY): one class with all voxels, full evaluation';
                elseif ~mrf.subsetForward
                    sfInfo.reason = 'disabled (prior.mrf.subsetForward = false): full evaluation in every colour step';
                else
                    % per colour: y, weights (and rssFloor, m) of the active voxels, [1,Na] injected
                    % values, and the likelihood handle on these columns (all on the GPU, built once)
                    mkPars  = @(x, fv, uf) mcmc_bayes.inject_values(mcmc_bayes.inject_fixed( ...
                                    this.array2struct(x, fitting.modelParams), lik.fixedParams, fv), uf);
                    loglikC = cell(1, Ncol); parsC = cell(1, Ncol);
                    for kc = 1:Ncol
                        act = mrfAct{kc}; NaK = numel(mrfActH{kc});
                        yA  = y(:, act); wA = weights(:, act);
                        fvA = ones(1, NaK, 'like', y);
                        ufA = struct();
                        for q = 1:numel(ufNames); ufA.(ufNames{q}) = lik.userFixed.(ufNames{q}) .* ones(1, NaK, 'like', y); end
                        if isMarginal
                            rfA = rssFloor(act);
                            if isscalar(mArg); mA = mArg; else; mA = mArg(act); end
                            loglikC{kc} = @(x) mcmc_bayes.loglik_marginal(FWDfunc(mkPars(x, fvA, ufA), varargin{:}), ...
                                                yA, wA, lik.name, rfA, mA);
                        elseif hasUserFixed
                            loglikC{kc} = @(x) mcmc_bayes.loglik_gaussian(mcmc_bayes.inject_values(this.array2struct(x,fitting.modelParams), ufA), ...
                                                yA, wA, Nm, FWDfunc, varargin{:});
                        else
                            loglikC{kc} = @(x) mcmc_bayes.loglik_gaussian(this.array2struct(x,fitting.modelParams), yA, wA, Nm, FWDfunc, varargin{:});
                        end
                        parsC{kc} = mkPars(xStart(:, act), fvA, ufA);
                    end
                    clear yA wA fvA ufA rfA mA
                    % safety check at the starting state: subset columns == columns of the full evaluation
                    sfInfo = this.check_subset_forward(FWDfunc, varargin, mkPars(xStart, ones(1, Nv, 'like', y), userFixed), ...
                                                       parsC, mrfActH, Nv, sfTol);
                    clear parsC
                    if ~sfInfo.used
                        warning('mcmc_bayes:subsetForwardFallback', ...
                            ['mcmc_bayes: prior.mrf.subsetForward: %s. Falling back to the full forward evaluation in every ' ...
                             'colour step (same target, ~%d x the forward cost). The forward model must be column-separable and ' ...
                             'accept any number of voxels; varargin entries with a voxel dimension are not subset. Set ' ...
                             'prior.mrf.subsetForward = false to skip this check.'], sfInfo.reason, Ncol);
                        clear loglikC
                    end
                end
                useSubset = sfInfo.used;
            else
                Ncol    = 1;
                useSubset = false;
            end

            disp('-------------------------');
            disp('MCMC optimisation process');
            disp('-------------------------');

            % loop (multiple) proposal
            for ii = 1:fitting.repetition
            fprintf('Repetition #%i/%i \n',ii,fitting.repetition)

            % reset start point, over-dispersed in u space for ii > 1
            uCurr = uStart;
            if ii > 1 && fitting.overdisp > 0
                uCurr = uCurr + fitting.overdisp .* Wod .* randn(size(uCurr),'like',uCurr);
                uCurr = max(uCurr,uLo); uCurr = min(uCurr,uHi);
            end
            if hasJac || ii > 1
                xCurr = this.transform_inverse(uCurr, method, lb, ub);
                xCurr = max(xCurr,lb); xCurr = min(xCurr,ub);
            else
                xCurr = xStart;
            end
            if isMarginal; [logLCurr, statsCurr] = loglik(xCurr); else; logLCurr = loglik(xCurr); end
            if hasJac
                logJCurr = this.transform_logjac(uCurr, method, lb, ub);
                if dropJac; logJCurr = logJCurr .* jacRow; end
            end

            % hyperparameters and cached per-voxel log-prior
            if isMix
                % mixture (Phase 7): free mode from k-means on the current u, fixed mode from the user values
                % with z = the most probable group of the current u
                if isGibbs
                    [~, mu, Sigma, piW] = this.mixture_init(double(uCurr(hIdx,:)), Kmix, floorVar);
                else
                    mu = hier.mu; Sigma = hier.Sigma; piW = hier.pi;
                end
                [muG, PG, mixD] = this.mixture_to_gpu(mu, Sigma, piW);
                lpC         = mixD.cG;
                lpCurr      = this.logprior_mix(uCurr(hIdx,:), lpC, muG, PG);
            elseif isHier
                if isGibbs
                    % initial hyperparameters: moments of the current u (variance floored)
                    uH0     = double(gather(uCurr(hIdx,:)));
                    mu      = mean(uH0, 2);
                    Sigma   = diag(max(var(uH0, 0, 2), floorVar));
                else
                    mu      = hier.mu;
                    Sigma   = hier.Sigma;
                end
                [muG, PG]   = this.prior_to_gpu(mu, Sigma);
                lpCurr      = this.logprior_normal(uCurr(hIdx,:), muG, PG);
            end

            % initial proposal scale in u space: xStepSize / |dx/du| at the start point
            sigma = repmat(xStepsize, 1, Nv);
            if hasJac
                logJ0           = this.transform_logjac(uCurr, method, lb, ub);
                sigma(~isLinear,:) = min( sigma(~isLinear,:) ./ exp(logJ0(~isLinear,:)), uWidth(~isLinear) );
            end

            accWin  = zeros(Nblock, Nv, 'like', uCurr);     % acceptance counts in the current adaptation window
            accPost = zeros(Nblock, Nv, 'like', uCurr);     % acceptance counts after burn-in
            jAdapt  = 0;

            % adaptive covariance: running moments (double), acceptances since the accumulation start,
            % and the proposal factor lambda.*L (used once acPhase is true)
            if isACov
                acPhase = false;
                acN     = 0;
                acMean  = zeros(Nvar, Nv, 'double', 'gpuArray');
                acM2    = zeros(Nvar, Nvar, Nv, 'double', 'gpuArray');
                acAccN  = zeros(1, Nv, 'like', uCurr);
                acValid = false(1, Nv, 'gpuArray');
                acLam   = [];  acL = [];  acLprop = [];
            end

            counter = 0; start = tic;
            for k = 1:fitting.iteration
                % 1-2. MH block, one colour class at a time (a single class with all voxels without MRF)
                if isComponent
                    isAccepted = false(Nvar, Nv, 'like', isRejectRow);
                elseif isMRF
                    isAccSweep = false(1, Nv, 'like', isRejectRow);
                end
                for kc = 1:Ncol
                if isMRF
                    act = mrfAct{kc};
                    if fitting.checkCache; uBefore = uCurr; end
                end

                if useSubset && ~isComponent
                    % ========== joint update, active colour only (Phase 4b) ==========
                    % the random numbers are drawn for all voxels (same stream as the full evaluation)
                    % and the active columns are used; everything else is computed on the Na active voxels
                    zAll            = randn(size(uCurr),'like',uCurr);
                    uA              = uCurr(:,act);
                    if isACov && acPhase
                        uProposed   = uA + mcmc_bayes.acov_step(acLprop(:,:,act), zAll(:,act));
                    else
                        uProposed   = uA + sigma(:,act).*zAll(:,act);
                    end
                    if hasJac
                        [xProposed, logJProposed] = this.transform_inverse_logjac_fused(uProposed, code, lb, ub);
                        if dropJac; logJProposed = logJProposed .* jacRow; end
                    else
                        xProposed = uProposed;
                    end
                    if allReject
                        isOutofbound = max(or(xProposed<lb, xProposed>ub),[],1);
                    else
                        isOutofbound = max(isRejectRow & or(xProposed<lb, xProposed>ub),[],1);
                    end
                    xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);

                    % forward model and likelihood on the active voxels only
                    if isMarginal; [logLProposed, statsProposed] = loglikC{kc}(xProposed); else; logLProposed = loglikC{kc}(xProposed); end
                    lpProposed  = this.logprior_mix(uProposed(hIdx,:), lpC, muG, PG);   % the MRF requires the hierarchical prior
                    dPhi        = this.mrf_local_delta(uProposed(hIdx,:), uA(hIdx,:), uCurr(hIdx,:), ...
                                                       mrfNbr{kc}, mrfW{kc}, mrfCoef, mrfDelta, mrf.potential, mrf.stateWeight);
                    if hasJac
                        logRatio    = logLProposed - logLCurr(act) + sum(logJProposed - logJCurr(:,act), 1);
                    else
                        logRatio    = logLProposed - logLCurr(act);
                    end
                    logRatio        = logRatio + (lpProposed - lpCurr(act));
                    logRatio        = logRatio - dPhi;
                    rAll            = rand(1,Nv,'like',logLProposed);
                    isAccA          = exp(logRatio) > rAll(act);    % NaN is rejected
                    isAccA(isOutofbound) = 0;
                    % scatter the accepted voxels back into the full-size state and caches
                    isAccSweep(act) = isAccA;
                    accIdx          = act(isAccA);
                    logLCurr(accIdx)    = logLProposed(isAccA);
                    uCurr(:,accIdx)     = uProposed(:,isAccA);
                    if isMarginal; statsCurr(:,accIdx) = statsProposed(:,isAccA); end
                    lpCurr(accIdx)      = lpProposed(isAccA);
                    if hasJac
                        xCurr(:,accIdx)     = xProposed(:,isAccA);
                        logJCurr(:,accIdx)  = logJProposed(:,isAccA);
                    end

                elseif useSubset
                    % ========== componentwise update, active colour only (Phase 4b) ==========
                    % local copies of the active columns, written back after the last parameter
                    uA      = uCurr(:,act);
                    xA      = xCurr(:,act);
                    sigmaA  = sigma(:,act);
                    logLA   = logLCurr(act);
                    lpA     = lpCurr(act);
                    if isMarginal; statsA = statsCurr(:,act); end
                    if hasJac; logJA = logJCurr(:,act); end
                    accA    = false(Nvar, numel(lpA), 'like', isRejectRow);
                    for kp = 1:Nvar
                        zAll            = randn(1,Nv,'like',uCurr);
                        uProposed_p     = uA(kp,:) + sigmaA(kp,:).*zAll(act);
                        if isLinear(kp)
                            xProposed_p = uProposed_p;
                        else
                            [xProposed_p, logJProposed_p] = this.transform_inverse_logjac_fused(uProposed_p, code(kp), lb(kp), ub(kp));
                        end
                        isOutofbound    = isRejectRow(kp) & or(xProposed_p<lb(kp), xProposed_p>ub(kp));
                        xProposed_p     = max(xProposed_p,lb(kp)); xProposed_p = min(xProposed_p,ub(kp));
                        xProposed       = xA; xProposed(kp,:) = xProposed_p;

                        if isMarginal; [logLProposed, statsProposed] = loglikC{kc}(xProposed); else; logLProposed = loglikC{kc}(xProposed); end
                        logRatio        = logLProposed - logLA;
                        if useJac(kp); logRatio = logRatio + logJProposed_p - logJA(kp,:); end
                        if isHierRow(kp)
                            uProposedH          = uA(hIdx,:);
                            uProposedH(hIdx==kp,:) = uProposed_p;
                            lpProposed          = this.logprior_mix(uProposedH, lpC, muG, PG);
                            logRatio            = logRatio + (lpProposed - lpA);
                            pH                  = find(hIdx==kp);
                            dPhi                = this.mrf_local_delta(uProposed_p, uA(kp,:), uCurr(kp,:), ...
                                                    mrfNbr{kc}, mrfW{kc}, mrfCoef(pH), mrfDelta(pH), mrf.potential, mrf.stateWeight);
                            logRatio            = logRatio - dPhi;
                        end
                        rAll                        = rand(1,Nv,'like',logLProposed);
                        isAccepted_p                = exp(logRatio) > rAll(act);
                        isAccepted_p(isOutofbound)  = 0;
                        logLA(isAccepted_p)         = logLProposed(isAccepted_p);
                        if isMarginal; statsA(:,isAccepted_p) = statsProposed(:,isAccepted_p); end
                        if isHierRow(kp); lpA(isAccepted_p) = lpProposed(isAccepted_p); end
                        uA(kp,isAccepted_p)         = uProposed_p(isAccepted_p);
                        xA(kp,isAccepted_p)         = xProposed_p(isAccepted_p);
                        if useJac(kp); logJA(kp,isAccepted_p) = logJProposed_p(isAccepted_p); end
                        accA(kp,:)                  = isAccepted_p;
                    end
                    % scatter the active columns back (inactive voxels untouched)
                    uCurr(:,act)    = uA;
                    xCurr(:,act)    = xA;
                    logLCurr(act)   = logLA;
                    lpCurr(act)     = lpA;
                    if isMarginal; statsCurr(:,act) = statsA; end
                    if hasJac; logJCurr(:,act) = logJA; end
                    isAccepted(:,act) = accA;

                elseif ~isComponent
                    % ========== joint update: all parameters of a voxel together ==========
                    % 1. make a proposal with normal distribution in u space
                    if isACov && acPhase
                        uProposed   = uCurr + mcmc_bayes.acov_step(acLprop, randn(size(uCurr),'like',uCurr));
                    else
                        uProposed   = uCurr + sigma.*randn(size(uCurr),'like',uCurr);
                    end
                    % back to native space; find proposal that is out of bound for exclusion
                    if hasJac
                        % fused inverse transform and log-Jacobian (one GPU kernel)
                        [xProposed, logJProposed] = this.transform_inverse_logjac_fused(uProposed, code, lb, ub);
                        if dropJac; logJProposed = logJProposed .* jacRow; end
                    else
                        xProposed = uProposed;
                    end
                    if allReject
                        isOutofbound = max(or(xProposed<lb, xProposed>ub),[],1);
                    else
                        isOutofbound = max(isRejectRow & or(xProposed<lb, xProposed>ub),[],1);
                    end
                    % replace out of bound by boundary values to avoid error when computing probability
                    xProposed = max(xProposed,lb); xProposed = min(xProposed,ub);

                    % 2. Metropolis sampling
                    % 2.1 proposal probability (+ log-Jacobian of the transform, + hierarchical log-prior)
                    if isMarginal; [logLProposed, statsProposed] = loglik(xProposed); else; logLProposed = loglik(xProposed); end
                    if isHier; lpProposed = this.logprior_mix(uProposed(hIdx,:), lpC, muG, PG); end
                    % MRF: change of the local term of the active voxels, current neighbours (never cached)
                    if isMRF
                        dPhi = this.mrf_local_delta(uProposed(hIdx,act), uCurr(hIdx,act), uCurr(hIdx,:), ...
                                                    mrfNbr{kc}, mrfW{kc}, mrfCoef, mrfDelta, mrf.potential, mrf.stateWeight);
                    end
                    % 2.2 accept with probability min(1, exp(logRatio)); NaN is rejected
                    if hasJac
                        logRatio            = logLProposed - logLCurr + sum(logJProposed - logJCurr, 1);
                        if isHier; logRatio = logRatio + (lpProposed - lpCurr); end
                        if isMRF; logRatio(act) = logRatio(act) - dPhi; end
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    elseif isMarginal || isHier
                        % degenerate states have logL = -Inf; -Inf - (-Inf) = NaN is rejected
                        logRatio            = logLProposed - logLCurr;
                        if isHier; logRatio = logRatio + (lpProposed - lpCurr); end
                        if isMRF; logRatio(act) = logRatio(act) - dPhi; end
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    else
                        % neutral path: same expression as mcmc.metropolis_hastings (the GPU evaluates
                        % fused elementwise expressions slightly differently, so keep it verbatim)
                        acceptanceRatio     = min(exp(logLProposed-logLCurr), 1);
                        isAccepted          = acceptanceRatio > rand(1,Nv,'like',logLProposed);
                        isOutofbound        = isOutofbound | isnan(logLProposed);  % mcmc would accept NaN via min(NaN,1) = 1
                    end
                    isAccepted(isOutofbound)= 0;    % reject out of bound (and NaN) proposal
                    % MRF: only the voxels of the active colour can move
                    if isMRF
                        isAccepted  = isAccepted & mrfMask{kc};
                        isAccSweep  = isAccSweep | isAccepted;
                    end
                    % 2.3 update parameters if accepted
                    logLCurr(isAccepted)    = logLProposed(isAccepted);
                    uCurr(:,isAccepted)     = uProposed(:,isAccepted);
                    if isMarginal; statsCurr(:,isAccepted) = statsProposed(:,isAccepted); end
                    if isHier; lpCurr(isAccepted) = lpProposed(isAccepted); end
                    % neutral path (all linear): x == u, only u is tracked in the joint loop
                    if hasJac
                        xCurr(:,isAccepted)     = xProposed(:,isAccepted);
                        logJCurr(:,isAccepted)  = logJProposed(:,isAccepted);
                    end

                else
                    % ========== componentwise update: one parameter at a time ==========
                    for kp = 1:Nvar
                        % 1. proposal for parameter kp only
                        uProposed_p     = uCurr(kp,:) + sigma(kp,:).*randn(1,Nv,'like',uCurr);
                        if isLinear(kp)
                            xProposed_p = uProposed_p;
                        else
                            [xProposed_p, logJProposed_p] = this.transform_inverse_logjac_fused(uProposed_p, code(kp), lb(kp), ub(kp));
                        end
                        isOutofbound    = isRejectRow(kp) & or(xProposed_p<lb(kp), xProposed_p>ub(kp));
                        xProposed_p     = max(xProposed_p,lb(kp)); xProposed_p = min(xProposed_p,ub(kp));
                        xProposed       = xCurr; xProposed(kp,:) = xProposed_p;

                        % 2. Metropolis sampling, cached loglik (and log-prior) is the current state
                        if isMarginal; [logLProposed, statsProposed] = loglik(xProposed); else; logLProposed = loglik(xProposed); end
                        logRatio        = logLProposed - logLCurr;
                        if useJac(kp); logRatio = logRatio + logJProposed_p - logJCurr(kp,:); end
                        if isHierRow(kp)
                            uProposedH          = uCurr(hIdx,:);
                            uProposedH(hIdx==kp,:) = uProposed_p;
                            lpProposed          = this.logprior_mix(uProposedH, lpC, muG, PG);
                            logRatio            = logRatio + (lpProposed - lpCurr);
                            % MRF: change of the local term of parameter kp of the active voxels
                            if isMRF
                                pH              = find(hIdx==kp);
                                dPhi            = this.mrf_local_delta(uProposed_p(act), uCurr(kp,act), uCurr(kp,:), ...
                                                    mrfNbr{kc}, mrfW{kc}, mrfCoef(pH), mrfDelta(pH), mrf.potential, mrf.stateWeight);
                                logRatio(act)   = logRatio(act) - dPhi;
                            end
                        end
                        isAccepted_p                = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                        isAccepted_p(isOutofbound)  = 0;
                        if isMRF; isAccepted_p = isAccepted_p & mrfMask{kc}; end
                        % 3. update parameter kp and the cache before the next parameter
                        logLCurr(isAccepted_p)      = logLProposed(isAccepted_p);
                        if isMarginal; statsCurr(:,isAccepted_p) = statsProposed(:,isAccepted_p); end
                        if isHierRow(kp); lpCurr(isAccepted_p) = lpProposed(isAccepted_p); end
                        uCurr(kp,isAccepted_p)      = uProposed_p(isAccepted_p);
                        xCurr(kp,isAccepted_p)      = xProposed_p(isAccepted_p);
                        if useJac(kp); logJCurr(kp,isAccepted_p) = logJProposed_p(isAccepted_p); end
                        if isMRF
                            isAccepted(kp,:)        = isAccepted(kp,:) | isAccepted_p;
                        else
                            isAccepted(kp,:)        = isAccepted_p;
                        end
                    end
                end
                % test only: voxels outside the active colour must not have changed
                if isMRF && fitting.checkCache
                    cacheErr.inactiveMoved = max(cacheErr.inactiveMoved, this.max_abs_diff(uCurr(:,~mrfMask{kc}), uBefore(:,~mrfMask{kc})));
                end
                end     % colour classes
                if isMRF && ~isComponent; isAccepted = isAccSweep; end

                % 3. Gibbs block: exact conditional draws of (mu, Sigma), then refresh the log-prior cache
                if isMix && isGibbs
                    % mixture (Phase 7, partially collapsed Gibbs): the MH block above moved u given theta with
                    % z collapsed; now z | u, theta (exact, all voxels), per-group NIW | u, z, pi | z, then the
                    % collapsed log-prior cache. Fixed mode: no z at all (u updates do not need it)
                    uH          = uCurr(hIdx,:);
                    zCurr       = this.mixture_zstep(this.mixture_logweights(uH, mixD), rand(1, Nv, 'double', 'gpuArray'));
                    [nk, ubarK, SK] = this.hyper_suffstats_groups(uH, zCurr, Kmix);
                    for kk = 1:Kmix
                        [mu(:,kk), Sigma(:,:,kk)] = this.gibbs_hyper(ubarK(:,kk), SK(:,:,kk), nk(kk), mu(:,kk), Sigma(:,:,kk), hp);
                    end
                    piW         = this.draw_dirichlet(hier.alpha + nk);
                    [muG, PG, mixD] = this.mixture_to_gpu(mu, Sigma, piW);
                    lpC         = mixD.cG;
                    lpCurr      = this.logprior_mix(uH, lpC, muG, PG);
                elseif isMix
                    % fixed mode: nothing to update
                elseif isGibbs
                    [ubar, S]   = this.hyper_suffstats(uCurr(hIdx,:));
                    [mu, Sigma] = this.gibbs_hyper(ubar, S, Nv, mu, Sigma, hp);
                    [muG, PG]   = this.prior_to_gpu(mu, Sigma);
                    lpCurr      = this.logprior_normal(uCurr(hIdx,:), muG, PG);
                end

                % test only: compare the caches with a fresh computation of the current state
                if fitting.checkCache
                    if hasJac; xNow = xCurr; else; xNow = uCurr; end
                    if isMarginal; [lFresh, sFresh] = loglik(xNow); else; lFresh = loglik(xNow); end
                    cacheErr.loglik = max(cacheErr.loglik, this.max_abs_diff(lFresh, logLCurr));
                    if isMarginal; cacheErr.loglik = max(cacheErr.loglik, this.max_abs_diff(sFresh, statsCurr)); end
                    if isHier
                        cacheErr.logprior = max(cacheErr.logprior, this.max_abs_diff(this.logprior_mix(uCurr(hIdx,:), lpC, muG, PG), lpCurr));
                    end
                    if hasJac
                        jFresh = this.transform_logjac(uCurr, method, lb, ub);
                        if dropJac; jFresh = jFresh .* jacRow; end
                        cacheErr.logjac = max(cacheErr.logjac, this.max_abs_diff(jFresh, logJCurr));
                    end
                    cacheErr.Ncheck = cacheErr.Ncheck + 1;
                end

                % 4. acceptance bookkeeping and adaptation (burn-in only, frozen afterwards)
                if k <= Nburnin
                    if isAdapt
                        accWin = accWin + isAccepted;
                        % adaptive covariance: accumulate the state after this iteration (after the warm-up)
                        if isACov && k > acWarm
                            [acMean, acM2, acN] = this.welford_update(acMean, acM2, acN, uCurr);
                            acAccN  = acAccN + isAccepted;
                        end
                        if mod(k, fitting.adaptInterval) == 0
                            jAdapt  = jAdapt + 1;
                            delta   = this.adapt_gain(jAdapt) .* (accWin./fitting.adaptInterval - fitting.adaptTarget);
                            if isACov && acPhase
                                % covariance phase: Robbins-Monro on log lambda (one scalar per voxel)
                                acLam   = acLam .* exp(delta);
                            else
                            % joint: delta is [1,Nv] and scales all parameters of a voxel together
                            sigma   = sigma .* exp(delta);
                            if hasJac; sigma(~isLinear,:) = min(sigma(~isLinear,:), uWidth(~isLinear)); end
                            end
                            accWin(:) = 0;
                            % adaptive covariance: switch (first time) and refresh C_i, L_i
                            if isACov && k >= kSwitch
                                if ~acPhase
                                    % previous factor at the switch: the diagonal step, lambda = 2.38/sqrt(d)
                                    acL     = eye(Nvar, 'like', uCurr) .* reshape(sigma ./ acScale, 1, Nvar, Nv);
                                    acLam   = acScale .* ones(1, Nv, 'like', uCurr);
                                    acPhase = true;
                                end
                                [acL, acValid] = this.acov_refresh(acM2, acN, acAccN, acL, acValid, acEps, acTiny);
                            end
                            if isACov && acPhase
                                acLprop = reshape(acLam, 1, 1, Nv) .* acL;
                            end
                        end
                    end
                else
                    accPost = accPost + isAccepted;
                end

                % 5. Maintain the independence between iterations
                % 5.1 discard the first burnin*100% iterations
                % 5.2 keep an iteration every N iterations
                if ( k > Nburnin ) && mod(k-Nburnin+1, fitting.thinning) == 0
                    counter = counter+1;
                    if hasJac; xPosterior(:,:,counter,ii) = gather(xCurr); else; xPosterior(:,:,counter,ii) = gather(uCurr); end
                    if isMarginal; statsPost(:,:,counter,ii) = gather(statsCurr); end
                    if isMix
                        % ordered groups (label switching); Rao-Blackwellised membership P(z_i = k | u_i, theta)
                        % of this kept state, rows in the ordered labelling
                        ord = this.mixture_order(mu, isGibbs);
                        if isGibbs
                            muPost(:,:,counter,ii) = mu(:,ord); SigmaPost(:,:,:,counter,ii) = Sigma(:,:,ord); piPost(:,counter,ii) = piW(ord);
                        end
                        Rk       = this.mixture_responsibilities(uCurr(hIdx,:), mixD);
                        memCount = memCount + Rk(ord,:);
                    elseif isGibbs; muPost(:,counter,ii) = mu; SigmaPost(:,:,counter,ii) = Sigma; end
                end

                % display message at 1000 iteration and every 10000 iteration
                if mod(k,fitting.iteration/50) == 0 || k == min(1e3, fitting.iteration/100)
                    ET  = duration(0,0,toc(start),'Format','hh:mm:ss');
                    ERT = ET / (k/fitting.iteration) - ET;
                    fprintf('Iteration #%6d,    Elapsed time (hh:mm:ss):%s,     Estimated remaining time (hh:mm:ss):%s \n',k,string(ET),string(ERT));
                end
            end

            acceptance(:,:,ii)  = gather(accPost.' ./ (fitting.iteration - Nburnin));
            stepSize(:,:,ii)    = gather(sigma.');
            if isACov
                % final (frozen) proposal covariance lambda^2 L L' (the diagonal one if never switched)
                if acPhase
                    Pc = gather(pagemtimes(acLprop, 'none', acLprop, 'transpose'));
                else
                    Pc = gather(eye(Nvar, 'like', uCurr) .* reshape(sigma.^2, 1, Nvar, Nv));
                    acLam = ones(1, Nv, 'like', uCurr);
                end
                acPropCov(:,:,:,ii) = permute(Pc, [3 1 2]);
                dPc                 = reshape(Pc, Nvar*Nvar, Nv);
                stepSize(:,:,ii)    = sqrt(dPc(1:Nvar+1:end, :)).';
                sv                  = pagesvd(double(Pc));                  % [d,1,Nv], descending
                acCond(:,ii)        = reshape(sv(1,1,:) ./ sv(end,1,:), Nv, 1);
                acLambda(:,ii)      = gather(acLam(:));
                acValidOut(:,ii)    = gather(acValid(:));
            end
            if isMix
                ord = this.mixture_order(mu, isGibbs);
                if isGibbs; muLast(:,:,ii) = mu(:,ord); SigmaLast(:,:,:,ii) = Sigma(:,:,ord); piLast(:,ii) = piW(ord); end
                Rk      = this.mixture_responsibilities(uCurr(hIdx,:), mixD);
                memLast = memLast + Rk(ord,:);
            elseif isHier && isGibbs; muLast(:,ii) = mu; SigmaLast(:,:,ii) = Sigma; end
            end

            % convert final posterior distribution into structure
            xPosterior = this.array2struct(xPosterior,fitting.modelParams);
            for kvar = 1:Nvar; xPosterior.(fitting.modelParams{kvar}) = shiftdim(xPosterior.(fitting.modelParams{kvar}),1); end

            % nuisance recovery: exact conditional draws of sigma (and S0) for every retained sample,
            % from the cached statistics (no extra forward evaluation); output in full modelParams order
            if isMarginal
                xPosterior = this.recover_nuisance(xPosterior, statsPost, lik, mNZ);
            end

            % diagnostics and resolved settings
            diagnostics.acceptance  = acceptance;
            if isComponent; diagnostics.acceptanceBlocks = fitting.modelParams(:).'; else; diagnostics.acceptanceBlocks = {'joint'}; end
            diagnostics.stepSize    = stepSize;
            diagnostics.sampledParams   = fitting.modelParams(:).';
            diagnostics.recoveredParams = lik.recoveredParams;
            if fitting.checkCache; diagnostics.cacheCheck = cacheErr; end
            if isACov
                diagnostics.adaptCovariance = struct('params', {fitting.modelParams(:).'}, ...
                    'proposalCov', acPropCov, 'lambda', acLambda, 'condition', acCond, 'valid', acValidOut, ...
                    'switchIteration', kSwitch);
                acRule = mcmc.acov_rule(acWarm, acMinN, kSwitch, acScale);
            else
                acRule = [];
            end
            if isHier
                hyper = struct('params', {hier.params}, ...
                               'transform', {this.transform_description(method(hIdx), fitting.lb(hIdx), fitting.ub(hIdx))}, ...
                               'hyperprior', hier.hyperprior, 'fixed', hier.fixed);
                if isMix
                    % mixture (Phase 7): ordered samples, membership frequency [Nv,K] over the kept samples
                    % of all repetitions (final z if there is none)
                    hyper.K         = Kmix;
                    hyper.alpha     = hier.alpha;
                    hyper.ordering  = this.mixture_ordering_rule(isGibbs);
                    if isGibbs
                        hyper.muPost    = muPost;
                        hyper.SigmaPost = SigmaPost;
                        hyper.piPost    = piPost;
                        hyper.muLast    = muLast;
                        hyper.SigmaLast = SigmaLast;
                        hyper.piLast    = piLast;
                    else
                        hyper.mu        = hier.mu;
                        hyper.Sigma     = hier.Sigma;
                        hyper.pi        = hier.pi;
                    end
                    if Ns > 0
                        hyper.membership = gather(memCount.' ./ (Ns * fitting.repetition));
                    else
                        hyper.membership = gather(memLast.' ./ fitting.repetition);
                    end
                elseif isGibbs
                    hyper.muPost    = muPost;
                    hyper.SigmaPost = SigmaPost;
                    hyper.muLast    = muLast;
                    hyper.SigmaLast = SigmaLast;
                else
                    hyper.mu        = hier.mu;
                    hyper.Sigma     = hier.Sigma;
                end
                diagnostics.hyper = hyper;
                priorSettings = this.prior_settings(hier, hp);
            else
                diagnostics.hyper = [];
                priorSettings = [];
            end
            if isMRF
                mrfSettings = this.mrf_settings(mrf, hier, Knb, NcolNominal, Ncol, Nedges, sfInfo);
                priorSettings.mrf = mrfSettings;
            else
                mrfSettings = [];
            end
            diagnostics.settings    = struct( ...
                'parameterTransform',   {method}, ...
                'updateScheme',         lower(fitting.updateScheme), ...
                'adaptStepSize',        isAdapt, ...
                'adaptInterval',        fitting.adaptInterval, ...
                'adaptTarget',          fitting.adaptTarget, ...
                'adaptRule',            'log(sigma) += 2*j^(-0.6)*(acc_j - adaptTarget), every adaptInterval iterations within burn-in', ...
                'Nadapt',               Nadapt*isAdapt, ...
                'adaptCovariance',      isACov, ...
                'adaptCovarianceRule',  acRule, ...
                'Nburnin',              Nburnin, ...
                'overdisp',             fitting.overdisp, ...
                'overdispRule',         'u0 + overdisp*W.*randn for repetition > 1, W = T(ub)-T(lb), or 100 x initial u-step if unbounded', ...
                'stepSizeInit',         'xStepSize ./ |dx/du| at the start point (u space)', ...
                'forceNewPath',         logical(fitting.forceNewPath), ...
                'likelihood',           lik.name, ...
                'S0Param',              lik.S0Param, ...
                'sampledParams',        {fitting.modelParams(:).'}, ...
                'droppedParams',        {lik.droppedParams}, ...
                'fixedParams',          lik.userFixed, ...
                'nuisance',             lik.nuisance, ...
                'prior',                priorSettings, ...
                'mrf',                  mrfSettings, ...
                'rngState',             rngState, ...
                'gpuRngState',          gpuRngState, ...
                'geom',                 geom);

            disp('The Metropolis-Hastings MCMC sampling (mcmc_bayes) is completed.')

        end

    end

    methods(Static)

        % check whether all new options are absent or at their legacy defaults
        function [tf, nonDefault] = isLegacy(fitting)
        % Output
        % ------
        % tf            : true if no new option is set to a non-default value
        % nonDefault    : cell array of names of the non-default option(s)
        %
            % option name, legacy default (the first 7 rows are mcmc.sampler_infra_defaults)
            defaults = { 'parameterTransform',  'linear';
                         'likelihood',          'gaussian';
                         'S0Param',             '';
                         'updateScheme',        'joint';
                         'adaptStepSize',       false;
                         'adaptInterval',       50;
                         'adaptTarget',         [];
                         'adaptCovariance',     false;
                         'overdisp',            0;
                         'prior',               [];
                         'fixedParams',         []};

            nonDefault = mcmc.nondefault_options(fitting, defaults);
            tf = isempty(nonDefault);
        end

        % check and set default fitting algorithm parameters for the new path
        function fitting2 = check_set_default_bayes(fitting)
        % Input
        % -----
        % fitting       : structure contains fitting algorithm parameters, see class header
        %
            % sampler options shared with mcmc (parameterTransform, updateScheme, adaptStepSize,
            % adaptInterval, adaptTarget, adaptCovariance, overdisp): defaults and validation
            fitting2 = mcmc.check_set_default_infra(mcmc.check_set_default_basic(fitting));

            if ~isfield(fitting,'likelihood');          fitting2.likelihood         = 'gaussian';    end
            if ~isfield(fitting,'S0Param');             fitting2.S0Param            = '';            end
            if ~isfield(fitting,'prior');               fitting2.prior              = [];            end
            if ~isfield(fitting,'forceNewPath');        fitting2.forceNewPath       = false;         end
            if ~isfield(fitting,'fixedParams');         fitting2.fixedParams        = [];            end
            if ~isfield(fitting,'checkCache');          fitting2.checkCache         = false;         end
            if ~isfield(fitting,'mrfUpdate');           fitting2.mrfUpdate          = 'chromatic';   end
        end

        % display the new sampler settings
        function display_bayes_algorithm_parameters(fitting)
            mcmc.display_infra_algorithm_parameters(fitting);
            disp(['Likelihood        : ', char(fitting.likelihood)]);
            if ~isempty(fitting.S0Param); disp(['S0 parameter      : ', char(fitting.S0Param), ' (marginalised)']); end
            if isstruct(fitting.prior) && isfield(fitting.prior,'hierarchical') && ~isempty(fitting.prior.hierarchical) && ...
                    ~(islogical(fitting.prior.hierarchical) && ~fitting.prior.hierarchical)
                h = fitting.prior.hierarchical;
                if isstruct(h) && isfield(h,'K') && ~isempty(h.K) && isnumeric(h.K) && isscalar(h.K) && h.K > 1
                    disp(['Prior             : hierarchical Gaussian mixture on u, K = ', num2str(h.K), ...
                          ', fixed = ', num2str(isfield(h,'fixed') && ~isempty(h.fixed) && logical(h.fixed))]);
                elseif isstruct(h) && isfield(h,'fixed') && ~isempty(h.fixed) && h.fixed
                    disp( 'Prior             : hierarchical Normal on u, fixed mu/Sigma');
                elseif isstruct(h) && isfield(h,'hyperprior') && ~isempty(h.hyperprior)
                    disp(['Prior             : hierarchical Normal on u, hyperprior ', char(h.hyperprior)]);
                else
                    disp( 'Prior             : hierarchical Normal on u, hyperprior niw');
                end
            end
            if isstruct(fitting.prior) && isfield(fitting.prior,'mrf') && ~isempty(fitting.prior.mrf) && ...
                    ~(islogical(fitting.prior.mrf) && ~fitting.prior.mrf)
                m = fitting.prior.mrf;
                if ~isstruct(m); m = struct(); end
                disp(['MRF prior         : potential ', char(field_or_default(m,'potential','l1')), ...
                      ', tau ', num2str(field_or_default(m,'tau',1)), ', mode ', char(field_or_default(m,'mode','3d')), ...
                      ', radius ', num2str(field_or_default(m,'radius',1)), ', update ', char(fitting.mrfUpdate)]);
            end
        end

        %% marginal likelihoods (Phase 2), see the derivation in the class header
        % resolve the likelihood and the sampled/marginalised parameter sets
        function [fittingS, lik] = setup_likelihood(fitting)
        % Input
        % -----
        % fitting   : fitting structure with the FULL modelParams/lb/ub/xStepSize
        %             (and per-parameter parameterTransform), as the model defines them
        % Output
        % ------
        % fittingS  : same as fitting, with modelParams/lb/ub/xStepSize/parameterTransform
        %             restricted to the sampled parameters
        % lik       : structure
        %   .name           : canonical likelihood name
        %   .isMarginal     : true for the marginal likelihoods
        %   .S0Param        : amplitude parameter name ('' if none)
        %   .droppedParams  : parameters removed from the sampled set
        %   .fixedParams    : fields injected (value 1) before each FWDfunc call
        %   .userFixed      : structure, test-only fitting.fixedParams (removed from the sampled
        %                     set and from the output, value injected before each likelihood call)
        %   .recoveredParams: fields restored by post-hoc conditional draws
        %   .shapeOffset    : InvGamma shape is (m - shapeOffset)/2
        %   .Nstat          : # cached statistics per voxel ([R] or [RSS; Shat; g'Wg])
        %   .fittingOut     : fitting with the full output parameter list (for res2out)
        %   .nuisance       : description for out.settings ([] for 'gaussian')
        %
            valid = {'gaussian','marginal_noise','marginal_S0noise','marginal_S0noise_flat'};
            if ~isfield(fitting,'likelihood') || isempty(fitting.likelihood); fitting.likelihood = 'gaussian'; end
            if ~isfield(fitting,'S0Param'); fitting.S0Param = ''; end
            if ~(ischar(fitting.likelihood) || (isstring(fitting.likelihood) && isscalar(fitting.likelihood))) || ~any(strcmpi(fitting.likelihood, valid))
                error('mcmc_bayes:invalidLikelihood', 'mcmc_bayes: fitting.likelihood must be one of: %s.', strjoin(valid, ', '));
            end
            name    = valid{strcmpi(fitting.likelihood, valid)};
            S0Param = char(fitting.S0Param);

            if ~isfield(fitting,'modelParams') || isempty(fitting.modelParams)
                error('mcmc_bayes:noModelParams', 'mcmc_bayes: fitting.modelParams is required.');
            end
            params  = cellstr(fitting.modelParams);
            params  = params(:).';
            Nfull   = numel(params);
            if ~isfield(fitting,'xStepSize') || numel(fitting.xStepSize) ~= Nfull
                error('mcmc_bayes:xStepSize', 'mcmc_bayes: fitting.xStepSize must have one entry per modelParams (%d).', Nfull);
            end
            if numel(fitting.lb) ~= Nfull || numel(fitting.ub) ~= Nfull
                error('mcmc_bayes:invalidBounds', 'mcmc_bayes: fitting.lb and fitting.ub must have one entry per modelParams (%d).', Nfull);
            end

            isNoise = strcmp(params, 'noise');
            isS0    = false(1, Nfull);
            switch name
                case 'gaussian'
                    if ~isempty(S0Param)
                        error('mcmc_bayes:S0Param', 'mcmc_bayes: fitting.S0Param is only used with likelihood ''marginal_S0noise'' or ''marginal_S0noise_flat''.');
                    end
                    if ~any(isNoise)
                        error('mcmc_bayes:noNoise', 'mcmc_bayes: likelihood ''gaussian'' requires ''noise'' in fitting.modelParams.');
                    end
                case 'marginal_noise'
                    if ~isempty(S0Param)
                        error('mcmc_bayes:S0Param', 'mcmc_bayes: fitting.S0Param is only used with likelihood ''marginal_S0noise'' or ''marginal_S0noise_flat'' (got likelihood ''marginal_noise'').');
                    end
                otherwise   % marginal_S0noise(_flat)
                    if isempty(S0Param)
                        error('mcmc_bayes:S0Param', 'mcmc_bayes: likelihood ''%s'' requires fitting.S0Param (the amplitude parameter in modelParams).', name);
                    end
                    isS0 = strcmp(params, S0Param);
                    if ~any(isS0) || strcmp(S0Param,'noise')
                        error('mcmc_bayes:S0Param', 'mcmc_bayes: fitting.S0Param ''%s'' is not a model parameter (modelParams: %s).', S0Param, strjoin(params, ', '));
                    end
            end

            isMarginal  = ~strcmp(name, 'gaussian');
            isDrop      = isMarginal & (isNoise | isS0);

            % test-only fixed parameters (removed from the sampled set, value injected)
            userFixed   = struct();
            if isfield(fitting,'fixedParams') && ~isempty(fitting.fixedParams)
                userFixed = fitting.fixedParams;
                if ~isstruct(userFixed) || ~isscalar(userFixed)
                    error('mcmc_bayes:fixedParams', 'mcmc_bayes: fitting.fixedParams must be a scalar structure of scalar values.');
                end
                fnF = fieldnames(userFixed);
                for k = 1:numel(fnF)
                    v = userFixed.(fnF{k});
                    if ~any(strcmp(params, fnF{k})) || any(strcmp(params(isDrop), fnF{k}))
                        error('mcmc_bayes:fixedParams', 'mcmc_bayes: fitting.fixedParams.%s is not a sampled model parameter.', fnF{k});
                    end
                    if ~(isnumeric(v) && isscalar(v) && isfinite(v))
                        error('mcmc_bayes:fixedParams', 'mcmc_bayes: fitting.fixedParams.%s must be a finite scalar.', fnF{k});
                    end
                end
            end
            isUserFixed = ismember(params, fieldnames(userFixed));
            isSampled   = ~isDrop & ~isUserFixed;
            if ~any(isSampled)
                error('mcmc_bayes:noSampledParams', 'mcmc_bayes: no parameter left to sample after removing %s.', strjoin(params(~isSampled), ', '));
            end

            % sampled subset
            fittingS                = fitting;
            fittingS.modelParams    = reshape(params(isSampled), [], 1);
            fittingS.lb             = reshape(fitting.lb(isSampled), [], 1);
            fittingS.ub             = reshape(fitting.ub(isSampled), [], 1);
            fittingS.xStepSize      = reshape(fitting.xStepSize(isSampled), [], 1);
            if isfield(fitting,'parameterTransform') && (iscell(fitting.parameterTransform) || (isstring(fitting.parameterTransform) && ~isscalar(fitting.parameterTransform)))
                if numel(fitting.parameterTransform) ~= Nfull
                    error('mcmc_bayes:invalidTransform', ...
                        'mcmc_bayes: per-parameter parameterTransform must have one entry per modelParams (%d, including any marginalised parameter).', Nfull);
                end
                fittingS.parameterTransform = fitting.parameterTransform(isSampled);
            end

            % output parameter list: full modelParams order (without test-only fixed
            % parameters), 'noise' appended if the model has none
            outParams = params(~isUserFixed); outLb = fitting.lb(~isUserFixed); outUb = fitting.ub(~isUserFixed);
            outLb = outLb(:).'; outUb = outUb(:).';
            if isMarginal && ~any(isNoise)
                outParams{end+1} = 'noise'; outLb(end+1) = 0; outUb(end+1) = Inf;
            end
            fittingOut              = fitting;
            fittingOut.modelParams  = reshape(outParams, [], 1);
            fittingOut.lb           = outLb(:);
            fittingOut.ub           = outUb(:);

            lik.name            = name;
            lik.isMarginal      = isMarginal;
            lik.S0Param         = S0Param;
            lik.droppedParams   = params(isDrop);
            lik.userFixed       = userFixed;
            lik.fittingOut      = fittingOut;
            lik.shapeOffset     = double(strcmp(name, 'marginal_S0noise_flat'));
            switch name
                case 'gaussian'
                    lik.fixedParams = {}; lik.recoveredParams = {}; lik.Nstat = 0;
                case 'marginal_noise'
                    lik.fixedParams = {'noise'}; lik.recoveredParams = {'noise'}; lik.Nstat = 1;
                otherwise
                    lik.fixedParams = {S0Param, 'noise'}; lik.recoveredParams = {'noise', S0Param}; lik.Nstat = 3;
            end
            if isMarginal
                switch name
                    case 'marginal_noise'
                        logLForm = '-(m/2) log(r''Wr), r = y - g';
                        rule     = 'sigma^2 | u,y ~ InvGamma(m/2, R/2), R = r''Wr';
                    case 'marginal_S0noise'
                        logLForm = '-(m/2) log(RSS), RSS = y''Wy - (y''Wg)^2/(g''Wg)  (Zellner g-prior on S0, broad limit, + 1/sigma^2)';
                        rule     = 'sigma^2 | u,y ~ InvGamma(m/2, RSS/2); S0 | sigma^2,u,y ~ N(y''Wg/g''Wg, sigma^2/g''Wg)';
                    case 'marginal_S0noise_flat'
                        logLForm = '-(1/2) log(g''Wg) - ((m-1)/2) log(RSS)  (flat S0 on R, + 1/sigma^2)';
                        rule     = 'sigma^2 | u,y ~ InvGamma((m-1)/2, RSS/2); S0 | sigma^2,u,y ~ N(y''Wg/g''Wg, sigma^2/g''Wg)';
                end
                lik.nuisance = struct( ...
                    'fields',       {lik.recoveredParams}, ...
                    'method',       'post-hoc exact conditional draw for every retained sample (not sampled by MCMC)', ...
                    'logLikelihood',logLForm, ...
                    'conditionals', rule, ...
                    'noiseIs',      'sigma (not sigma^2)', ...
                    'statistics',   'cached at the retained iterations inside the loop (no extra forward evaluation)');
            else
                lik.nuisance = [];
            end
        end

        % set the given fields of a parameter structure to a constant array
        function x_struct = inject_fixed(x_struct, names, val)
            for k = 1:numel(names)
                x_struct.(names{k}) = val;
            end
        end

        % copy all fields of vals into a parameter structure
        function x_struct = inject_values(x_struct, vals)
            fn = fieldnames(vals);
            for k = 1:numel(fn)
                x_struct.(fn{k}) = vals.(fn{k});
            end
        end

        % marginal log-likelihood (theta-dependent part) and cached statistics
        function [logL, stats] = loglik_marginal(g, y, weights, name, rssFloor, m)
        % Input
        % -----
        % g         : forward model without amplitude, [Nm, Nv]
        % y         : measurements, [Nm, Nv]
        % weights   : weights (W diagonal), [Nm, Nv]
        % name      : 'marginal_noise' | 'marginal_S0noise' | 'marginal_S0noise_flat'
        % rssFloor  : [1,Nv] (or scalar), R/RSS at or below this value is rejected
        % m         : (optional) # measurements with non-zero weight, scalar or [1,Nv];
        %             default sum(weights ~= 0, 1)
        % Output
        % ------
        % logL      : [1, Nv], -Inf for degenerate states
        % stats     : [1, Nv] R ('marginal_noise') or [3, Nv] [RSS; Shat; g'Wg]
        %
            if nargin < 6 || isempty(m); m = sum(weights ~= 0, 1); end
            Nm      = m;
            gFloor  = double(realmin('single'))/double(eps('single'));    % double, so CPU double inputs stay double
            switch name
                case 'marginal_noise'
                    R       = sum(weights.*(y - g).^2, 1);
                    logL    = mcmc_bayes.marginal_apply(R, 1, Nm/2, 0, rssFloor, 0);
                    stats   = R;
                otherwise
                    gw      = weights.*g;
                    gWg     = sum(gw.*g, 1);
                    Shat    = sum(gw.*y, 1) ./ gWg;
                    % residual form, avoids the y'Wy - (y'Wg)^2/g'Wg cancellation
                    RSS     = sum(weights.*(y - Shat.*g).^2, 1);
                    if strcmp(name, 'marginal_S0noise_flat')
                        logL = mcmc_bayes.marginal_apply(RSS, gWg, (Nm-1)/2, 0.5, rssFloor, gFloor);
                    else
                        logL = mcmc_bayes.marginal_apply(RSS, gWg, Nm/2, 0, rssFloor, gFloor);
                    end
                    stats   = [RSS; Shat; gWg];
            end
        end

        % logL = -a*log(R) - cG*log(gWg), -Inf for degenerate states: one fused
        % kernel on the GPU (marginal_kernel), the same maths vectorised on the CPU
        % (a = m/2 or (m-1)/2, scalar or [1,Nv])
        function logL = marginal_apply(R, gWg, a, cG, rssFloor, gFloor)
            if isa(R, 'gpuArray')
                logL = arrayfun(@marginal_kernel, R, gWg, a, cG, rssFloor, gFloor);
            else
                ok      = (R > rssFloor) & (gWg > gFloor) & (gWg < Inf);
                logL    = -a.*log(max(R, rssFloor)) - cG.*log(min(max(gWg, gFloor), 1/gFloor));
                logL    = logL .* ones(size(ok), 'like', logL);
                logL(~ok) = -Inf;
            end
        end

        % post-hoc exact conditional draws of sigma (and S0) for every retained sample
        function xPosterior = recover_nuisance(xPosterior, statsPost, lik, Nm)
        % Input
        % -----
        % xPosterior: structure, sampled parameters, each [Nv, Ns, Nrep]
        % statsPost : cached statistics at the retained samples, [Nstat, Nv, Ns, Nrep]
        % lik       : see setup_likelihood
        % Nm        : # measurements with non-zero weight, scalar or [1,Nv]
        % Output
        % ------
        % xPosterior: structure with the recovered fields added ('noise' = sigma, S0Param),
        %             fields in lik.fittingOut.modelParams order
        %
            a       = (Nm - lik.shapeOffset)/2;             % InvGamma shape
            sz      = size(statsPost, 2:4);
            if ~isscalar(a)
                % per-voxel shape, expanded to the [Nv*Ns*Nrep] element order (voxel fastest)
                a = repmat(double(a(:)).', 1, prod(sz(2:end)));
            end
            stats   = reshape(statsPost, size(statsPost,1), []);
            Nel     = size(stats, 2);
            noise   = zeros(1, Nel, 'single');
            if lik.Nstat == 3; S0 = zeros(1, Nel, 'single'); end

            chunk = 2^24;   % elements per GPU chunk
            for kc = 1:chunk:Nel
                idx     = kc:min(kc+chunk-1, Nel);
                st      = gpuArray(stats(:, idx));
                % sigma^2 = (R/2)/G, G ~ Gamma(a,1)
                if isscalar(a)
                    sigma2  = (st(1,:)./2) ./ randg(a, [1 numel(idx)], 'like', st);
                else
                    sigma2  = (st(1,:)./2) ./ randg(gpuArray(single(a(idx))));
                end
                noise(idx) = gather(sqrt(sigma2));
                if lik.Nstat == 3
                    S0(idx) = gather(st(2,:) + sqrt(sigma2./st(3,:)) .* randn(1, numel(idx), 'like', st));
                end
            end

            recovered.noise = reshape(noise, sz);
            if lik.Nstat == 3; recovered.(lik.S0Param) = reshape(S0, sz); end

            % full output order
            outParams = lik.fittingOut.modelParams;
            xOut = struct();
            for k = 1:numel(outParams)
                p = outParams{k};
                if isfield(recovered, p); xOut.(p) = recovered.(p); else; xOut.(p) = xPosterior.(p); end
            end
            xPosterior = xOut;
        end

        %% hierarchical Normal prior (Phase 3), see the derivation in the class header
        % resolve and validate fitting.prior.hierarchical for the SAMPLED parameters
        function hier = setup_hierarchical(fitting)
        % Input
        % -----
        % fitting   : fitting structure with the sampled parameter set (setup_likelihood output)
        % Output
        % ------
        % hier      : structure
        %   .on             : true if the hierarchical prior is used
        %   .params, .idx   : hierarchical parameters and their rows in fitting.modelParams (model order)
        %   .d              : # hierarchical parameters
        %   .hyperprior     : 'niw' | 'jeffreys_half'
        %   .m0, .kappa0, .Psi0, .nu0 : NIW hyperprior ([] -> resolved at the start of sampling)
        %   .fixed, .mu, .Sigma       : fixed mode and its hyperparameters (mu [d,1], Sigma [d,d];
        %                               K > 1: mu [d,K], Sigma [d,d,K])
        %   .subsetFraction, .maxGPUMemory
        %   .K, .alpha, .init, .pi    : mixture (Phase 7): # groups, Dirichlet concentration, free-mode
        %                               initialisation, fixed group weights [K,1] (K > 1 only, else [])
        %
            hier = struct('on', false, 'params', {{}}, 'idx', [], 'd', 0, 'hyperprior', '', ...
                          'm0', [], 'kappa0', [], 'Psi0', [], 'nu0', [], 'fixed', false, 'mu', [], 'Sigma', [], ...
                          'subsetFraction', 1, 'maxGPUMemory', [], 'K', 1, 'alpha', 1, 'init', 'kmeans', 'pi', []);
            if ~isfield(fitting,'prior') || isempty(fitting.prior); return; end
            prior = fitting.prior;
            if ~isstruct(prior) || ~isscalar(prior)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: fitting.prior must be a structure with the field(s) hierarchical and/or mrf.');
            end
            bad = setdiff(fieldnames(prior), {'hierarchical','mrf'});
            if ~isempty(bad)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: unknown field(s) in fitting.prior: %s (valid: hierarchical, mrf).', strjoin(bad, ', '));
            end
            if ~isfield(prior,'hierarchical') || isempty(prior.hierarchical); return; end
            h = prior.hierarchical;
            if islogical(h) && isscalar(h)
                if ~h; return; end
                h = struct();
            end
            if ~isstruct(h) || ~isscalar(h)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: fitting.prior.hierarchical must be a structure (or true).');
            end
            valid = {'hyperprior','m0','kappa0','Psi0','nu0','fixed','mu','Sigma','params','subsetFraction','maxGPUMemory', ...
                     'K','alpha','init','pi'};
            bad   = setdiff(fieldnames(h), valid);
            if ~isempty(bad)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: unknown field(s) in fitting.prior.hierarchical: %s (valid: %s).', ...
                    strjoin(bad, ', '), strjoin(valid, ', '));
            end

            % parameters under the hierarchy (default: all sampled parameters except 'noise')
            params  = cellstr(fitting.modelParams); params = params(:).';
            hp      = field_or_default(h, 'params', []);
            if isempty(hp)
                hp = params(~strcmp(params,'noise'));
                if isempty(hp)
                    error('mcmc_bayes:hierarchicalParams', 'mcmc_bayes: no sampled parameter left for the hierarchical prior.');
                end
            else
                hp = cellstr(hp); hp = hp(:).';
                if numel(unique(hp)) ~= numel(hp)
                    error('mcmc_bayes:hierarchicalParams', 'mcmc_bayes: prior.hierarchical.params has duplicate entries.');
                end
                missing = hp(~ismember(hp, params));
                if ~isempty(missing)
                    error('mcmc_bayes:hierarchicalParams', ...
                        ['mcmc_bayes: prior.hierarchical.params: %s is not a sampled parameter (sampled: %s). ' ...
                         'Marginalised or fixed parameters cannot be under the hierarchy.'], strjoin(missing, ', '), strjoin(params, ', '));
                end
            end
            idx     = sort(find(ismember(params, hp)));     % model order
            d       = numel(idx);

            % transforms and bounds allowed without bound rejection
            if isfield(fitting,'parameterTransform') && ~isempty(fitting.parameterTransform); spec = fitting.parameterTransform; else; spec = 'linear'; end
            method  = mcmc_bayes.parse_transform(spec, numel(params));
            lb      = fitting.lb(:); ub = fitting.ub(:);
            for j = idx
                switch method{j}
                    case 'sigmoid'; ok = isfinite(lb(j)) && isfinite(ub(j)) && ub(j) > lb(j);
                    case 'log';     ok = lb(j) == 0 && ub(j) == Inf;
                    otherwise;      ok = lb(j) == -Inf && ub(j) == Inf;
                end
                if ~ok
                    error('mcmc_bayes:hierarchicalTransform', ...
                        ['mcmc_bayes: hierarchical parameter %s has transform ''%s'' with bounds [%g, %g]. The Normal prior lives in u space ' ...
                         'with no bound rejection, so the allowed combinations are ''sigmoid'' with finite lb < ub, ''log'' with lb = 0 and ' ...
                         'ub = Inf, and ''linear'' with lb = -Inf and ub = Inf.'], params{j}, method{j}, lb(j), ub(j));
                end
            end

            % hyperprior
            hyperprior = lower(char(field_or_default(h, 'hyperprior', 'niw')));
            if ~any(strcmp(hyperprior, {'niw','jeffreys_half'}))
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: prior.hierarchical.hyperprior must be ''niw'' or ''jeffreys_half'' (got ''%s'').', hyperprior);
            end
            kappa0  = field_or_default(h, 'kappa0', 1e-3);
            if ~(isnumeric(kappa0) && isscalar(kappa0) && isfinite(kappa0) && kappa0 > 0)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: prior.hierarchical.kappa0 must be a positive scalar.');
            end
            nu0     = field_or_default(h, 'nu0', []);
            if ~isempty(nu0) && ~(isnumeric(nu0) && isscalar(nu0) && isfinite(nu0) && nu0 > d - 1)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: prior.hierarchical.nu0 must be a scalar > d - 1 = %d.', d - 1);
            end
            Psi0    = field_or_default(h, 'Psi0', []);
            if ~isempty(Psi0)
                Psi0 = mcmc_bayes.check_spd(double(Psi0), d, 'mcmc_bayes:invalidPrior', 'prior.hierarchical.Psi0');
            end
            m0      = field_or_default(h, 'm0', []);
            if ~isempty(m0)
                if ~(isnumeric(m0) && numel(m0) == d && all(isfinite(m0)))
                    error('mcmc_bayes:invalidPrior', 'mcmc_bayes: prior.hierarchical.m0 must be a finite vector with %d entries.', d);
                end
                m0 = double(m0(:));
            end

            % mixture (Phase 7)
            K       = field_or_default(h, 'K', 1);
            if ~(isnumeric(K) && isscalar(K) && isfinite(K) && K >= 1 && K == round(K))
                error('mcmc_bayes:hierarchicalMixture', 'mcmc_bayes: prior.hierarchical.K must be a positive integer.');
            end
            K       = double(K);
            alpha   = field_or_default(h, 'alpha', 1);
            if ~(isnumeric(alpha) && isscalar(alpha) && isfinite(alpha) && alpha > 0)
                error('mcmc_bayes:hierarchicalMixture', 'mcmc_bayes: prior.hierarchical.alpha must be a positive finite scalar.');
            end
            init    = lower(char(field_or_default(h, 'init', 'kmeans')));
            if ~strcmp(init, 'kmeans')
                error('mcmc_bayes:hierarchicalMixture', 'mcmc_bayes: prior.hierarchical.init must be ''kmeans'' (got ''%s'').', init);
            end
            if K > 1 && strcmp(hyperprior, 'jeffreys_half')
                error('mcmc_bayes:hierarchicalMixture', ...
                    ['mcmc_bayes: hyperprior ''jeffreys_half'' is not supported with K > 1 (improper per group: an empty or ' ...
                     'small group has no proper conditional). Use ''niw''.']);
            end

            % fixed mode
            fixed   = logical(field_or_default(h, 'fixed', false));
            mu      = field_or_default(h, 'mu', []);
            Sigma   = field_or_default(h, 'Sigma', []);
            piFix   = field_or_default(h, 'pi', []);
            if ~isempty(piFix) && (~fixed || (K == 1 && ~isequal(double(piFix), 1)))
                error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: prior.hierarchical.pi is only used with fixed = true and K > 1.');
            end
            if fixed && K > 1
                if ~(isnumeric(mu) && isequal(size(mu), [d K]) && all(isfinite(mu(:))))
                    error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: fixed mode with K = %d needs prior.hierarchical.mu [d,K] = [%d,%d], finite (u space).', K, d, K);
                end
                mu      = double(mu);
                if ~(isnumeric(Sigma) && size(Sigma,1) == d && size(Sigma,2) == d && size(Sigma,3) == K && ndims(Sigma) <= 3)
                    error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: fixed mode with K = %d needs prior.hierarchical.Sigma [d,d,K] = [%d,%d,%d].', K, d, d, K);
                end
                SigmaK  = zeros(d, d, K);
                for k = 1:K
                    SigmaK(:,:,k) = mcmc_bayes.check_spd(double(Sigma(:,:,k)), d, 'mcmc_bayes:hierarchicalFixed', sprintf('prior.hierarchical.Sigma(:,:,%d)', k));
                end
                Sigma   = SigmaK;
                if ~(isnumeric(piFix) && numel(piFix) == K && all(isfinite(piFix(:))) && all(piFix(:) > 0) && abs(sum(piFix(:)) - 1) <= 1e-6)
                    error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: fixed mode with K = %d needs prior.hierarchical.pi [K,1], positive, summing to 1.', K);
                end
                piFix   = double(piFix(:)) ./ sum(double(piFix(:)));
            elseif fixed
                if ~(isnumeric(mu) && numel(mu) == d && all(isfinite(mu)))
                    error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: fixed mode needs prior.hierarchical.mu with %d finite entries (u space).', d);
                end
                mu      = double(mu(:));
                Sigma   = mcmc_bayes.check_spd(double(Sigma), d, 'mcmc_bayes:hierarchicalFixed', 'prior.hierarchical.Sigma');
            elseif ~isempty(mu) || ~isempty(Sigma)
                error('mcmc_bayes:hierarchicalFixed', 'mcmc_bayes: prior.hierarchical.mu/Sigma are only used with fixed = true.');
            end

            subsetFraction = field_or_default(h, 'subsetFraction', 1);
            if ~(isnumeric(subsetFraction) && isscalar(subsetFraction) && subsetFraction > 0 && subsetFraction <= 1)
                error('mcmc_bayes:subsetFraction', 'mcmc_bayes: prior.hierarchical.subsetFraction must be in (0,1].');
            end

            hier.on             = true;
            hier.params         = params(idx);
            hier.idx            = idx;
            hier.d              = d;
            hier.hyperprior     = hyperprior;
            hier.m0             = m0;
            hier.kappa0         = kappa0;
            hier.Psi0           = Psi0;
            hier.nu0            = nu0;
            hier.fixed          = fixed;
            hier.mu             = mu;
            hier.Sigma          = Sigma;
            hier.subsetFraction = subsetFraction;
            hier.maxGPUMemory   = field_or_default(h, 'maxGPUMemory', []);
            hier.K              = K;
            hier.alpha          = double(alpha);
            hier.init           = init;
            if K > 1; hier.pi = piFix; else; hier.pi = []; end
        end

        % symmetric positive definite d x d check (returns the symmetrised matrix)
        function A = check_spd(A, d, id, name)
            if ~(isnumeric(A) && isequal(size(A), [d d]) && all(isfinite(A(:))))
                error(id, 'mcmc_bayes: %s must be a finite %d x %d matrix.', name, d, d);
            end
            if max(abs(A - A.'), [], 'all') > 1e-10 * max(abs(A), [], 'all')
                error(id, 'mcmc_bayes: %s must be symmetric.', name);
            end
            A = (A + A.')/2;
            [~, flag] = chol(A);
            if flag
                error(id, 'mcmc_bayes: %s must be positive definite.', name);
            end
        end

        % resolve the NIW defaults from the starting u, see the class header
        function hp = resolve_hyperprior(hier, uStartH, floorVar)
        % uStartH   : [d, Nv] starting u of the hierarchical parameters (double)
        % floorVar  : [d, 1] variance floor, (10 x median initial u-step)^2
            hp          = hier;
            d           = hier.d;
            hp.floorVar = floorVar(:);
            hp.rules    = struct('m0','user','nu0','user','Psi0','user');
            if isempty(hp.m0);  hp.m0  = mean(uStartH, 2);  hp.rules.m0  = 'mean of the starting u'; end
            if isempty(hp.nu0); hp.nu0 = d + 2;             hp.rules.nu0 = 'd + 2'; end
            if isempty(hp.Psi0)
                v0          = max(var(uStartH, 0, 2), hp.floorVar);
                hp.Psi0     = max(hp.nu0 - d - 1, 1) .* diag(v0);
                hp.rules.Psi0 = 'max(nu0-d-1,1)*diag(max(var(u_start), (10*median initial u-step)^2))';
            end
        end

        % the hyperparameters on the GPU: mean [d,1] and precision [d,d], single
        function [muG, PG] = prior_to_gpu(mu, Sigma)
            d   = numel(mu);
            R   = chol(Sigma);
            Ri  = R \ eye(d);
            P   = Ri * Ri.';
            P   = (P + P.')/2;
            muG = gpuArray(single(mu(:)));
            PG  = gpuArray(single(P));
        end

        % per-voxel log-prior (u-dependent part), -(u-mu)' P (u-mu)/2, [1,Nv]
        function lp = logprior_normal(uH, mu, P)
            r  = uH - mu;
            lp = -0.5 .* sum(r .* (P*r), 1);
        end

        % ubar [d,1] and scatter matrix S [d,d] of u [d,n] in double (on the GPU for gpuArray input), gathered
        function [ubar, S] = hyper_suffstats(uH)
            ud      = double(uH);
            n       = size(ud, 2);
            ubar    = sum(ud, 2) ./ n;
            rc      = ud - ubar;
            S       = rc * rc.';
            [ubar, S] = gather(ubar, S);
            S       = (S + S.')/2;
        end

        % one Gibbs block of the hyperparameters given ubar, S of n voxels (host, double)
        function [mu, Sigma] = gibbs_hyper(ubar, S, n, mu, Sigma, hp)
        % hp : resolved hyperprior (resolve_hyperprior); mu/Sigma are the current values
            d = numel(ubar);
            switch hp.hyperprior
                case 'niw'
                    % exact joint draw from the conjugate posterior
                    [mn, kn, Psin, nun] = mcmc_bayes.niw_posterior(ubar, S, n, hp.m0, hp.kappa0, hp.Psi0, hp.nu0);
                    Sigma   = mcmc_bayes.draw_iw_bartlett(Psin, nun);
                    mu      = mcmc_bayes.draw_mvn(mn, Sigma ./ kn);
                case 'jeffreys_half'
                    % Sigma | u, mu (current mu), then mu | Sigma, u
                    dm      = ubar - mu;
                    Smu     = S + n .* (dm * dm.');
                    Smu     = (Smu + Smu.')/2;
                    Sigma   = mcmc_bayes.draw_iw_bartlett(Smu, n - d);
                    mu      = mcmc_bayes.draw_mvn(ubar, Sigma ./ n);
            end
        end

        % NIW posterior parameters from the sufficient statistics
        function [mn, kn, Psin, nun] = niw_posterior(ubar, S, n, m0, kappa0, Psi0, nu0)
        % ubar [d,1], S [d,d] = sum_i (u_i-ubar)(u_i-ubar)', n # voxels; NIW(m0, kappa0, Psi0, nu0)
            ubar    = ubar(:); m0 = m0(:);
            kn      = kappa0 + n;
            nun     = nu0 + n;
            mn      = (kappa0 .* m0 + n .* ubar) ./ kn;
            dm      = ubar - m0;
            Psin    = Psi0 + S + (kappa0 * n / kn) .* (dm * dm.');
            Psin    = (Psin + Psin.')/2;
        end

        % inverse-Wishart draw by the Bartlett decomposition (host, no Statistics toolbox)
        function Sigma = draw_iw_bartlett(Psi, nu)
        % Sigma ~ IW(Psi, nu): density ∝ |Sigma|^(-(nu+d+1)/2) exp(-tr(Psi Sigma^-1)/2), E = Psi/(nu-d-1), nu > d-1
            d = size(Psi, 1);
            U = chol(Psi);                                  % Psi = U'U
            A = zeros(d);
            for j = 1:d
                A(j,j) = sqrt(2 * randg((nu - j + 1)/2));   % chi^2_(nu-j+1)
            end
            if d > 1
                A(tril(true(d), -1)) = randn(d*(d-1)/2, 1);
            end
            T       = A \ U;                                % Sigma^-1 = U^-1 A A' U^-T ~ W(Psi^-1, nu)
            Sigma   = T.' * T;
        end

        % multivariate normal draw (host)
        function x = draw_mvn(m, C)
            x = m(:) + chol(C, 'lower') * randn(numel(m), 1);
        end

        %% Gaussian-mixture hierarchical prior (Phase 7), see the class header
        % per-voxel log-prior (u-dependent part). K > 1: the collapsed mixture density (z summed out),
        %   lp = log sum_k exp(c_k - (u-mu_k)' P_k (u-mu_k)/2),  c_k = log pi_k - log|Sigma_k|/2   [1,Nv],
        % as a stable log-sum-exp (max subtracted), in the class of uH (single on the GPU).
        % c = [] or a single group: exactly logprior_normal (K = 1 path)
        function lp = logprior_mix(uH, c, mu, P)
            if isempty(c) || size(mu, 2) == 1
                lp = mcmc_bayes.logprior_normal(uH, mu, P);
                return
            end
            K   = size(mu, 2);
            q   = zeros(K, size(uH, 2), 'like', uH);
            for k = 1:K
                q(k,:) = c(k) + mcmc_bayes.logprior_normal(uH, mu(:,k), P(:,:,k));
            end
            qm  = max(q, [], 1);
            lp  = qm + log(sum(exp(q - qm), 1));
        end

        % exact responsibilities P(z_i = k | u_i, theta), [K,Nv] double (normalised mixture_logweights)
        function R = mixture_responsibilities(uH, mixD)
            lw  = mcmc_bayes.mixture_logweights(uH, mixD);
            R   = exp(lw - max(lw, [], 1));
            R   = R ./ sum(R, 1);
        end

        % group hyperparameters on the GPU: single mu [d,K] and precision [d,d,K] for the MH log-prior
        % (per group exactly prior_to_gpu), and the double quantities of the z-step
        % (mixD.mu [d,K], .P [d,d,K], .c [K,1] = log pi_k - log|Sigma_k|/2), and .cG = single(c) on the GPU
        % (the constants of the collapsed MH log-prior)
        function [muG, PG, mixD] = mixture_to_gpu(mu, Sigma, piW)
            [d, K]  = size(mu);
            muG     = zeros(d, K, 'single', 'gpuArray');
            PG      = zeros(d, d, K, 'single', 'gpuArray');
            PD      = zeros(d, d, K);
            c       = zeros(K, 1);
            for k = 1:K
                [muG(:,k), PG(:,:,k)] = mcmc_bayes.prior_to_gpu(mu(:,k), Sigma(:,:,k));
                R           = chol(Sigma(:,:,k));
                Ri          = R \ eye(d);
                Pk          = Ri * Ri.';
                PD(:,:,k)   = (Pk + Pk.')/2;
                c(k)        = log(piW(k)) - sum(log(diag(R)));
            end
            mixD = struct('mu', gpuArray(double(mu)), 'P', gpuArray(PD), 'c', gpuArray(c), 'cG', gpuArray(single(c)));
        end

        % unnormalised log responsibilities [K,Nv] (double): log pi_k - log|Sigma_k|/2 - (u-mu_k)'P_k(u-mu_k)/2
        function lw = mixture_logweights(uH, mixD)
            ud  = double(uH);
            K   = size(mixD.mu, 2);
            lw  = zeros(K, size(ud, 2), 'like', ud);
            for k = 1:K
                r       = ud - mixD.mu(:,k);
                lw(k,:) = mixD.c(k) - 0.5 .* sum(r .* (mixD.P(:,:,k) * r), 1);
            end
        end

        % exact z draw from the log weights lw [K,Nv] with one uniform r [1,Nv] per voxel (log-sum-exp
        % stable): z = 1 + #{k < K : cumsum_k(w) < r * sum(w)}, w = exp(lw - max_k lw); z [1,Nv] double
        function z = mixture_zstep(lw, r)
            K   = size(lw, 1);
            w   = exp(lw - max(lw, [], 1));
            cw  = cumsum(w, 1);
            z   = 1 + sum(cw(1:K-1,:) < r .* cw(K,:), 1);
        end

        % per-group n_k [K,1], ubar_k [d,K] and scatter S_k [d,d,K] (double, on the GPU for gpuArray input),
        % gathered; an empty group has n_k = 0, ubar_k = 0, S_k = 0
        function [n, ubar, S] = hyper_suffstats_groups(uH, z, K)
            ud      = double(uH);
            d       = size(ud, 1);
            n       = zeros(K, 1, 'like', ud);
            ubar    = zeros(d, K, 'like', ud);
            S       = zeros(d, d, K, 'like', ud);
            for k = 1:K
                mk          = double(z == k);                    % [1,Nv]
                n(k)        = sum(mk);
                ubar(:,k)   = (ud * mk.') ./ max(n(k), 1);
                rc          = (ud - ubar(:,k)) .* mk;
                S(:,:,k)    = rc * rc.';
            end
            [n, ubar, S] = gather(n, ubar, S);
            S       = (S + permute(S, [2 1 3]))/2;
        end

        % Dirichlet draw (host): G_k ~ Gamma(a_k, 1) (randg), p = G/sum(G)
        function p = draw_dirichlet(a)
            g = randg(a(:));
            p = g ./ sum(g);
        end

        % marginal covariance of a Gaussian mixture: sum_k pi_k (Sigma_k + mu_k mu_k') - mubar mubar'
        function V = mixture_marginal_cov(mu, Sigma, piW)
            [d, K]  = size(mu);
            piW     = piW(:);
            mubar   = mu * piW;
            V       = -mubar * mubar.';
            for k = 1:K
                V = V + piW(k) .* (Sigma(:,:,k) + mu(:,k) * mu(:,k).');
            end
            V = (V + V.')/2;
            if d == 1; V = max(V, 0); end
        end

        % label ordering of one state: ord = groups sorted by mu(1,:) ascending (free mode) or 1:K (fixed),
        % rnk(ord) = 1:K maps an old label to its ordered label
        function [ord, rnk] = mixture_order(mu, doSort)
            K = size(mu, 2);
            if doSort; [~, ord] = sort(mu(1,:), 'ascend'); else; ord = 1:K; end
            rnk = zeros(1, K); rnk(ord) = 1:K;
        end

        function r = mixture_ordering_rule(isFree)
            if isFree
                r = ['groups of every kept sample ordered by mu_k of the first hierarchical parameter, ascending ' ...
                     '(sort; ties keep the lower label); membership rows reordered with the same permutation'];
            else
                r = 'fixed mode: the user''s group labels (no ordering)';
            end
        end

        % free-mode initialisation from k-means (see Phase 7 in the class header)
        function [z, mu, Sigma, piW] = mixture_init(uH, K, floorVar)
        % uH [d,Nv] double (CPU or GPU); floorVar [d,1]; z [1,Nv] (class of uH), mu [d,K], Sigma [d,d,K], piW [K,1] host
            d       = size(uH, 1);
            mX      = mean(uH, 2);
            sX      = std(uH, 0, 2);
            sX(~(sX > 0)) = 1;
            [z, C]  = mcmc_bayes.kmeans_pp((uH - mX) ./ sX, K, 100);
            C       = double(gather(C .* sX + mX));
            mu      = zeros(d, K); Sigma = zeros(d, d, K); nk = zeros(K, 1);
            for k = 1:K
                mk      = (z == k);
                nk(k)   = gather(sum(mk));
                if nk(k) > 0
                    uk      = uH(:, mk);
                    mu(:,k) = double(gather(mean(uk, 2)));
                else
                    mu(:,k) = C(:,k);
                end
                if nk(k) > 1; v = double(gather(var(uH(:, mk), 0, 2))); else; v = zeros(d, 1); end
                Sigma(:,:,k) = diag(max(v, floorVar(:)));
            end
            piW     = max(nk, 1) ./ sum(max(nk, 1));
        end

        % k-means with k-means++ seeding and Lloyd iterations (no Statistics toolbox)
        function [z, C, nIter] = kmeans_pp(X, K, maxIter)
        % X [d,N] points as columns (CPU or GPU, double); the seeding uses the host rng (rand), so the
        % result is deterministic given the seed. z [1,N] labels 1..K (class of X), C [d,K] centres.
        % Lloyd: assign to the nearest centre (lowest label on ties), recompute the means; an emptied
        % cluster is reseeded at the point farthest from its nearest centre; stop when the labels no
        % longer change or after maxIter iterations.
            if nargin < 3 || isempty(maxIter); maxIter = 100; end
            [d, N]  = size(X);
            C       = zeros(d, K, 'like', X);
            idx     = min(N, floor(rand * N) + 1);
            C(:,1)  = X(:, idx);
            D2      = sum((X - C(:,1)).^2, 1);
            for k = 2:K
                tot = double(gather(sum(D2)));
                if tot > 0
                    cs  = double(gather(cumsum(D2)));
                    idx = find(cs >= rand * tot, 1);
                    if isempty(idx); idx = N; end
                else
                    idx = min(N, floor(rand * N) + 1);
                end
                C(:,k)  = X(:, idx);
                D2      = min(D2, sum((X - C(:,k)).^2, 1));
            end
            z = zeros(1, N, 'like', X);
            nIter = 0;
            for it = 1:maxIter
                nIter   = it;
                D       = zeros(K, N, 'like', X);
                for k = 1:K; D(k,:) = sum((X - C(:,k)).^2, 1); end
                [Dmin, zNew] = min(D, [], 1);
                zNew    = cast(zNew, 'like', X);
                for k = 1:K
                    mk  = (zNew == k);
                    nk  = gather(sum(mk));
                    if nk > 0
                        C(:,k) = sum(X .* mk, 2) ./ nk;
                    else
                        [~, j]      = max(Dmin);
                        C(:,k)      = X(:, j);
                        Dmin(j)     = 0;
                        zNew(j)     = k;
                    end
                end
                if isequal(gather(zNew), gather(z)); z = zNew; break; end
                z = zNew;
            end
        end

        % heuristic GPU memory (bytes) of one sampler call, see the class header
        function bytes = estimate_gpu_memory(Nm, Nv, Nvar)
            bytes = 2 * 4 * Nv * (8*Nm + 24*Nvar + 16);
        end

        % error if the coupled free-hyperparameter run does not fit on the GPU
        function check_hierarchical_memory(Nm, Nv, Nvar, hier)
            need = mcmc_bayes.estimate_gpu_memory(Nm, Nv, Nvar);
            if isempty(hier.maxGPUMemory)
                dev   = gpuDevice;
                avail = dev.AvailableMemory;
            else
                avail = hier.maxGPUMemory;
            end
            if need > avail
                error('mcmc_bayes:hierarchicalMemory', ...
                    ['mcmc_bayes: the free-hyperparameter hierarchical prior couples all %d voxels into one GPU call; the estimated ' ...
                     'memory (%.3g GB) exceeds the available %.3g GB. Use the two-stage scheme instead: ' ...
                     '[mu,Sigma,fittingFixed] = mcmc_bayes().estimate_hyper_subset(...) with prior.hierarchical.subsetFraction < 1, ' ...
                     'then mcmc_bayes().optimisation(..., fittingFixed, ...) (fixed mode, voxels independent, can be segmented).'], ...
                    Nv, need/1e9, avail/1e9);
            end
        end

        % u in terms of x for each hierarchical parameter
        function desc = transform_description(method, lb, ub)
            desc = cell(1, numel(method));
            for k = 1:numel(method)
                switch method{k}
                    case 'sigmoid'; desc{k} = sprintf('u = log((x - %g)/(%g - x))  (x = %g + %g*sigmoid(u))', lb(k), ub(k), lb(k), ub(k)-lb(k));
                    case 'log';     desc{k} = 'u = log(x)';
                    otherwise;      desc{k} = 'u = x';
                end
            end
        end

        % resolved prior settings for out.settings.prior
        function s = prior_settings(hier, hp)
            s.hierarchical = struct( ...
                'params',       {hier.params}, ...
                'd',            hier.d, ...
                'hyperprior',   hier.hyperprior, ...
                'fixed',        hier.fixed, ...
                'space',        'N(u | mu, Sigma) on the transformed parameters; no log-Jacobian and no bound rejection for these parameters');
            if hier.fixed
                s.hierarchical.mu       = hier.mu;
                s.hierarchical.Sigma    = hier.Sigma;
            else
                switch hier.hyperprior
                    case 'niw'
                        s.hierarchical.m0       = hp.m0;
                        s.hierarchical.kappa0   = hp.kappa0;
                        s.hierarchical.Psi0     = hp.Psi0;
                        s.hierarchical.nu0      = hp.nu0;
                        s.hierarchical.rules    = hp.rules;
                        s.hierarchical.gibbs    = 'after every MH sweep: Sigma|u ~ IW(Psi_n, nu_n), mu|Sigma,u ~ N(m_n, Sigma/kappa_n)';
                    case 'jeffreys_half'
                        s.hierarchical.gibbs    = 'after every MH sweep: Sigma|u,mu ~ IW(S_mu, n-d), then mu|Sigma,u ~ N(ubar, Sigma/n); p(mu) flat, p(Sigma) ∝ |Sigma|^(-1/2)';
                end
                s.hierarchical.init     = 'every repetition: mu = mean(u), Sigma = diag(max(var(u), floorVar))';
                s.hierarchical.floorVar = hp.floorVar;
            end
            % mixture (Phase 7): fields added for K > 1 only
            if hier.K > 1
                s.hierarchical.K        = hier.K;
                s.hierarchical.space    = ['u_i | z_i = k ~ N(mu_k, Sigma_k), z_i ~ Categorical(pi) on the transformed parameters; ' ...
                                           'no log-Jacobian and no bound rejection for these parameters'];
                s.hierarchical.mhPrior  = ['collapsed mixture: log sum_k exp(log pi_k - log|Sigma_k|/2 - (u-mu_k)''Sigma_k^-1(u-mu_k)/2) ' ...
                                           '(z summed out; stable log-sum-exp, GPU single)'];
                s.hierarchical.membership = 'Rao-Blackwellised: mean over kept samples of P(z_i = k | u_i, theta)';
                s.hierarchical.ordering = mcmc_bayes.mixture_ordering_rule(~hier.fixed);
                if hier.fixed
                    s.hierarchical.pi       = hier.pi;
                    s.hierarchical.zStep    = 'none (fixed mode: z is not needed for the collapsed u updates)';
                else
                    s.hierarchical.alpha    = hier.alpha;
                    s.hierarchical.zStep    = ['partially collapsed Gibbs (van Dyk & Park 2008): after the MH block on u | theta (z collapsed), ' ...
                                               'z_i | u_i, theta ~ Categorical(w_i), log w_ik = log pi_k - log|Sigma_k|/2 - (u_i-mu_k)''Sigma_k^-1(u_i-mu_k)/2, ' ...
                                               'one uniform per voxel (GPU); then theta | u, z']; 
                    s.hierarchical.gibbs    = ['after the z-step: per group k, Sigma_k|u,z ~ IW(Psi_n, nu_n), mu_k|Sigma_k,u,z ~ N(m_n, Sigma_k/kappa_n) ' ...
                                               'from the voxels with z = k (empty group: the hyperprior); pi|z ~ Dirichlet(alpha + n_k)'];
                    s.hierarchical.init     = ['every repetition: k-means (k-means++ seeding, Lloyd, standardised dimensions) on the starting u; ' ...
                                               'mu_k = cluster mean, Sigma_k = diag(max(var_k(u), floorVar)), pi_k = max(n_k,1)/sum max(n_j,1)'];
                    s.hierarchical.emptyGroup = 'n_k = 0: (mu_k, Sigma_k) drawn from the hyperprior NIW(m0, kappa0, Psi0, nu0)';
                end
            end
            s.mrf = [];
        end

        %% MRF prior (Phase 4), see the derivation in the class header
        % resolve and validate fitting.prior.mrf (pure, no GPU)
        function mrf = setup_mrf(fitting, hier)
        % Input
        % -----
        % fitting   : fitting structure (sampled parameter set)
        % hier      : setup_hierarchical output
        % Output
        % ------
        % mrf       : structure
        %   .on                         : true if the MRF prior is used
        %   .potential                  : 'l1' | 'huber' | 'quadratic'
        %   .tau, .W [d,1], .delta [d,1]: temperature, per-parameter weights, Huber thresholds (u space)
        %   .Wrule, .huberDelta         : how W was set, Huber threshold in units of sqrt(Sigma_pp)
        %   .mode, .radius, .connectivity, .edgeWeights, .maxGPUMemory
        %   .update                     : 'chromatic' | 'simultaneous' (TEST ONLY)
        %
            mrf = struct('on', false, 'potential', '', 'tau', [], 'W', [], 'Wrule', '', 'huberDelta', [], 'delta', [], ...
                         'mode', '', 'radius', [], 'connectivity', '', 'edgeWeights', [], 'maxGPUMemory', [], 'update', 'chromatic', ...
                         'subsetForward', true, 'stateWeight', false);
            update = lower(char(field_or_default(fitting, 'mrfUpdate', 'chromatic')));
            if ~any(strcmp(update, {'chromatic','simultaneous'}))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: fitting.mrfUpdate must be ''chromatic'' or ''simultaneous'' (TEST ONLY).');
            end
            isOn = isfield(fitting,'prior') && isstruct(fitting.prior) && isfield(fitting.prior,'mrf') && ...
                   ~isempty(fitting.prior.mrf) && ~(islogical(fitting.prior.mrf) && isscalar(fitting.prior.mrf) && ~fitting.prior.mrf);
            if ~isOn
                if strcmp(update, 'simultaneous')
                    error('mcmc_bayes:invalidMrf', 'mcmc_bayes: fitting.mrfUpdate = ''simultaneous'' needs fitting.prior.mrf.');
                end
                return
            end
            m = fitting.prior.mrf;
            if islogical(m) && isscalar(m); m = struct(); end
            if ~isstruct(m) || ~isscalar(m)
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: fitting.prior.mrf must be a structure (or true).');
            end
            valid = {'potential','tau','W','huberDelta','edgeWeights','mode','radius','connectivity','maxGPUMemory','subsetForward', ...
                     'bayesivimWeights'};
            bad   = setdiff(fieldnames(m), valid);
            if ~isempty(bad)
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: unknown field(s) in fitting.prior.mrf: %s (valid: %s).', ...
                    strjoin(bad, ', '), strjoin(valid, ', '));
            end

            % the MRF alone is improper and needs fixed hyperparameters (two-stage scheme)
            if ~hier.on
                error('mcmc_bayes:mrfRequiresHierarchical', ...
                    'mcmc_bayes: prior.mrf requires prior.hierarchical (the MRF alone is improper: it is shift invariant).');
            end
            if ~hier.fixed
                error('mcmc_bayes:mrfFreeHyperparameters', ...
                    ['mcmc_bayes: prior.mrf with free hyperparameters (prior.hierarchical.fixed = false) is not a valid Gibbs ' ...
                     'conditional (the normalising constant of the joint prior depends on Sigma). Use ' ...
                     'mcmc_bayes().run_two_stage(...) (two-stage empirical Bayes), or set prior.hierarchical.fixed = true with mu/Sigma.']);
            end
            d = hier.d;

            potential = lower(char(field_or_default(m, 'potential', 'l1')));
            if ~any(strcmp(potential, {'l1','huber','quadratic'}))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.potential must be ''l1'', ''huber'' or ''quadratic'' (got ''%s'').', potential);
            end
            tau = field_or_default(m, 'tau', 1);
            if ~(isnumeric(tau) && isscalar(tau) && isfinite(tau) && tau > 0)
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.tau must be a positive finite scalar.');
            end
            if hier.K > 1
                % mixture (Phase 7): marginal covariance of the fixed mixture
                sdPrior = sqrt(diag(mcmc_bayes.mixture_marginal_cov(hier.mu, hier.Sigma, hier.pi)));
                WruleDefault = ['1./sqrt(diag(V)), V = sum_k pi_k (Sigma_k + mu_k mu_k'') - mubar mubar'' ' ...
                                '(marginal covariance of the fixed mixture)'];
            else
                sdPrior = sqrt(diag(hier.Sigma));
                WruleDefault = '1./sqrt(diag(Sigma)) of the fixed Sigma';
            end
            W = field_or_default(m, 'W', []);
            if isempty(W)
                W = 1 ./ sdPrior; Wrule = WruleDefault;
            else
                if ~(isnumeric(W) && any(numel(W) == [1 d]) && all(isfinite(W(:))) && all(W(:) > 0))
                    error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.W must be positive and finite, a scalar or %d entries.', d);
                end
                W = double(W(:)) .* ones(d, 1); Wrule = 'user';
            end
            % TEST ONLY: BayesIVIM's state-dependent spatial weight (see the class header)
            stateWeight = field_or_default(m, 'bayesivimWeights', false);
            if ~(islogical(stateWeight) || isnumeric(stateWeight)) || ~isscalar(stateWeight) || ~any(stateWeight == [0 1])
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.bayesivimWeights must be true or false (TEST ONLY).');
            end
            stateWeight = logical(stateWeight);
            if stateWeight
                if ~strcmp(potential, 'l1')
                    error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.bayesivimWeights (TEST ONLY) requires potential ''l1''.');
                end
                if isfield(m,'W') && ~isempty(m.W)
                    error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.bayesivimWeights (TEST ONLY) replaces W; do not set prior.mrf.W.');
                end
                W = ones(d, 1); Wrule = 'TEST ONLY bayesivimWeights: W = 1, per-voxel weight 1/|u_i| of the current state';
            end
            huberDelta = field_or_default(m, 'huberDelta', 1);
            if ~(isnumeric(huberDelta) && any(numel(huberDelta) == [1 d]) && all(isfinite(huberDelta(:))) && all(huberDelta(:) > 0))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.huberDelta must be positive and finite, a scalar or %d entries.', d);
            end
            delta = double(huberDelta(:)) .* sdPrior;

            mode = lower(char(field_or_default(m, 'mode', '3d')));
            if ~any(strcmp(mode, {'3d','2d'}))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.mode must be ''3d'' or ''2d'' (got ''%s'').', mode);
            end
            radius = field_or_default(m, 'radius', 1);
            if ~(isnumeric(radius) && isscalar(radius) && radius >= 1 && radius == round(radius))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.radius must be a positive integer.');
            end
            connectivity = lower(char(field_or_default(m, 'connectivity', '')));
            if isempty(connectivity)
                if strcmp(mode,'3d') && radius == 1; connectivity = 'face'; else; connectivity = 'full'; end
            end
            if ~any(strcmp(connectivity, {'face','full'}))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.connectivity must be ''face'' or ''full'' (got ''%s'').', connectivity);
            end
            if strcmp(connectivity,'face') && radius ~= 1
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.connectivity = ''face'' needs radius = 1 (use ''full'' for a larger radius).');
            end
            edgeWeights = field_or_default(m, 'edgeWeights', []);
            if ~isempty(edgeWeights) && ~isnumeric(edgeWeights)
                error('mcmc_bayes:invalidEdgeWeights', 'mcmc_bayes: prior.mrf.edgeWeights must be numeric [K, Nv] (or []).');
            end

            mrf.on              = true;
            mrf.potential       = potential;
            mrf.tau             = double(tau);
            mrf.W               = W;
            mrf.Wrule           = Wrule;
            mrf.huberDelta      = double(huberDelta(:));
            mrf.delta           = delta;
            mrf.mode            = mode;
            mrf.radius          = double(radius);
            mrf.connectivity    = connectivity;
            mrf.edgeWeights     = edgeWeights;
            mrf.maxGPUMemory    = field_or_default(m, 'maxGPUMemory', hier.maxGPUMemory);
            mrf.update          = update;
            subsetForward = field_or_default(m, 'subsetForward', true);
            if ~((islogical(subsetForward) || isnumeric(subsetForward)) && isscalar(subsetForward) && any(double(subsetForward) == [0 1]))
                error('mcmc_bayes:invalidMrf', 'mcmc_bayes: prior.mrf.subsetForward must be true or false.');
            end
            mrf.subsetForward   = logical(subsetForward);
            mrf.stateWeight     = stateWeight;
        end

        % neighbour offsets [K,3], sorted lexicographically (offset K+1-k = -offset k)
        function off = mrf_offsets(mode, radius, connectivity)
            r = radius;
            if strcmpi(connectivity, 'face')
                off = [eye(3); -eye(3)];
            else
                [a, b, c] = ndgrid(-r:r, -r:r, -r:r);
                off = [a(:) b(:) c(:)];
            end
            if strcmpi(mode, '2d'); off = off(off(:,3) == 0, :); end
            off = off(any(off ~= 0, 2), :);
            off = unique(off, 'rows');                              % sorted lexicographically
        end

        % neighbour table in the masked-voxel index, see the class header
        function nbr = build_neighbours(mask_idx, dims, mode, radius, connectivity)
        % Input
        % -----
        % mask_idx      : linear indices of the masked voxels (find(mask)), Nv entries
        % dims          : size(mask), up to 3 dimensions
        % mode          : '3d' | '2d' (in-plane: dims 1-2 of the same slice)
        % radius        : positive integer r
        % connectivity  : 'face' (r = 1) | 'full'
        % Output
        % ------
        % nbr           : [K, Nv] int32, position of neighbour k of voxel v in mask_idx, 0 if the
        %                 neighbour is outside the mask or the volume. Asserted symmetric:
        %                 nbr(K+1-k, nbr(k,v)) == v.
        %
            if nargin < 5 || isempty(connectivity)
                if strcmpi(mode,'3d') && radius == 1; connectivity = 'face'; else; connectivity = 'full'; end
            end
            dims = [dims(:).' ones(1, 3)];
            if any(dims(4:end-3) ~= 1)
                error('mcmc_bayes:mrfGeometry', 'mcmc_bayes: the MRF needs a mask with at most 3 dimensions.');
            end
            dims    = dims(1:3);
            mask_idx= double(mask_idx(:));
            Nv      = numel(mask_idx);
            off     = mcmc_bayes.mrf_offsets(mode, radius, connectivity);
            K       = size(off, 1);
            map     = zeros(dims, 'int32');
            map(mask_idx) = int32(1:Nv);
            [i, j, k] = ind2sub(dims, mask_idx);
            nbr     = zeros(K, Nv, 'int32');
            for kk = 1:K
                ii  = i + off(kk,1); jj = j + off(kk,2); ll = k + off(kk,3);
                in  = ii >= 1 & ii <= dims(1) & jj >= 1 & jj <= dims(2) & ll >= 1 & ll <= dims(3);
                nbr(kk, in) = map(sub2ind(dims, ii(in), jj(in), ll(in)));
            end
            mcmc_bayes.check_neighbour_symmetry(nbr);
        end

        % assert nbr(K+1-k, nbr(k,v)) == v for every present neighbour
        function check_neighbour_symmetry(nbr)
            [K, Nv] = size(nbr);
            [kk, v] = find(nbr > 0);
            n       = double(nbr(sub2ind([K Nv], kk, v)));
            back    = nbr(sub2ind([K Nv], K + 1 - kk, n));
            if ~isequal(double(back(:)), v(:))
                error('mcmc_bayes:mrfNeighbours', 'mcmc_bayes: the neighbour table is not symmetric.');
            end
        end

        % colour labels, see the class header; the colouring is asserted proper
        function [colours, Ncolours] = build_colours(mask_idx, dims, mode, radius, connectivity, nbr)
        % Output
        % ------
        % colours   : [1, Nv] colour labels in 1..Ncolours
        % Ncolours  : nominal number of colours (2, (r+1)^2 or (r+1)^3); classes can be empty
        %
            if nargin < 5 || isempty(connectivity)
                if strcmpi(mode,'3d') && radius == 1; connectivity = 'face'; else; connectivity = 'full'; end
            end
            if nargin < 6 || isempty(nbr)
                nbr = mcmc_bayes.build_neighbours(mask_idx, dims, mode, radius, connectivity);
            end
            dims = [dims(:).' ones(1, 3)]; dims = dims(1:3);
            [i, j, k] = ind2sub(dims, double(mask_idx(:)).');
            is2d = strcmpi(mode, '2d');
            if strcmpi(connectivity, 'face')
                if is2d; colours = mod(i + j, 2) + 1; else; colours = mod(i + j + k, 2) + 1; end
                Ncolours = 2;
            else
                q = radius + 1;
                colours = mod(i, q) + q .* mod(j, q) + 1;
                Ncolours = q^2;
                if ~is2d
                    colours = colours + q^2 .* mod(k, q);
                    Ncolours = q^3;
                end
            end
            mcmc_bayes.check_colouring(nbr, colours);
        end

        % assert that no two neighbours share a colour
        function check_colouring(nbr, colours)
            has = nbr > 0;
            cN  = zeros(size(nbr));
            cN(has) = colours(nbr(has));
            cV  = repmat(colours(:).', size(nbr, 1), 1);
            if any(cN(has) == cV(has))
                error('mcmc_bayes:mrfColouring', 'mcmc_bayes: improper colouring (two neighbours share a colour).');
            end
        end

        % validate fixed edge weights [K, Nv]: finite, >= 0 and symmetric; absent neighbours -> 0
        function w = check_edge_weights(w, nbr)
            [K, Nv] = size(nbr);
            if ~(isnumeric(w) && isequal(size(w), [K Nv]))
                error('mcmc_bayes:invalidEdgeWeights', 'mcmc_bayes: prior.mrf.edgeWeights must be [K, Nv] = [%d, %d] (rows as build_neighbours).', K, Nv);
            end
            w   = double(w);
            has = nbr > 0;
            w(~has) = 0;
            if any(~isfinite(w(has))) || any(w(has) < 0)
                error('mcmc_bayes:invalidEdgeWeights', 'mcmc_bayes: prior.mrf.edgeWeights must be finite and >= 0.');
            end
            [kk, v] = find(has);
            n   = double(nbr(sub2ind([K Nv], kk, v)));
            wf  = w(sub2ind([K Nv], kk, v));
            wb  = w(sub2ind([K Nv], K + 1 - kk, n));
            if any(abs(wf - wb) > 1e-6 * max(abs(w(:))))
                error('mcmc_bayes:invalidEdgeWeights', 'mcmc_bayes: prior.mrf.edgeWeights must be symmetric (w_ij == w_ji).');
            end
        end

        % true if prior.mrf is set (non-empty, not false)
        function tf = has_mrf(fitting)
            tf = isstruct(fitting) && isfield(fitting,'prior') && isstruct(fitting.prior) && isfield(fitting.prior,'mrf') && ...
                 ~isempty(fitting.prior.mrf) && ~(islogical(fitting.prior.mrf) && isscalar(fitting.prior.mrf) && ~fitting.prior.mrf);
        end

        % true if prior.hierarchical is set with free hyperparameters (fixed = false)
        function tf = has_free_hierarchy(fitting)
            tf = false;
            if ~(isstruct(fitting) && isfield(fitting,'prior') && isstruct(fitting.prior) && isfield(fitting.prior,'hierarchical')); return; end
            h = fitting.prior.hierarchical;
            if isempty(h) || (islogical(h) && isscalar(h) && ~h); return; end
            tf = ~(isstruct(h) && isfield(h,'fixed') && ~isempty(h.fixed) && logical(h.fixed));
        end

        % free hierarchical prior + MRF: dispatched to run_two_stage by optimisation
        function tf = is_two_stage(fitting)
            tf = mcmc_bayes.has_mrf(fitting) && mcmc_bayes.has_free_hierarchy(fitting);
        end

        % true if the prior couples voxels, so the whole volume must be sampled in one call
        % (free hierarchical prior or MRF); model wrappers must then not segment the data
        function tf = needs_single_segment(fitting)
            tf = mcmc_bayes.has_mrf(fitting) || mcmc_bayes.has_free_hierarchy(fitting);
        end

        % potential rho(x), elementwise (CPU or GPU); delta is the Huber threshold (u space)
        function r = mrf_rho(x, potential, delta)
            switch potential
                case 'l1'
                    r = abs(x);
                case 'quadratic'
                    r = 0.5 .* x.^2;
                case 'huber'
                    % Huber / delta: x^2/(2 delta) for |x| <= delta, |x| - delta/2 otherwise
                    a = abs(x);
                    m = min(a, delta);
                    r = m .* (a - 0.5 .* m) ./ delta;
            end
        end

        % change of the local MRF term Phi_i of the active voxels, current neighbours
        function dPhi = mrf_local_delta(uNewA, uOldA, uAll, nbrA, wA, coef, delta, potential, stateWeight)
        % Input
        % -----
        % uNewA, uOldA  : [dA, Na] proposed and current u of the active voxels (MRF parameters)
        % uAll          : [dA, Nv] current u of all voxels (the neighbours' values)
        % nbrA          : [K, Na] neighbour positions (an absent neighbour points to the voxel
        %                 itself with weight 0)
        % wA            : [K, Na] edge weights w_ij (0 for absent neighbours)
        % coef          : [dA, 1] W_p / tau
        % delta         : [dA, 1] Huber thresholds
        % Output
        % ------
        % stateWeight   : (optional, TEST ONLY) BayesIVIM's term: add the centre voxel
        %                 |uNew - uOld| and divide by |uOld| (weight from the current state)
        % Output
        % ------
        % dPhi          : [1, Na] Phi_i(uNew) - Phi_i(uOld)
        %
            if nargin < 9; stateWeight = false; end
            Na   = size(uNewA, 2);
            dPhi = zeros(1, Na, 'like', uNewA);
            for p = 1:size(uNewA, 1)
                v    = uAll(p, :);
                Unb  = reshape(v(nbrA), size(nbrA));                 % [K, Na]
                r    = mcmc_bayes.mrf_rho(uNewA(p,:) - Unb, potential, delta(p)) - ...
                       mcmc_bayes.mrf_rho(uOldA(p,:) - Unb, potential, delta(p));
                s    = sum(wA .* r, 1);
                if stateWeight
                    s = (s + abs(uNewA(p,:) - uOldA(p,:))) ./ abs(uOldA(p,:));
                end
                dPhi = dPhi + coef(p) .* s;
            end
        end

        % heuristic GPU memory (bytes) of one MRF sampler call, see the class header
        function bytes = estimate_mrf_memory(Nm, Nv, Nvar, K)
            bytes = mcmc_bayes.estimate_gpu_memory(Nm, Nv, Nvar) + 4 * K * Nv * 6;
        end

        % error if the coupled MRF run does not fit on the GPU
        function check_mrf_memory(Nm, Nv, Nvar, K, mrf)
            need = mcmc_bayes.estimate_mrf_memory(Nm, Nv, Nvar, K);
            % Phase 4b: per-colour copies of y and weights
            if isfield(mrf,'subsetForward') && mrf.subsetForward && strcmp(mrf.update, 'chromatic'); need = need + 8 * Nm * Nv; end
            if isempty(mrf.maxGPUMemory)
                dev   = gpuDevice;
                avail = dev.AvailableMemory;
            else
                avail = mrf.maxGPUMemory;
            end
            if need > avail
                error('mcmc_bayes:mrfMemory', ...
                    ['mcmc_bayes: the MRF prior couples all %d voxels (%d neighbours each) into one GPU call; the estimated ' ...
                     'memory (%.3g GB) exceeds the available %.3g GB. The MRF cannot be split into segments; reduce the mask ' ...
                     '(e.g. fewer slices), the neighbourhood (radius, ''face'') or the number of measurements.'], ...
                    Nv, K, need/1e9, avail/1e9);
            end
        end

        % resolved MRF settings for out.settings.mrf
        function s = mrf_settings(mrf, hier, K, NcolNominal, NcolUsed, Nedges, sfInfo)
            switch mrf.potential
                case 'l1';        rho = '|x|';
                case 'huber';     rho = 'x^2/(2 delta) for |x| <= delta, |x| - delta/2 otherwise (Huber/delta)';
                case 'quadratic'; rho = 'x^2/2';
            end
            if isempty(mrf.edgeWeights); ew = 'uniform (1)'; else; ew = 'user (fixed, symmetric)'; end
            if sfInfo.used
                cost = 'forward model on the active colour only (Phase 4b): ~1 full forward evaluation per sweep (x Nvar componentwise)';
            else
                cost = 'forward model evaluated on all voxels per colour step: Ncolours forward evaluations per sweep (x Nvar componentwise)';
            end
            if strcmp(mrf.update, 'simultaneous')
                upd = 'simultaneous: TEST ONLY negative control, targets the WRONG distribution';
            else
                upd = 'chromatic: one MH step per colour class, conditioning on the current other voxels';
            end
            if mrf.stateWeight
                energy = ['TEST ONLY bayesivimWeights (Spinner et al. 2021 code): local term of voxel i, parameter p = ' ...
                          '(1/tau) [sum_{j in N(i)} |u_i^p - u_j^p| + |u_i^p - u_i^p,curr|] / |u_i^p,curr|, weight from the ' ...
                          'CURRENT state; not a well-defined joint prior (the target distribution is undefined)'];
            else
                energy = 'Phi = (1/tau) sum_p W_p sum_{(i,j) in E} w_ij rho(u_i^p - u_j^p), each edge once, u space';
            end
            s = struct( ...
                'params',       {hier.params}, ...
                'potential',    mrf.potential, ...
                'rho',          rho, ...
                'energy',       energy, ...
                'tau',          mrf.tau, ...
                'W',            mrf.W, ...
                'Wrule',        mrf.Wrule, ...
                'huberDelta',   mrf.huberDelta, ...
                'delta',        mrf.delta, ...
                'mode',         mrf.mode, ...
                'radius',       mrf.radius, ...
                'connectivity', mrf.connectivity, ...
                'Kneighbours',  K, ...
                'Ncolours',     NcolNominal, ...
                'NcoloursUsed', NcolUsed, ...
                'Nedges',       Nedges, ...
                'edgeWeights',  ew, ...
                'update',       upd, ...
                'cost',         cost, ...
                'subsetForward', sfInfo);
        end

        % Phase 4b safety check: FWDfunc on each colour subset vs the columns of the full evaluation
        function info = check_subset_forward(FWDfunc, fwdArgs, parsFull, parsSub, actIdx, Nv, tol)
        % Input
        % -----
        % FWDfunc   : forward model handle
        % fwdArgs   : cell, varargin of FWDfunc (passed unchanged)
        % parsFull  : parameter structure of all Nv voxels (injected values included)
        % parsSub   : cell, parameter structure of the active voxels of every colour
        % actIdx    : cell, host indices (1..Nv) of the active voxels of every colour
        % Nv        : # voxels
        % tol       : relative tolerance, |gS - gF(:,act)| <= tol * max_rows |gF(:,v)| per element
        % Output
        % ------
        % info      : structure .requested (true), .used, .reason, .tolerance, .maxRelDiff, .bitwise
        %
            info = struct('requested', true, 'used', false, 'reason', '', 'tolerance', tol, 'maxRelDiff', [], 'bitwise', []);
            try
                gF = FWDfunc(parsFull, fwdArgs{:});
            catch err
                info.reason = sprintf('the forward model errors on all %d voxels (%s)', Nv, err.message);
                return
            end
            if ~ismatrix(gF) || size(gF, 2) ~= Nv
                info.reason = sprintf('the full forward output has size %s, expected [Nm, %d]', mat2str(size(gF)), Nv);
                return
            end
            gF      = double(gather(gF));
            maxRel  = 0; bitwise = true;
            for kc = 1:numel(parsSub)
                act = actIdx{kc};
                try
                    gS = FWDfunc(parsSub{kc}, fwdArgs{:});
                catch err
                    info.reason = sprintf(['the forward model errors on the colour-%d subset of %d voxels (%s); a varargin ' ...
                                           'entry with a voxel dimension or a fixed voxel count'], kc, numel(act), err.message);
                    return
                end
                ref = gF(:, act);
                if ~isequal(size(gS), size(ref))
                    info.reason = sprintf(['the forward output on the colour-%d subset has size %s, expected %s; a varargin ' ...
                                           'entry with a voxel dimension or a fixed voxel count'], kc, mat2str(size(gS)), mat2str(size(ref)));
                    return
                end
                [rel, same] = mcmc_bayes.subset_rel_diff(double(gather(gS)), ref);
                maxRel  = max(maxRel, rel);
                bitwise = bitwise && same;
                if ~(rel <= tol)
                    info.maxRelDiff = maxRel; info.bitwise = false;
                    info.reason = sprintf(['the forward output on the colour-%d subset differs from the full evaluation ' ...
                                           '(max relative difference %.3g > %.3g): the model is not column-separable, e.g. a ' ...
                                           'varargin entry with a voxel dimension'], kc, rel, tol);
                    return
                end
            end
            info.used       = true;
            info.maxRelDiff = maxRel;
            info.bitwise    = bitwise;
            if bitwise
                info.reason = 'subset outputs equal the columns of the full evaluation (bitwise) at the starting state';
            else
                info.reason = sprintf('subset outputs match the full evaluation at the starting state within %.3g (max %.3g)', tol, maxRel);
            end
        end

        % largest element difference relative to the column scale max_rows |ref(:,v)| (finite
        % entries); equal values (incl. Inf and NaN at the same positions) count as 0, a one-sided
        % NaN/Inf as Inf. same: true if a and b are identical (NaN == NaN)
        function [rel, same] = subset_rel_diff(a, b)
            same    = isequaln(a, b);
            if same; rel = 0; return; end
            bf      = b; bf(~isfinite(bf)) = 0;
            scale   = max(max(abs(bf), [], 1), realmin);
            dif     = abs(a - b);
            dif((a == b) | (isnan(a) & isnan(b))) = 0;
            dif(isnan(dif)) = Inf;
            rel     = max(dif ./ scale, [], 'all');
            if isempty(rel); rel = 0; end
        end

        % largest |a - b| over all elements (equal infinities count as 0, one-sided NaN/Inf as Inf)
        function e = max_abs_diff(a, b)
            a   = double(gather(a)); b = double(gather(b));
            dif = abs(a - b);
            same = (a == b) | (isnan(a) & isnan(b));
            dif(same) = 0;
            dif(isnan(dif)) = Inf;
            e   = max([dif(:); 0]);
        end

        % subset of the masked voxels of an image (or [Nm, Nvox] matrix, as in mcmc)
        function out = subset_voxels(data, mask, idx)
        % idx : linear indices into mask
            if ismatrix(data)
                out = data(:, idx);
                return
            end
            sz      = size(data);
            nd      = find(cumprod(sz) == numel(mask), 1, 'last');     % last spatial dimension
            out     = reshape(data, numel(mask), []);
            out     = out(idx, :);
            out     = reshape(out, [numel(idx) 1 1 sz(nd+1:end)]);
        end

        % convert estimation into organised output structure
        function out = res2out(xPosterior,fitting,mask,diagnostics)
        % Same as mcmc.res2out; with the 4th input (new path only), also
        % attaches out.diagnostics and out.settings. The legacy call with
        % 3 inputs is passed through unchanged.
        %
            if nargin < 4 || isempty(diagnostics)
                out = res2out@mcmc(xPosterior,fitting,mask);
                return
            end

            % nuisance fields recovered by post-hoc draws are not confined to [lb,ub]
            % (e.g. S0 on the real line); widen their bounds to the sample range so
            % that the 'mode' histogram (edges from lb/ub by modelParams index) covers them
            if isfield(diagnostics,'recoveredParams')
                for k = 1:numel(diagnostics.recoveredParams)
                    p   = diagnostics.recoveredParams{k};
                    idx = find(strcmp(fitting.modelParams, p));
                    if isempty(idx) || isempty(xPosterior.(p)); continue; end   % e.g. no kept samples (burn-in >= iterations)
                    fitting.lb(idx) = min(fitting.lb(idx), double(min(xPosterior.(p)(:))));
                    fitting.ub(idx) = max(fitting.ub(idx), double(max(xPosterior.(p)(:))));
                end
            end
            out = res2out@mcmc(xPosterior,fitting,mask);

            % acceptance, step size, ESS/R-hat, cache check and adaptive covariance (shared with mcmc)
            out = mcmc.diagnostics2out(out, xPosterior, diagnostics, mask);

            % hierarchical prior: hyperparameter samples and summaries (u space)
            if isfield(diagnostics,'hyper') && ~isempty(diagnostics.hyper)
                out.hyper = mcmc_bayes.hyper2out(diagnostics.hyper);
                % mixture (Phase 7): membership probabilities [x,y,z,K] and MAP label [x,y,z] (0 outside the mask)
                if isfield(diagnostics.hyper, 'membership')
                    memb = diagnostics.hyper.membership;                % [Nv,K]
                    [~, lab] = max(memb, [], 2);
                    out.hyper.membership = mcmc_bayes.vec2image(memb, mask);
                    out.hyper.mapLabel   = mcmc_bayes.vec2image(lab, mask);
                end
            end

            out.settings = diagnostics.settings;
        end

        % out.hyper from the hyperparameter samples, see the class header
        function H = hyper2out(hyper)
            if isfield(hyper, 'K') && hyper.K > 1
                H = mcmc_bayes.hyper2out_mixture(hyper);
                return
            end
            H.params        = hyper.params;
            H.transform     = hyper.transform;
            H.hyperprior    = hyper.hyperprior;
            H.fixed         = hyper.fixed;
            if hyper.fixed
                H.mean.mu       = hyper.mu;     H.mean.Sigma    = hyper.Sigma;
                H.median.mu     = hyper.mu;     H.median.Sigma  = hyper.Sigma;
                return
            end
            mu      = hyper.muPost;                     % [d, Ns, Nrep]
            Sigma   = hyper.SigmaPost;                  % [d, d, Ns, Nrep]
            [d, Ns, Nrep] = size(mu, 1:3);
            SigmaF  = reshape(Sigma, d*d, Ns, Nrep);
            H.posterior.mu      = mu;
            H.posterior.Sigma   = Sigma;
            if Ns == 0 && isfield(hyper,'muLast')
                % no kept samples (burn-in >= iterations, e.g. a model wrapper's short memory probe):
                % summarise by the final Gibbs state, averaged over chains, so that run_two_stage can proceed
                H.mean.mu       = mean(hyper.muLast, 2);        H.mean.Sigma    = mean(hyper.SigmaLast, 3);
                H.median.mu     = H.mean.mu;                    H.median.Sigma  = H.mean.Sigma;
                H.summary       = 'no kept samples: mean/median are the final Gibbs state averaged over chains';
                return
            end
            H.mean.mu           = mean(reshape(mu, d, []), 2);
            H.mean.Sigma        = reshape(mean(reshape(SigmaF, d*d, []), 2), d, d);
            H.median.mu         = median(reshape(mu, d, []), 2);
            H.median.Sigma      = reshape(median(reshape(SigmaF, d*d, []), 2), d, d);
            if Ns >= 4
                H.ess.mu        = mcmc_bayes.ess(mu);
                H.ess.Sigma     = reshape(mcmc_bayes.ess(SigmaF), d, d);
                if Nrep > 1
                    H.rhat.mu       = mcmc_bayes.rhat(mu);
                    H.rhat.Sigma    = reshape(mcmc_bayes.rhat(SigmaF), d, d);
                end
            end
        end

        % out.hyper of the mixture prior (K > 1), ordered groups, see Phase 7 in the class header
        function H = hyper2out_mixture(hyper)
            H.params        = hyper.params;
            H.transform     = hyper.transform;
            H.hyperprior    = hyper.hyperprior;
            H.fixed         = hyper.fixed;
            H.K             = hyper.K;
            H.ordering      = hyper.ordering;
            if hyper.fixed
                H.mean.mu   = hyper.mu;     H.mean.Sigma    = hyper.Sigma;      H.mean.pi   = hyper.pi;
                H.median    = H.mean;
                return
            end
            H.alpha         = hyper.alpha;
            mu      = hyper.muPost;                     % [d, K, Ns, Nrep]
            Sigma   = hyper.SigmaPost;                  % [d, d, K, Ns, Nrep]
            piP     = hyper.piPost;                     % [K, Ns, Nrep]
            d       = size(mu, 1); K = hyper.K;
            Ns      = size(piP, 2); Nrep = size(piP, 3);
            muF     = reshape(mu, d*K, Ns, Nrep);
            SigmaF  = reshape(Sigma, d*d*K, Ns, Nrep);
            H.posterior.mu      = mu;
            H.posterior.Sigma   = Sigma;
            H.posterior.pi      = piP;
            if Ns == 0
                % no kept samples: final Gibbs state (ordered), averaged over chains
                H.mean.mu       = mean(hyper.muLast, 3);
                H.mean.Sigma    = mean(hyper.SigmaLast, 4);
                H.mean.pi       = mean(hyper.piLast, 2);
                H.median        = H.mean;
                H.summary       = 'no kept samples: mean/median are the final Gibbs state (ordered groups) averaged over chains';
                return
            end
            H.mean.mu           = reshape(mean(reshape(muF, d*K, []), 2), d, K);
            H.mean.Sigma        = reshape(mean(reshape(SigmaF, d*d*K, []), 2), d, d, K);
            H.mean.pi           = mean(reshape(piP, K, []), 2);
            H.median.mu         = reshape(median(reshape(muF, d*K, []), 2), d, K);
            H.median.Sigma      = reshape(median(reshape(SigmaF, d*d*K, []), 2), d, d, K);
            H.median.pi         = median(reshape(piP, K, []), 2);
            if Ns >= 4
                H.ess.mu        = reshape(mcmc_bayes.ess(muF), d, K);
                H.ess.Sigma     = reshape(mcmc_bayes.ess(SigmaF), d, d, K);
                H.ess.pi        = mcmc_bayes.ess(piP);
                if Nrep > 1
                    H.rhat.mu       = reshape(mcmc_bayes.rhat(muF), d, K);
                    H.rhat.Sigma    = reshape(mcmc_bayes.rhat(SigmaF), d, d, K);
                    H.rhat.pi       = mcmc_bayes.rhat(piP);
                end
            end
        end

    end

end

% elementwise kernel of mcmc_bayes.loglik_marginal (GPU arrayfun):
%   logL = -a*log(R) - cG*log(gWg), -Inf for R <= rssFloor, gWg <= gFloor, or NaN
%   (the comparisons are false for NaN). The logs are evaluated on clamped
%   values (max/min ignore NaN), so logL stays finite before the final -Inf
%   and keeps the class of R in both branches.
function logL = marginal_kernel(R, gWg, a, cG, rssFloor, gFloor)
ok      = (R > rssFloor) && (gWg > gFloor) && (gWg < Inf);
logL    = -a*log(max(R, rssFloor)) - cG*log(min(max(gWg, gFloor), 1/gFloor));
if ~ok
    logL = logL - Inf;
end
end

% value of an optional field, or the default if the field is absent or empty
function v = field_or_default(s, name, def)
if isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = def;
end
end
