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
% New fitting options (legacy defaults)
%   .parameterTransform : 'linear'      'linear'|'sigmoid'|'log', or a cell with one entry per modelParams
%   .likelihood         : 'gaussian'    'gaussian'|'marginal_noise'|'marginal_S0noise'|'marginal_S0noise_flat'
%   .S0Param            : ''            name of the amplitude parameter in modelParams, required by
%                                       (and only allowed with) 'marginal_S0noise(_flat)'
%   .updateScheme       : 'joint'       'joint'|'componentwise'
%   .adaptStepSize      : false         adapt proposal scale during burn-in (frozen afterwards)
%   .adaptInterval      : 50            # iterations between two adaptation steps
%   .adaptTarget        : []            target acceptance rate, [] -> 0.234 (joint) | 0.44 (componentwise)
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
%       .mrf            : []            MRF prior (Phase 4, not implemented yet)
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
%                                       largest difference to the cached values (diagnostics.cacheCheck)
%
% Phase 1 (sampler infrastructure) design notes
% ---------------------------------------------
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
% Date created: 26 September 2026
% Date modified: 26 September 2026 (Phase 1: transforms, update schemes, adaptation, overdisp, diagnostics)
% Date modified: 26 September 2026 (Phase 2: marginal likelihoods, S0Param injection, nuisance recovery)
% Date modified: 26 September 2026 (Phase 3: hierarchical Normal prior; m counts non-zero weights only)
%

    methods
        function out = optimisation(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
        % Input
        % ----------
        % Same as mcmc.optimisation, plus the new fitting options listed in
        % the class header.
        %

            [isLegacy, nonDefault] = mcmc_bayes.isLegacy(fitting);
            forceNewPath = isstruct(fitting) && isfield(fitting,'forceNewPath') && ~isempty(fitting.forceNewPath) && fitting.forceNewPath;

            % legacy path: identical to mcmc
            if isLegacy && ~forceNewPath
                out = optimisation@mcmc(this, data, mask, weights, pars0, fitting, FWDfunc, varargin{:});
                return
            end

            % Phase 4 options (MRF prior) are not available yet
            if ismember('prior', nonDefault) && isstruct(fitting.prior) && isfield(fitting.prior,'mrf') && ~isempty(fitting.prior.mrf)
                error('mcmc_bayes:notImplemented', ...
                    'mcmc_bayes: non-default option(s) not implemented yet: prior.mrf');
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
            % geometry for later phases (neighbour tables for MRF priors)
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

        function [muHat, SigmaHat, fittingFixed, outSub, idxSub] = estimate_hyper_subset(this, data, mask, weights, pars0, fitting, FWDfunc, varargin)
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
        % muHat         : [d,1] posterior mean of mu (u space)
        % SigmaHat      : [d,d] posterior mean of Sigma (u space)
        % fittingFixed  : fitting with prior.hierarchical.fixed = true, .mu = muHat,
        %                 .Sigma = SigmaHat, .subsetFraction = 1 (stage 2 input)
        % outSub        : mcmc_bayes output of the subset run (image dims [Nsub,1,1])
        % idxSub        : linear indices (into mask) of the subset voxels
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

            fittingFixed = fitting;
            h.fixed = true; h.mu = muHat; h.Sigma = SigmaHat; h.subsetFraction = 1;
            fittingFixed.prior.hierarchical = h;
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
                if isGibbs
                    muPost      = zeros(d, Ns, fitting.repetition);
                    SigmaPost   = zeros(d, d, Ns, fitting.repetition);
                end
            end
            if fitting.checkCache
                cacheErr = struct('loglik', 0, 'logprior', 0, 'logjac', 0, 'Ncheck', 0);
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
            if isHier
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

            counter = 0; start = tic;
            for k = 1:fitting.iteration

                if ~isComponent
                    % ========== joint update: all parameters of a voxel together ==========
                    % 1. make a proposal with normal distribution in u space
                    uProposed       = uCurr + sigma.*randn(size(uCurr),'like',uCurr);
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
                    if isHier; lpProposed = this.logprior_normal(uProposed(hIdx,:), muG, PG); end
                    % 2.2 accept with probability min(1, exp(logRatio)); NaN is rejected
                    if hasJac
                        logRatio            = logLProposed - logLCurr + sum(logJProposed - logJCurr, 1);
                        if isHier; logRatio = logRatio + (lpProposed - lpCurr); end
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    elseif isMarginal || isHier
                        % degenerate states have logL = -Inf; -Inf - (-Inf) = NaN is rejected
                        logRatio            = logLProposed - logLCurr;
                        if isHier; logRatio = logRatio + (lpProposed - lpCurr); end
                        isAccepted          = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                    else
                        % neutral path: same expression as mcmc.metropolis_hastings (the GPU evaluates
                        % fused elementwise expressions slightly differently, so keep it verbatim)
                        acceptanceRatio     = min(exp(logLProposed-logLCurr), 1);
                        isAccepted          = acceptanceRatio > rand(1,Nv,'like',logLProposed);
                        isOutofbound        = isOutofbound | isnan(logLProposed);  % mcmc would accept NaN via min(NaN,1) = 1
                    end
                    isAccepted(isOutofbound)= 0;    % reject out of bound (and NaN) proposal
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
                    isAccepted = false(Nvar, Nv, 'like', isRejectRow);
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
                            lpProposed          = this.logprior_normal(uProposedH, muG, PG);
                            logRatio            = logRatio + (lpProposed - lpCurr);
                        end
                        isAccepted_p                = exp(logRatio) > rand(1,Nv,'like',logLProposed);
                        isAccepted_p(isOutofbound)  = 0;
                        % 3. update parameter kp and the cache before the next parameter
                        logLCurr(isAccepted_p)      = logLProposed(isAccepted_p);
                        if isMarginal; statsCurr(:,isAccepted_p) = statsProposed(:,isAccepted_p); end
                        if isHierRow(kp); lpCurr(isAccepted_p) = lpProposed(isAccepted_p); end
                        uCurr(kp,isAccepted_p)      = uProposed_p(isAccepted_p);
                        xCurr(kp,isAccepted_p)      = xProposed_p(isAccepted_p);
                        if useJac(kp); logJCurr(kp,isAccepted_p) = logJProposed_p(isAccepted_p); end
                        isAccepted(kp,:)            = isAccepted_p;
                    end
                end

                % 3. Gibbs block: exact conditional draws of (mu, Sigma), then refresh the log-prior cache
                if isGibbs
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
                        cacheErr.logprior = max(cacheErr.logprior, this.max_abs_diff(this.logprior_normal(uCurr(hIdx,:), muG, PG), lpCurr));
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
                        if mod(k, fitting.adaptInterval) == 0
                            jAdapt  = jAdapt + 1;
                            delta   = this.adapt_gain(jAdapt) .* (accWin./fitting.adaptInterval - fitting.adaptTarget);
                            % joint: delta is [1,Nv] and scales all parameters of a voxel together
                            sigma   = sigma .* exp(delta);
                            if hasJac; sigma(~isLinear,:) = min(sigma(~isLinear,:), uWidth(~isLinear)); end
                            accWin(:) = 0;
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
                    if isGibbs; muPost(:,counter,ii) = mu; SigmaPost(:,:,counter,ii) = Sigma; end
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
            if isHier
                hyper = struct('params', {hier.params}, ...
                               'transform', {this.transform_description(method(hIdx), fitting.lb(hIdx), fitting.ub(hIdx))}, ...
                               'hyperprior', hier.hyperprior, 'fixed', hier.fixed);
                if isGibbs
                    hyper.muPost    = muPost;
                    hyper.SigmaPost = SigmaPost;
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
            diagnostics.settings    = struct( ...
                'parameterTransform',   {method}, ...
                'updateScheme',         lower(fitting.updateScheme), ...
                'adaptStepSize',        isAdapt, ...
                'adaptInterval',        fitting.adaptInterval, ...
                'adaptTarget',          fitting.adaptTarget, ...
                'adaptRule',            'log(sigma) += 2*j^(-0.6)*(acc_j - adaptTarget), every adaptInterval iterations within burn-in', ...
                'Nadapt',               Nadapt*isAdapt, ...
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
            nonDefault = {};

            if isempty(fitting) || ~isstruct(fitting)
                tf = true;
                return
            end

            % option name, legacy default
            defaults = { 'parameterTransform',  'linear';
                         'likelihood',          'gaussian';
                         'S0Param',             '';
                         'updateScheme',        'joint';
                         'adaptStepSize',       false;
                         'adaptInterval',       50;
                         'adaptTarget',         [];
                         'overdisp',            0;
                         'prior',               [];
                         'fixedParams',         []};

            for k = 1:size(defaults,1)
                name = defaults{k,1};
                if ~isfield(fitting,name); continue; end

                val = fitting.(name);
                ref = defaults{k,2};

                if isempty(ref)
                    isDefault = isempty(val);
                elseif ischar(ref)
                    if iscell(val)  % per-parameter setting, e.g. parameterTransform
                        isDefault = ~isempty(val) && all(cellfun(@(x) (ischar(x)||isstring(x)) && strcmpi(x,ref), val));
                    else
                        isDefault = (ischar(val)||isstring(val)) && strcmpi(val,ref);
                    end
                else
                    isDefault = (isnumeric(val)||islogical(val)) && isscalar(val) && val == ref;
                end

                if ~isDefault; nonDefault{end+1} = name; end %#ok<AGROW>
            end

            tf = isempty(nonDefault);
        end

        % check and set default fitting algorithm parameters for the new path
        function fitting2 = check_set_default_bayes(fitting)
        % Input
        % -----
        % fitting       : structure contains fitting algorithm parameters, see class header
        %
            fitting2 = mcmc.check_set_default_basic(fitting);

            if ~isfield(fitting,'parameterTransform');  fitting2.parameterTransform = 'linear';      end
            if ~isfield(fitting,'likelihood');          fitting2.likelihood         = 'gaussian';    end
            if ~isfield(fitting,'S0Param');             fitting2.S0Param            = '';            end
            if ~isfield(fitting,'updateScheme');        fitting2.updateScheme       = 'joint';       end
            if ~isfield(fitting,'adaptStepSize');       fitting2.adaptStepSize      = false;         end
            if ~isfield(fitting,'adaptInterval');       fitting2.adaptInterval      = 50;            end
            if ~isfield(fitting,'adaptTarget');         fitting2.adaptTarget        = [];            end
            if ~isfield(fitting,'overdisp');            fitting2.overdisp           = 0;             end
            if ~isfield(fitting,'prior');               fitting2.prior              = [];            end
            if ~isfield(fitting,'forceNewPath');        fitting2.forceNewPath       = false;         end
            if ~isfield(fitting,'fixedParams');         fitting2.fixedParams        = [];            end
            if ~isfield(fitting,'checkCache');          fitting2.checkCache         = false;         end

            if ~any(strcmpi(fitting2.updateScheme,{'joint','componentwise'}))
                error('mcmc_bayes:invalidUpdateScheme', ...
                    'mcmc_bayes: fitting.updateScheme must be ''joint'' or ''componentwise'' (got ''%s'').', char(fitting2.updateScheme));
            end
            % target acceptance rate per update scheme
            if isempty(fitting2.adaptTarget)
                if strcmpi(fitting2.updateScheme,'componentwise'); fitting2.adaptTarget = 0.44; else; fitting2.adaptTarget = 0.234; end
            end
            if isempty(fitting2.overdisp); fitting2.overdisp = 0; end
        end

        % display the new sampler settings
        function display_bayes_algorithm_parameters(fitting)
            method = fitting.parameterTransform;
            if iscell(method); method = strjoin(cellstr(method),','); end
            disp(['Transform(s)      : ', char(method)]);
            disp(['Update scheme     : ', char(fitting.updateScheme)]);
            if fitting.adaptStepSize
                disp(['Adapt step size   : true (interval ', num2str(fitting.adaptInterval), ', target ', num2str(fitting.adaptTarget), ')']);
            else
                disp( 'Adapt step size   : false');
            end
            disp(['Over-dispersion   : ', num2str(fitting.overdisp)]);
            disp(['Likelihood        : ', char(fitting.likelihood)]);
            if ~isempty(fitting.S0Param); disp(['S0 parameter      : ', char(fitting.S0Param), ' (marginalised)']); end
            if isstruct(fitting.prior) && isfield(fitting.prior,'hierarchical') && ~isempty(fitting.prior.hierarchical) && ...
                    ~(islogical(fitting.prior.hierarchical) && ~fitting.prior.hierarchical)
                h = fitting.prior.hierarchical;
                if isstruct(h) && isfield(h,'fixed') && ~isempty(h.fixed) && h.fixed
                    disp( 'Prior             : hierarchical Normal on u, fixed mu/Sigma');
                elseif isstruct(h) && isfield(h,'hyperprior') && ~isempty(h.hyperprior)
                    disp(['Prior             : hierarchical Normal on u, hyperprior ', char(h.hyperprior)]);
                else
                    disp( 'Prior             : hierarchical Normal on u, hyperprior niw');
                end
            end
        end

        % Gaussian log-likelihood, same computation as mcmc.metropolis_hastings
        function logL = loglik_gaussian(x_struct, y, weights, Nm, FWDfunc, varargin)
            logL = arrayfun(@logP_Gaussian, sum( weights.* (FWDfunc(x_struct,varargin{:})-y).^2, 1 ), x_struct.noise, Nm);
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
        %   .fixed, .mu, .Sigma       : fixed mode and its hyperparameters (mu [d,1], Sigma [d,d])
        %   .subsetFraction, .maxGPUMemory
        %
            hier = struct('on', false, 'params', {{}}, 'idx', [], 'd', 0, 'hyperprior', '', ...
                          'm0', [], 'kappa0', [], 'Psi0', [], 'nu0', [], 'fixed', false, 'mu', [], 'Sigma', [], ...
                          'subsetFraction', 1, 'maxGPUMemory', []);
            if ~isfield(fitting,'prior') || isempty(fitting.prior); return; end
            prior = fitting.prior;
            if ~isstruct(prior) || ~isscalar(prior)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: fitting.prior must be a structure with the field(s) hierarchical and/or mrf.');
            end
            bad = setdiff(fieldnames(prior), {'hierarchical','mrf'});
            if ~isempty(bad)
                error('mcmc_bayes:invalidPrior', 'mcmc_bayes: unknown field(s) in fitting.prior: %s (valid: hierarchical, mrf).', strjoin(bad, ', '));
            end
            if isfield(prior,'mrf') && ~isempty(prior.mrf)
                error('mcmc_bayes:notImplemented', 'mcmc_bayes: non-default option(s) not implemented yet: prior.mrf');
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
            valid = {'hyperprior','m0','kappa0','Psi0','nu0','fixed','mu','Sigma','params','subsetFraction','maxGPUMemory'};
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

            % fixed mode
            fixed   = logical(field_or_default(h, 'fixed', false));
            mu      = field_or_default(h, 'mu', []);
            Sigma   = field_or_default(h, 'Sigma', []);
            if fixed
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
            s.mrf = [];
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

        % Robbins-Monro gain of the j-th adaptation step
        function gamma = adapt_gain(j)
            gamma = 2 * j.^(-0.6);
        end

        %% parameter transforms
        % All transform helpers operate row-wise: x/u is [Nvar, ...], row k
        % uses method{k}, lb(k), ub(k). They work on CPU and GPU arrays.

        % parse transform specification into a 1xNvar cellstr (lower case)
        function method = parse_transform(spec, Nvar)
            valid = {'linear','sigmoid','log'};
            if ischar(spec) || (isstring(spec) && isscalar(spec))
                method = repmat({lower(char(spec))}, 1, Nvar);
            elseif iscell(spec) || isstring(spec)
                method = cellfun(@(s) lower(char(s)), cellstr(spec), 'UniformOutput', false);
                method = method(:).';
                if numel(method) ~= Nvar
                    error('mcmc_bayes:invalidTransform', ...
                        'mcmc_bayes: parameterTransform has %d entries but there are %d model parameters.', numel(method), Nvar);
                end
            else
                error('mcmc_bayes:invalidTransform', 'mcmc_bayes: parameterTransform must be a string or a cell of strings.');
            end
            isValid = ismember(method, valid);
            if ~all(isValid)
                error('mcmc_bayes:invalidTransform', ...
                    'mcmc_bayes: unknown parameterTransform ''%s'' (valid: %s).', method{find(~isValid,1)}, strjoin(valid,', '));
            end
        end

        % check that the bounds are compatible with the transforms
        function check_transform_bounds(method, lb, ub, modelParams)
            if nargin < 4; modelParams = arrayfun(@(k) sprintf('#%d',k), 1:numel(method), 'UniformOutput', false); end
            for k = 1:numel(method)
                if strcmp(method{k},'linear'); continue; end
                if ~(isfinite(lb(k)) && isfinite(ub(k)) && ub(k) > lb(k))
                    error('mcmc_bayes:invalidBounds', ...
                        'mcmc_bayes: parameter %s with transform ''%s'' needs finite bounds with ub > lb.', modelParams{k}, method{k});
                end
                if strcmp(method{k},'log') && lb(k) < 0
                    error('mcmc_bayes:invalidBounds', ...
                        'mcmc_bayes: parameter %s with transform ''log'' needs lb >= 0 (got %g).', modelParams{k}, lb(k));
                end
            end
        end

        % native -> u
        function u = transform_forward(x, method, lb, ub)
            Nvar    = size(x,1);
            method  = mcmc_bayes.parse_transform(method, Nvar);
            u       = x;
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % clamp slightly inside the box, then logit
                        epsB    = 1e-4 * (ub(k) - lb(k));
                        xc      = min(max(x(k,:), lb(k)+epsB), ub(k)-epsB);
                        t       = (xc - lb(k)) ./ (ub(k) - lb(k));
                        u(k,:)  = log(t) - log1p(-t);
                    case 'log'
                        if isfinite(ub(k))
                            epsB    = 1e-4 * (ub(k) - lb(k));
                            xc      = min(max(x(k,:), lb(k)+epsB), ub(k)-epsB);
                        else
                            % unbounded above (hierarchical 'log', lb = 0): only keep x > 0
                            xc      = max(max(x(k,:), lb(k)), 1e-30);
                        end
                        u(k,:)  = log(xc);
                end
            end
        end

        % u -> native
        function x = transform_inverse(u, method, lb, ub)
            Nvar    = size(u,1);
            method  = mcmc_bayes.parse_transform(method, Nvar);
            x       = u;
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % 1/(1+exp(-u)) is finite for all u (exp overflow -> 0)
                        x(k,:)  = lb(k) + (ub(k) - lb(k)) ./ (1 + exp(-u(k,:)));
                    case 'log'
                        x(k,:)  = exp(u(k,:));
                end
            end
        end

        % log|dx/du| as a function of u
        function logJ = transform_logjac(u, method, lb, ub)
            Nvar    = size(u,1);
            method  = mcmc_bayes.parse_transform(method, Nvar);
            logJ    = zeros(size(u), 'like', u);
            for k = 1:Nvar
                switch method{k}
                    case 'sigmoid'
                        % log[(ub-lb) s(u) s(-u)], log s(u) = -softplus(-u)
                        uk          = u(k,:);
                        logJ(k,:)   = log(ub(k) - lb(k)) - mcmc_bayes.softplus(uk) - mcmc_bayes.softplus(-uk);
                    case 'log'
                        logJ(k,:)   = u(k,:);
                end
            end
        end

        % numeric code of each transform for the fused GPU kernel: 0 linear, 1 sigmoid, 2 log
        function code = transform_code(method)
            code = zeros(numel(method),1);
            code(strcmp(method,'sigmoid'))  = 1;
            code(strcmp(method,'log'))      = 2;
        end

        % fused u -> native and log|dx/du| in one GPU kernel (same maths as
        % transform_inverse and transform_logjac), used inside the sampling loop
        function [x, logJ] = transform_inverse_logjac_fused(u, code, lb, ub)
        % u         : [Nvar, Nv] gpuArray
        % code      : [Nvar, 1] gpuArray, see transform_code
        % lb, ub    : [Nvar, 1] gpuArray
            [x, logJ] = arrayfun(@transform_kernel, u, code, lb, ub);
        end

        % numerically stable log(1+exp(z))
        function y = softplus(z)
            y = max(z,0) + log1p(exp(-abs(z)));
        end

        %% diagnostics
        % split-R-hat (Gelman et al., BDA3), not rank-normalised
        function R = rhat(x)
        % Input
        % -----
        % x     : samples, [Nv, Ns, Nchains]
        % Output
        % ------
        % R     : split-R-hat, [Nv, 1]
        %
            xs          = mcmc_bayes.split_chains(double(x));
            n           = size(xs,2);
            chainMean   = mean(xs,2);                       % [Nv,1,M]
            chainVar    = var(xs,0,2);                      % [Nv,1,M]
            W           = mean(chainVar,3);
            B_n         = var(chainMean,0,3);               % B/n
            varPlus     = (n-1)/n .* W + B_n;
            R           = sqrt(varPlus ./ W);
        end

        % multi-chain effective sample size (Geyer initial positive + monotone sequence, on split chains)
        function N_eff = ess(x)
        % Input
        % -----
        % x     : samples, [Nv, Ns, Nchains]
        % Output
        % ------
        % N_eff : effective sample size, [Nv, 1]
        %
            xs          = mcmc_bayes.split_chains(double(x));
            [~, n, M]   = size(xs);

            chainMean   = mean(xs,2);
            xc          = xs - chainMean;
            % autocovariance per chain via FFT (biased, divided by n)
            nfft        = 2^nextpow2(2*n);
            f           = fft(xc, nfft, 2);
            acov        = real(ifft(abs(f).^2, [], 2));
            acov        = acov(:,1:n,:) ./ n;               % [Nv,n,M]

            chainVar    = acov(:,1,:) .* n ./ (n-1);        % unbiased per-chain variance
            W           = mean(chainVar,3);                 % [Nv,1]
            varPlus     = (n-1)/n .* W;
            if M > 1; varPlus = varPlus + var(chainMean,0,3); end

            rho         = 1 - (W - mean(acov,3)) ./ varPlus;   % [Nv,n]

            % Geyer: pair sums P_k = rho_2k + rho_2k+1, truncated at the first non-positive pair, made monotone
            K           = floor(n/2);
            P           = rho(:,1:2:2*K) + rho(:,2:2:2*K);  % [Nv,K]
            isPos       = cumprod(P > 0, 2);
            P           = cummin(P, 2);
            tau         = -1 + 2 .* sum(P .* isPos, 2);
            tau         = max(tau, 1/log10(n*M));           % guard as in Stan (tau >= 1/log10(N))
            N_eff       = n*M ./ tau;
            N_eff(~(varPlus > 0)) = NaN;
        end

        % split each chain in two halves (drop the first sample if Ns is odd)
        function xs = split_chains(x)
            n = size(x,2);
            if mod(n,2) == 1; x = x(:,2:end,:); n = n-1; end
            h  = n/2;
            xs = cat(3, x(:,1:h,:), x(:,h+1:end,:));
        end

        % reshape [Nv, ...] masked data to [x,y,z, ...] image (any number of trailing dims)
        function img = vec2image(v, mask)
            extraDims   = size(v,2:max(ndims(v),2));
            img         = utils.reshape_ND2image(reshape(v, size(v,1), []), mask);
            img         = reshape(img, [size(mask,1:3), extraDims]);
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
                    fitting.lb(idx) = min(fitting.lb(idx), double(min(xPosterior.(p)(:))));
                    fitting.ub(idx) = max(fitting.ub(idx), double(max(xPosterior.(p)(:))));
                end
            end
            out = res2out@mcmc(xPosterior,fitting,mask);

            fields  = fieldnames(xPosterior);
            Nrep    = size(xPosterior.(fields{1}),3);
            if isfield(diagnostics,'sampledParams'); sampled = diagnostics.sampledParams; else; sampled = fitting.modelParams; end

            % acceptance [x,y,z,Nblock,Nrep] and final u-space step size [x,y,z,Nrep] (sampled parameters only)
            out.diagnostics.acceptance          = mcmc_bayes.vec2image(diagnostics.acceptance,mask);
            out.diagnostics.acceptanceBlocks    = diagnostics.acceptanceBlocks;
            for kvar = 1:numel(sampled)
                out.diagnostics.stepSize.(sampled{kvar}) = mcmc_bayes.vec2image(permute(diagnostics.stepSize(:,kvar,:),[1 3 2]),mask);
            end

            % R-hat (repetition > 1) and ESS on native-space samples, per voxel per parameter
            chunk = 4096;   % voxels per chunk to bound the FFT memory
            for kvar = 1:numel(fields)
                xp      = xPosterior.(fields{kvar});
                Nv      = size(xp,1);
                essV    = zeros(Nv,1,'single');
                rhatV   = zeros(Nv,1,'single');
                for kc = 1:chunk:Nv
                    idx = kc:min(kc+chunk-1,Nv);
                    essV(idx) = mcmc_bayes.ess(xp(idx,:,:));
                    if Nrep > 1; rhatV(idx) = mcmc_bayes.rhat(xp(idx,:,:)); end
                end
                out.diagnostics.ess.(fields{kvar}) = utils.reshape_ND2image(essV,mask);
                if Nrep > 1; out.diagnostics.rhat.(fields{kvar}) = utils.reshape_ND2image(rhatV,mask); end
            end

            % test-only cache check
            if isfield(diagnostics,'cacheCheck'); out.diagnostics.cacheCheck = diagnostics.cacheCheck; end

            % hierarchical prior: hyperparameter samples and summaries (u space)
            if isfield(diagnostics,'hyper') && ~isempty(diagnostics.hyper)
                out.hyper = mcmc_bayes.hyper2out(diagnostics.hyper);
            end

            out.settings = diagnostics.settings;
        end

        % out.hyper from the hyperparameter samples, see the class header
        function H = hyper2out(hyper)
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

    end

end

% elementwise kernel of mcmc_bayes.transform_inverse_logjac_fused (GPU arrayfun)
function [x, logJ] = transform_kernel(u, code, lb, ub)
if code == 1
    % sigmoid: x = lb + (ub-lb) s(u), log|dx/du| = log(ub-lb) - softplus(u) - softplus(-u)
    %                                             = log(ub-lb) - |u| - 2 log1p(exp(-|u|))
    e       = exp(-abs(u));
    if u >= 0; s = 1/(1+e); else; s = e/(1+e); end
    x       = lb + (ub-lb)*s;
    logJ    = log(ub-lb) - abs(u) - 2*log1p(e);
elseif code == 2
    % log: x = exp(u), log|dx/du| = u
    x       = exp(u);
    logJ    = u;
else
    % linear
    x       = u;
    logJ    = u*0;
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
