classdef NEXI < handle
    properties (Constant = true, Access = protected)
        
    end
    
    properties (GetAccess = public, SetAccess = protected)
        b;
        Delta;
        Nav;
    end
    
    properties (GetAccess = private, SetAccess = protected)
        
    end
    
    methods (Access = public)
        function this = NEXI(b, Delta, varargin)
%NEXI Exchange rate estimation using NEXI model
% smt = NEXI(b, Delta[, Nav])
%       output:
%           - smt: object of a fitting class
%
%       input:
%           - b: b-value [ms/um2]
%           - Delta: gradient seperation [ms]
%           - Nav (optional): # gradient direction for each b-shell
%
%       usage:
%           smt = NEXI(b, Delta, Nav);
%           [fa, Da, De, r] = smt.fit(S);
%           Sfit = smt.FWD([fa, Da, De, r]);
%
%           smt = NEXI(b, Delta, Nav);
%           [x_train, S_train] = smt.traindata(1e4);
%           pars0 = smt.likelihood(S, x_train, S_train);
%           [fa, Da, De, r] = smt.fit(S, pars0);
%           Sfit = smt.FWD([fa, Da, De, r]);
%
%  Authors: 
%  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
%  Copyright (c) 2023 Massachusetts General Hospital
%
%  Adapted from the code of
%  Dmitry Novikov (dmitry.novikov@nyulangone.org)
%  Copyright (c) 2023 New York University
            
            this.b = b;
            this.Delta = Delta;
            if nargin > 2
                this.Nav = varargin{1};
            else
                this.Nav = ones(size(b));
            end
        end
        
        function [fa, Da, De, ra]  = mcmc(this, y, N)
            fa   = zeros(N,1);
            Da   = zeros(N,1);
            De   = zeros(N,1);
            ra   = zeros(N,1);
            
            for i = 1:N
                [fa(i), Da(i), De(i), ra(i)] = this.fit(y);
            end
            idx = kmeans([ra, fa], 2);
            I = idx == mode(idx);
            fa = mean(fa(I));
            Da = mean(Da(I));
            De = mean(De(I));
            ra = mean(ra(I));
        end
        
        function [fa, Da, De, ra]  = fit(this, y, varargin)
%FIT Estimate the exchange rate from multi-shell data.
% [fa, Da, De, ra]  = fit(y)
%       output:
%           - fa: intra-neurite volume fraction
%           - Da: intra-neurite diffusivity [um2/ms]
%           - De: extra-cellular diffusivity [um2/ms]
%           - ra: exchange rate [1/ms]
%
%       input:
%           - y:      powder-averaged diffusion-weighted signal,
%                     normalized to y(b=0) = 1. If the dot comparment
%                     cannot be ignored, it need to be subtracted from y prior to fitting.  
%
%  Authors: 
%  Hong-Hsi Lee (hlee84@mgh.harvard.edu)
%  Copyright (c) 2023 Massachusetts General Hospital
%
%  Adapted from the code of
%  Dmitry Novikov (dmitry.novikov@nyulangone.org)
%  Copyright (c) 2023 New York University

            options = optimset('lsqnonlin');
            options = optimset(options,'Jacobian','on','TolFun',1e-12,'TolX',1e-12,'MaxIter',1e5,'Display','off');
            
            % Fitting parameters: [fa, Da, De, ra, fdot]
            if nargin < 3
                start = [0.01+rand*0.99, 3*rand, rand, 1/(1+99*rand)];  % initial values
                start(3) = start(2)*start(3);
            else
                start = varargin{1};
                start = start(:).';
            end
            lb = [0.01, 0, 0, 0];       % lower bound
            ub = [0.99, 3, 3, 1];       % upper bound
            
            % A = [0 -1 1 0]; B = 0;      % De <= Da
            % pars = lsqnonlin(@(x)this.residuals(x, y),...
            %     start,lb,ub,A,B,[],[],[],options);

            sqwt = sqrt(this.Nav(:));
            pars = lsqnonlin(@(x)this.residuals(x, y, sqwt),...
                start,lb,ub,options);

            fa   = pars(1);
            Da   = pars(2);
            De   = pars(3);
            ra   = pars(4);
        end
        
        function [E, J] = residuals(this, pars, y, sqwt)
            [shat, dshat] = this.FWD(pars); 
            E = shat(:) - y(:);
            J = dshat;
            E = E .* sqwt;
            J = J .* sqwt;
        end
        
        function [s, ds] = FWD(this, pars)
            fa   = pars(1);
            Da   = pars(2);
            De   = pars(3);
            ra   = pars(4);
            
            % Forward model
            s = this.S(fa, Da, De, ra);
                        
            % Jacobian
            if nargout > 1
                h = 1e-6;
                
                S1 = this.S(fa+h, Da, De, ra);
                % S2 = this.S(fa-h, Da, De, ra);
                dS_dfa = (S1-s)/h;
                % dS_dfa = (S1-S2)/h/2;
                
                S1 = this.S(fa, Da+h, De, ra);
                % S2 = this.S(fa, Da-h, De, ra);
                dS_dDa = (S1-s)/h;
                % dS_dDa = (S1-S2)/h/2;
                
                S1 = this.S(fa, Da, De+h, ra);
                % S2 = this.S(fa, Da, De-h, ra);
                dS_dDe = (S1-s)/h;
                % dS_dDe = (S1-S2)/h/2;
                
                S1 = this.S(fa, Da, De, ra+h);
                % S2 = this.S(fa, Da, De, ra-h);
                dS_dra = (S1-s)/h;
                % dS_dra = (S1-S2)/h/2;
                
%                 myfun_fa = @(x) this.dM_dfa(x, this.b, this.Delta, fa, Da, De, ra);
%                 dS_dfa = integral(myfun_fa, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);

%                 myfun_Da = @(x) this.dM_dDa(x, this.b, this.Delta, fa, Da, De, ra);
%                 dS_dDa = integral(myfun_Da, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
% 
%                 myfun_De = @(x) this.dM_dDe(x, this.b, this.Delta, fa, Da, De, ra);
%                 dS_dDe = integral(myfun_De, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
% 
%                 myfun_ra = @(x) this.dM_dra(x, this.b, this.Delta, fa, Da, De, ra);
%                 dS_dra = integral(myfun_ra, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
                
                ds = [dS_dfa, dS_dDa, dS_dDe, dS_dra];
            end
        end
        
        function [x_train, S_train, intervals] = traindata(this, N_samples, varargin)
            if nargin < 3
                intervals = [0.01 0.99  ;   % fa
                    1.5 3               ;   % Da
                    0.5 1.5             ;   % De
                    1 100               ];  % residence time = (1-fa)/r
            else
                intervals = varargin{1};
            end
            
            % batch size can be modified according to available hardware
            batch_size = 1e3;
            reps = ceil(N_samples/batch_size);
            x_train = zeros(size(intervals,1),batch_size,reps);
            S_train = zeros(numel(this.b),batch_size,reps);
            for k = 1:reps
                % generate random parameter guesses and construct batch for NN signal evaluation
                pars = intervals(:,1) + diff(intervals,[],2).*rand(size(intervals,1),batch_size);
                % pars(3,:) = pars(2,:).*pars(3,:);
                pars(4,:) = 1./pars(4,:).*(1-pars(1,:));

                % NEXI Kärger signal evaluation
                S = zeros(numel(this.b),batch_size);
                for j = 1:batch_size
                    S(:,j) = this.S(pars(1,j), pars(2,j), pars(3,j), pars(4,j));
                end
                
                % remaining signals (dot, soma)
                x_train(:,:,k) = pars;
                S_train(:,:,k) = S;
            end
            % intervals(3,:) = intervals(2,:).*intervals(3,:);
            intervals(4,:) = (1-intervals(1,end:-1:1))./intervals(4,end:-1:1);
        end

        function [pars_best, sse_best] = likelihood(this, S0, x_train, S_train)
            wt = this.Nav(:);
            % batch size can be modified according to available hardware
            [Nx, ~, reps] = size(x_train);
            [~, Nv] = size(S0);
            pars_best = zeros(Nx,Nv);
            sse_best  = inf(1, Nv);
            for k = 1:reps
                pars = x_train(:,:,k);
                S    = S_train(:,:,k);
                for i = 1:Nv
                    S0i = S0(:,i);

                    % scale generated signals (fit S0) to input signal
                    sse = sum(wt.*(S0i - (S0i'*S)./dot(S,S).*S).^2);

                    % store best encountered parameter combination
                    [sse_new,best_index] = min(sse);
                    if sse_new<sse_best(i)
                        sse_best(i)    = sse_new;
                        pars_best(:,i) = pars(:,best_index);
                    end
                end
            end
        end
        
        function S = S(this, fa, Da, De, ra)
            Da = this.b*Da;
            De = this.b*De;
            ra = ra*this.Delta;
            re = ra*fa/(1-fa);
            myfun = @(x) this.M(x, fa, Da, De, ra, re);
            S = integral(myfun, 0, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
        end

        function F = NEXIsh(this, fa, Da, De, ra, lmax)
            Nb = numel(this.b);
            Nl = ceil((lmax+1)/2);
            F = zeros(Nl,Nb);
            for i = 1:Nl
                li = (i-1)*2;
                F(i,:) = this.IntegralLegendreNEXI(li, fa, Da, De, ra) * sqrt((2*li+1)*pi);
            end
        end
        
        function I = IntegralLegendreNEXI(this, n, fa, Da, De, ra)
            Da = this.b*Da;
            De = this.b*De;
            ra = ra*this.Delta;
            re = ra*fa/(1-fa);
            if mod(n,2) == 1    % n is odd
                I = 0;
            else                % n is even
                I = 0;
                for k = 0:floor(n/2)
                    I = I + (-1)^k * nchoosek(n,k) * nchoosek(2*n-2*k,n) * ...
                        this.IntegralPolyNEXI(n-2*k, fa, Da, De, ra, re);
                end
                I = I / 2^n;
            end
        end
        
        function I = IntegralPolyNEXI(this, n, fa, Da, De, ra, re)
            if mod(n,2) == 1
                I = 0;
            else
                f = @(x) x.^n .* this.M(x, fa, Da, De, ra, re);
                I = integral(@(x)f(x),-1, 1, 'AbsTol', 1e-14, 'ArrayValued', true);
            end
        end

        function S = SHconv(this, F, pl, theta)
            lmax = 2*size(pl,1)-2;
            Sl = F.*pl;
            dirs = [zeros(size(theta)), theta];
            Y_N = getSH(lmax, dirs, 'real');
            I = this.findm0(lmax);
            Y_N = Y_N(:,I);
            S = Y_N*Sl;
            S = S.';
        end

        function pl = WatsonSH(this, kappa, lmax)
            f = @(x,k,l) exp(k*x.^2) .* this.mylegendreP(l,x);
            pl_theory = @(k,l) 1/2./this.myhypergeom(k) .* integral(@(x)f(x,k,l),-1,1);
            Nk = numel(kappa);
            Nl = ceil((lmax+1)/2);
            pl = zeros(Nl,Nk);
            pl(1,:) = 1;
            for j = 1:Nk
                for i = 1:Nl
                    if ~isinf(kappa(j))
                        pl(i,j) = pl_theory(kappa(j),2*(i-1));
                    else
                        pl(i,j) = 1;
                    end
                end
            end
        end

        function ang = WatsonAng(this, kappa)
            pl = this.WatsonSHexact(kappa);
            p2 = pl(2,:); p2 = p2(:);
            ang = acos( sqrt( (2*p2+1)/3 ) );
        end
        
    end
    
    methods(Static)
        function M = M(x, fa, Da, d2, r1, r2)
            % d1 = b*Da*x^2
            % d2 = b*De
            % r1 = ra*t
            % r2 = ra*t*fa/(1-fa)
            d1 = Da.*x.^2;
            l1 = (r1+r2+d1+d2)/2;
            l2 = sqrt( (r1-r2+d1-d2).^2 + 4*r1.*r2 )/2;
            lm = l1-l2;
            Pp = (fa*d1 + (1-fa)*d2 - lm)./(l2*2);
            M = Pp.*exp(-(l1+l2)) + (1-Pp).*exp(-lm); 
        end
        
        

        function I = findm0(Nord)
            I = false((Nord+1)^2,1);
            for i = 1:ceil((Nord+1)/2)
                l = 2*(i-1);
                I(l^2+l+1) = true;
            end
        end

        function y = mylegendreP(l,x)
            switch l
                case 0
                    y = 1;
                case 2
                    y = (3*x.^2)/2 - 1/2;
                case 4
                    y = (35*x.^4)/8 - (15*x.^2)/4 + 3/8;
                case 6
                    y = (231*x.^6)/16 - (315*x.^4)/16 + (105*x.^2)/16 - 5/16;
                case 8
                    y = (6435*x.^8)/128 - (3003*x.^6)/32 + (3465*x.^4)/64 - (315*x.^2)/32 + 35/128;
                case 10
                    y = (46189*x.^10)/256 - (109395*x.^8)/256 + (45045*x.^6)/128 - (15015*x.^4)/128 + (3465*x.^2)/256 - 63/256;
                case 12
                    y = (676039*x.^12)/1024 - (969969*x.^10)/512 + (2078505*x.^8)/1024 - (255255*x.^6)/256 + (225225*x.^4)/1024 - (9009*x.^2)/512 + 231/1024;
                case 14
                    y = (5014575*x.^14)/2048 - (16900975*x.^12)/2048 + (22309287*x.^10)/2048 - (14549535*x.^8)/2048 + (4849845*x.^6)/2048 - (765765*x.^4)/2048 + (45045*x.^2)/2048 - 429/2048;
                case 16
                    y = (300540195*x.^16)/32768 - (145422675*x.^14)/4096 + (456326325*x.^12)/8192 - (185910725*x.^10)/4096 + (334639305*x.^8)/16384 - (20369349*x.^6)/4096 + (4849845*x.^4)/8192 - (109395*x.^2)/4096 + 6435/32768;
                case 18
                    y = (2268783825*x.^18)/65536 - (9917826435*x.^16)/65536 + (4508102925*x.^14)/16384 - (4411154475*x.^12)/16384 + (5019589575*x.^10)/32768 - (1673196525*x.^8)/32768 + (156165009*x.^6)/16384 - (14549535*x.^4)/16384 + (2078505*x.^2)/65536 - 12155/65536;
                case 20
                    y = (34461632205*x.^20)/262144 - (83945001525*x.^18)/131072 + (347123925225*x.^16)/262144 - (49589132175*x.^14)/32768 + (136745788725*x.^12)/131072 - (29113619535*x.^10)/65536 + (15058768725*x.^8)/131072 - (557732175*x.^6)/32768 + (334639305*x.^4)/262144 - (4849845*x.^2)/131072 + 46189/262144;
                
                otherwise
                    y = legendreP(l,x);
            end
        end

        function pl = WatsonSHexact(k)
            k = k(:).';
            p2 = 1/4*(3./sqrt(k)./dawson(sqrt(k)) -2 -3./k);
            p4 = 1/32./k.^2.*(105 + 12*k.*(5+k) + 5*sqrt(k).*(2*k-21)./dawson(sqrt(k)));
            pl = [ones(1,numel(k)); p2; p4];
        end

        function y = myhypergeom(x)
            y = sqrt(pi)/2 * (-x).^(-1/2) .* (1-gammainc(-x, 1/2, 'upper'));
            y = real(y);
        end
        
%         function dM_dfa = dM_dfa(x, b, t, fa, Da, De, ra)
%             dM_dfa = exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + 1).*((2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2)./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) + (ra.*t)./(2.*(fa - 1)) - (fa.*ra.*t)./(2.*(fa - 1).^2)) - exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(((2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2)./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - De.*b + (ra.*t)./(2.*(fa - 1)) + Da.*b.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1).^2))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + ((2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2))) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*((2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2)./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - De.*b + (ra.*t)./(2.*(fa - 1)) + Da.*b.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1).^2)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2)) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*((2.*((ra.*t)./(fa - 1) - (fa.*ra.*t)./(fa - 1).^2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (4.*ra.^2.*t.^2)./(fa - 1) + (4.*fa.*ra.^2.*t.^2)./(fa - 1).^2)./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - (ra.*t)./(2.*(fa - 1)) + (fa.*ra.*t)./(2.*(fa - 1).^2)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2);
%         end
%         
%         function dM_dDa = dM_dDa(x, b, t, fa, Da, De, ra)
%             dM_dDa = (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(b.*fa.*x.^2 - (b.*x.^2)./2 + (b.*x.^2.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) - exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + 1).*((b.*x.^2)./2 - (b.*x.^2.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))) - exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*((b.*fa.*x.^2 - (b.*x.^2)./2 + (b.*x.^2.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + (b.*x.^2.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2)) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*((b.*x.^2)./2 + (b.*x.^2.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + (b.*x.^2.*exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2);
%         end
%         
%         function dM_dDe = dM_dDe(x, b, t, fa, Da, De, ra)
%             dM_dDe = exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*((b./2 + b.*(fa - 1) + (b.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + (b.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2)) - (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(b./2 + b.*(fa - 1) + (b.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) - exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(b./2 + (b.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))).*(((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + 1) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(b./2 - (b.*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2))).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) - (b.*exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2);
%         end
%         
%         function dM_dra = dM_dra(x, b, t, fa, Da, De, ra)
%             dM_dra = (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*((2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1))./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - t./2 + (fa.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) - exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(((2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1))./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - t./2 + (fa.*t)./(2.*(fa - 1)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + ((2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2))) + exp(((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (De.*b)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 + (fa.*ra.*t)./(2.*(fa - 1))).*(((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1)))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + 1).*((2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1))./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - t./2 + (fa.*t)./(2.*(fa - 1))) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(t./2 + (2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1))./(4.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)) - (fa.*t)./(2.*(fa - 1))).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2) + (exp((fa.*ra.*t)./(2.*(fa - 1)) - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 - (ra.*t)./2 - (Da.*b.*x.^2)./2 - (De.*b)./2).*(2.*(t + (fa.*t)./(fa - 1)).*(Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)) - (8.*fa.*ra.*t.^2)./(fa - 1)).*((De.*b)./2 - ((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(1./2)./2 + (ra.*t)./2 + De.*b.*(fa - 1) + (Da.*b.*x.^2)./2 - Da.*b.*fa.*x.^2 - (fa.*ra.*t)./(2.*(fa - 1))))./(2.*((Da.*b.*x.^2 - De.*b + ra.*t + (fa.*ra.*t)./(fa - 1)).^2 - (4.*fa.*ra.^2.*t.^2)./(fa - 1)).^(3./2));
%         end
        
    end
end
