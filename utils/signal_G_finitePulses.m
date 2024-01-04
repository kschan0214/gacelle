function S = signal_G_finitePulses(pars,info,epsilon)

fa      = pars(1);   % volume fraction a
Da      = pars(2);   % diffusivity a
De      = pars(3);   % diffusivity e
ra      = pars(4);   % exchange rate from a to e
% epsilon = pars(5);   % angle between g and n (cos theta)

fb = 1-fa;
rb = ra*fa/(fb+1e-4);

b       = info.b;
Delta   = info.Delta;
delta   = info.delta;

% can solve differential equation for all eps simultanously
% D matrix
% [eps,w] = fun.utility.legpts(20,[0 1]);
D = zeros(2*length(w),1);
% D(1:2:end) = Da*eps.^2;
D(1:2:end)  = Da*epsilon.^2;
D(2:2:end)  = De;
D           = diag(D);
% R matrix
R = zeros(2*length(w));
for j = 1:2:2*length(w)
    R(j:j+1,j:j+1) = [-ra  rb
                       ra -rb];
end
f = zeros(2*length(w),1);
f(1:2:end) = fa;
f(2:2:end) = fb;

if numel(delta)==1
    delta = delta*ones(length(b),1);
    Delta = Delta*ones(length(b),1);
end
q = sqrt(b./(Delta-delta/3));

S = zeros(length(b),length(w));
for i = 1:length(b)
    St = fun.karger.finitePulses(f,D,R,delta(i),Delta(i),q(i));
    S(i,:) = St(1:2:end)+St(2:2:end);
end
% S = S*w';


end