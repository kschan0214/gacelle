
function [S0M, S0IEW] = model_BM_2T1_analytical3(TR,true_famp,f,R1f,R1r,kfr)

a = (-R1f-kfr);
b = (1-f).*kfr./f;
c = kfr;
d = -R1r-b;

% Eigenvalues
lambda1 = (((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 
lambda2 = (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 

ELambda1TR = exp(lambda1.*TR);
ELambda2TR = exp(lambda2.*TR);

% exponential matrix elements
e11 = (ELambda1TR.*(a-lambda2) - ELambda2TR.*(a-lambda1)) ./ (lambda1 - lambda2);
e12 = b.*(ELambda1TR-ELambda2TR)./ (lambda1 - lambda2);
e21 = c.*(ELambda1TR-ELambda2TR)./ (lambda1 - lambda2);
e22 = (ELambda1TR.*(d-lambda2) - ELambda2TR.*(d-lambda1)) ./ (lambda1 - lambda2);

% constant term
C = sin(true_famp)./((1-(e22+e11).*cos(true_famp)+(e11.*e22-e21.*e12).*cos(true_famp).^2).*(a.*d-b.*c));

ap = ((1-e22.*cos(true_famp)).*(e11-1) + e12.*e21.*cos(true_famp));
bp = ((1-e22.*cos(true_famp)).*e12 + (e12.*cos(true_famp)).*(e22-1));
cp = (e21.*cos(true_famp).*(e11-1) + (1-e11.*cos(true_famp)).*e21) ;
dp = (e21.*cos(true_famp).*e12 + (1-e11.*cos(true_famp)).*(e22-1));

S0IEW   = ((ap.*d-bp.*c).*R1f.*(1-f) + (-ap.*b+bp.*a).*R1r.*f).*C;
S0M     = ((cp.*d-c.*dp).*R1f.*(1-f) + (-cp.*b+a.*dp).*R1r.*f).*C;

end