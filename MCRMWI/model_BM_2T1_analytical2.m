
function [S0M, S0IEW] = model_BM_2T1_analytical2(TR,true_famp,f,R1f,R1r,kfr)

a = (-R1f-kfr);
b = (1-f).*kfr./f;
c = kfr;
d = -R1r-b;

% Eigenvalues
% lambda1 = (((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 
% lambda2 = (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2); 

% exponential matrix elements
e11 = (exp((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR).*(a-(((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2)) - exp((((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR).*(a-(((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2))) ./ ((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2) - (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2));
e12 = b.*(exp((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR)-exp((((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR))./ ((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2) - (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2));
e21 = c.*(exp((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR)-exp((((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR))./ ((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2) - (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2));
e22 = (exp((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR).*(d-(((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2)) - exp((((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2).*TR).*(d-(((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2))) ./ ((((a+d) + sqrt((a+d).^2-4.*(a.*d-b.*c)))./2) - (((a+d) - sqrt((a+d).^2-4.*(a.*d-b.*c)))./2));

% constant term
C = sin(true_famp)./((1-(e22+e11).*cos(true_famp)+(e11.*e22-e21.*e12).*cos(true_famp).^2).*(a.*d-b.*c));

S0IEW   = ((((1-e22.*cos(true_famp)).*(e11-1) + e12.*e21.*cos(true_famp)).*d-((1-e22.*cos(true_famp)).*e12 + (e12.*cos(true_famp)).*(e22-1)).*c).*R1f.*(1-f) + (-((1-e22.*cos(true_famp)).*(e11-1) + e12.*e21.*cos(true_famp)).*b+((1-e22.*cos(true_famp)).*e12 + (e12.*cos(true_famp)).*(e22-1)).*a).*R1r.*f).*C;
S0M     = (((e21.*cos(true_famp).*(e11-1) + (1-e11.*cos(true_famp)).*e21).*d-c.*(e21.*cos(true_famp).*e12 + (1-e11.*cos(true_famp)).*(e22-1))).*R1f.*(1-f) + (-(e21.*cos(true_famp).*(e11-1) + (1-e11.*cos(true_famp)).*e21).*b+a.*(e21.*cos(true_famp).*e12 + (1-e11.*cos(true_famp)).*(e22-1))).*R1r.*f).*C;

end