function s = model_GRE_steadystate_Bloch(m0,t1,tr,alpha)
    E1 = exp(-tr./t1);
    s = (m0.*(1-E1).*sin(alpha))./(1-E1.*cos(alpha));
end