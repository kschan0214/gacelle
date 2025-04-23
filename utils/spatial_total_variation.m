% loss_reg = spatial_total_variation(parameters, mask, lambda, regmap, TVmode, voxelSize)
%
% Input
% --------------
% parameters    : structure variable containing the model parameters (same as forward model function)
% mask          : 3D mask
% lambda        : 1D cell array of regularisation parameter
% regmap        : 1D cell array of the names of the parameter maps where TV applies to
% TVmode        : '2D' or '3D'
% voxelSize     : 1x2 ('2D') or 1x3 ('3D') numeric array of the voxel size  in mm
%
% Output
% --------------
% loss_reg      : regularisation loss
% 
% Description:  compute the loss value of applying 2D (or 3D) spatial TV regularisation on the model parameter map(s)

% Kwok-Shing Chan @ MGH
% kchan2@mgh.harvard.edu
%
% Date created: 11 April 2025 
% Date modified: 
%
function loss_reg = spatial_total_variation(parameters, mask, lambda, regmap, TVmode, voxelSize)

% regularisation term
loss_reg = 0;
if lambda{1} > 0
    Nsample     = numel(mask(mask ~= 0));

    for kreg = 1:numel(lambda)
        cost        = reg_TV(utils.reshape_GD2ND(parameters.(regmap{kreg}),mask),mask,TVmode,voxelSize);
        loss_reg    = sum(abs(cost),"all")/Nsample *lambda{kreg} + loss_reg;
    end
end

end

% compute the cost of Total variation regularisation
function cost = reg_TV(img,mask,TVmode,voxelSize)
    % voxel_size = [1 1 1];
    % Vr      = 1./sqrt(abs(mask.*askadam.gradient_operator(img,voxel_size)).^2+eps);
    cost = sum(abs(mask.*gradient_operator(img,voxelSize,TVmode)),4);

    % cost    = divergence_operator(mask.*(Vr.*(mask.*askadam.gradient_operator(img,voxel_size))),voxel_size);
end

% TV regularisation
function G = gradient_operator(img,voxel_size,TVmode)
    Dx = circshift(img,-1,1) - img;     % gradient in x
    Dy = circshift(img,-1,2) - img;     % gradient in y
    switch TVmode
        case '2D'
            G = cat(4,Dx/voxel_size(1),Dy/voxel_size(2));   % concatenate Dtheta/Dx and Dtheta/Dy
        case '3D'
            Dz = circshift(img,-1,3) - img; % gradient in z
            G = cat(4,Dx/voxel_size(1),Dy/voxel_size(2),Dz/voxel_size(3));
    end
    
end

function div = divergence_operator(G,voxel_size)

    G_x = G(:,:,:,1);
    G_y = G(:,:,:,2);
    G_z = G(:,:,:,3);
    
    [Mx, My, Mz] = size(G_x);
    
    Dx = [G_x(1:end-1,:,:); zeros(1,My,Mz)]...
        - [zeros(1,My,Mz); G_x(1:end-1,:,:)];
    
    Dy = [G_y(:,1:end-1,:), zeros(Mx,1,Mz)]...
        - [zeros(Mx,1,Mz), G_y(:,1:end-1,:)];
    
    Dz = cat(3, G_z(:,:,1:end-1), zeros(Mx,My,1))...
        - cat(3, zeros(Mx,My,1), G_z(:,:,1:end-1));
    
    div = -( Dx/voxel_size(1) + Dy/voxel_size(2) + Dz/voxel_size(3) );

end