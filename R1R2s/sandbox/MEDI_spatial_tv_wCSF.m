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
% Date modified: 19 July 2025
%
function loss_reg = MEDI_spatial_tv_wCSF(parameters, lambdaPhi, maskBrain, lambdaChi, maskEdge, lambdaCSF, maskCSF, TVmode, voxelSize, mask_tissue)

% TV on Phi
NsampleBrain = numel(maskBrain(maskBrain ~= 0));
if isequal(size(parameters.dpini,1:3), size(maskBrain,1:3))
    cost = reg_TV(parameters.dpini,maskBrain,TVmode,voxelSize);
else
    % if the dims are different then reshape the parameter map
    cost = reg_TV(utils.reshape_GD2ND(parameters.dpini,maskBrain),maskBrain,TVmode,voxelSize);
end
loss_phi = sum(abs(cost),"all")/NsampleBrain *lambdaPhi;

% |Chi| on CSF
NsampleCSF = numel(maskCSF(maskCSF ~= 0));
if isequal(size(parameters.Chi,1:3), size(maskBrain,1:3))
    cost = (real(parameters.Chi) - sum(real(parameters.Chi).*maskCSF,"all")/NsampleCSF).*maskCSF;
else
    % if the dims are different then reshape the parameter map
    % cost = (utils.reshape_GD2ND(real(parameters.Chi),maskBrain) - sum(real(parameters.Chi).*maskCSF,"all")/NsampleCSF).*maskCSF;
    cost = (utils.reshape_GD2ND(real(parameters.Chi),maskBrain)).*maskCSF;
end
loss_CSF = sum(abs(cost),"all")/NsampleCSF *lambdaCSF;

% TV on Chi, excluded sharp edge
NsampleEdge = numel(maskEdge(maskEdge ~= 0));
if isequal(size(parameters.Chi,1:3), size(maskEdge,1:3))
    cost = reg_TV(real(parameters.Chi),maskEdge,TVmode,voxelSize); % no need to use real but I use it anyway
else
    % if the dims are different then reshape the parameter map
    cost = reg_TV(utils.reshape_GD2ND(real(parameters.Chi),maskEdge),maskEdge,TVmode,voxelSize);
end
loss_chi = sum(abs(cost),"all")/NsampleEdge *lambdaChi;

% TV on Dr1pos, excluded sharp edge
if isfield(parameters,'Dr1pos')
mask_WM = mask_tissue(:,:,:,1);
NsampleWM = numel(mask_WM(mask_WM ~= 0));
if isequal(size(parameters.Dr1pos,1:3), size(maskBrain,1:3))
    cost = real(parameters.Dr1pos).*mask_WM;
else
    % if the dims are different then reshape the parameter map
    % cost = (utils.reshape_GD2ND(real(parameters.Chi),maskBrain) - sum(real(parameters.Chi).*maskCSF,"all")/NsampleCSF).*maskCSF;
    cost = (utils.reshape_GD2ND(real(parameters.Dr1pos),maskBrain)).*mask_WM;
end
    loss_Dr1pos_WM = sum(abs(cost),"all")/NsampleWM *lambdaChi/100;
else
    loss_Dr1pos_WM = 0;
end
% % TV on Dr1pos, excluded sharp edge
% if isequal(size(parameters.Dr2pos,1:3), size(maskBrain,1:3))
%     cost = real(parameters.Dr2pos).*mask_WM;
% else
%     % if the dims are different then reshape the parameter map
%     % cost = (utils.reshape_GD2ND(real(parameters.Chi),maskBrain) - sum(real(parameters.Chi).*maskCSF,"all")/NsampleCSF).*maskCSF;
%     cost = (utils.reshape_GD2ND(real(parameters.Dr2pos),maskBrain)).*mask_WM;
% end
% loss_Dr1pos_WM = sum(abs(cost),"all")/NsampleWM *lambdaChi/100;

% if isfield(parameters,'Dr1pos')
%     if isequal(size(parameters.Dr1pos,1:3), size(maskEdge,1:3))
%         cost = reg_TV(real(parameters.Dr1pos),maskEdge,TVmode,voxelSize); % no need to use real but I use it anyway
%     else
%         % if the dims are different then reshape the parameter map
%         cost = reg_TV(utils.reshape_GD2ND(real(parameters.Dr1pos),maskEdge),maskEdge,TVmode,voxelSize);
%     end
%     loss_Dr1pos = sum(abs(cost),"all")/NsampleEdge *lambdaChi/1000;
% else
%     loss_Dr1pos = 0;
% end
% % TV on Dr2pos, excluded sharp edge
% if isfield(parameters,'theta')
%     if isequal(size(parameters.theta,1:3), size(maskEdge,1:3))
%         cost = reg_TV(real(parameters.theta),maskEdge,TVmode,voxelSize); % no need to use real but I use it anyway
%     else
%         % if the dims are different then reshape the parameter map
%         cost = reg_TV(utils.reshape_GD2ND(real(parameters.theta),maskEdge),maskEdge,TVmode,voxelSize);
%     end
%     loss_theta = sum(abs(cost),"all")/NsampleEdge *lambdaChi/10;
% else
%     loss_theta = 0;
% end
% % TV on Dr2pos, excluded sharp edge
% if isfield(parameters,'Dr2pos')
%     if isequal(size(parameters.Dr2pos,1:3), size(maskEdge,1:3))
%         cost = reg_TV(real(parameters.Dr2pos),maskEdge,TVmode,voxelSize); % no need to use real but I use it anyway
%     else
%         % if the dims are different then reshape the parameter map
%         cost = reg_TV(utils.reshape_GD2ND(real(parameters.Dr2pos),maskEdge),maskEdge,TVmode,voxelSize);
%     end
%     loss_Dr2pos = sum(abs(cost),"all")/NsampleEdge *lambdaChi/10;
% else
%     loss_Dr2pos = 0;
% end
loss_reg = loss_phi + loss_CSF + loss_chi + loss_Dr1pos_WM;
% loss_reg = loss_phi + loss_CSF + loss_Dr1pos + loss_chi + loss_theta;
% loss_reg = loss_phi + loss_CSF + loss_Dr1pos + loss_Dr2pos;
% loss_reg = loss_phi + loss_chi + loss_CSF + loss_Dr1pos;% + loss_Dr2pos;

end

% compute the cost of Total variation regularisation
function cost = reg_TV(img,mask,TVmode,voxelSize)
    % voxel_size = [1 1 1];
    % Vr      = 1./sqrt(abs(mask.*askadam.gradient_operator(img,voxel_size)).^2+eps);
    cost = sum(abs(mask.*gradient_operator(img,voxelSize,TVmode)),4);
    % cost = sqrt(sum(abs(mask.*gradient_operator(img,voxelSize,TVmode)).^2,4));

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