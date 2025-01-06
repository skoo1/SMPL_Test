% 1) Load model data
data = load('model.mat');
% fid = fopen('model.json', 'r');
% raw = fread(fid, '*char')';
% fclose(fid);
% data = jsondecode(raw);

% 2) Build the SMPL model struct
SMPL_Model = buildSMPLModel(data);

% 3) Load motion data
fid = fopen('walking_20240125.json', 'r');
raw = fread(fid, '*char')';
fclose(fid);
motionData = jsondecode(raw);

% 4) Example usage: iterate through frames
numFrame = size(motionData, 1);
betas = zeros(300, 1);  % Example betas

figureHandle = figure(2);

for i = 1:numFrame
    % Read the motion for frame i
    temp_data = reshape(motionData(i,:), [], 1);
    
    trans = temp_data(1:3);
    thetas = temp_data(4:end) + 1e-5;  % offset to avoid exact zero angles
    
    % Compute SMPL vertices and joints
    [verts, joints] = SMPL_Calc(thetas, betas, SMPL_Model);

    plotVertices2(verts, SMPL_Model.faces, SMPL_Model.N, trans);
    % plotVertices2(frame1verts{:,:}, SMPL_Model.faces, SMPL_Model.N, trans);
    
    pause(0.001);
end

%% =========================================================================
function SMPL_Model = buildSMPLModel(data)
    % buildSMPLModel  Builds and returns the SMPL model struct from 'model.mat' data

    % Unpack from 'data' (as loaded by load('model.mat'))
    v_template     = data.v_template;
    shapedirs      = data.shapedirs;
    posedirs       = data.posedirs;
    J_regressor    = data.J_regressor;
    kintree_table  = data.kintree_table;
    weights        = data.weights;
    f              = data.f;
    
    % Basic sizes
    N = size(v_template, 1);
    K = size(weights, 2);
    
    % SMPL uses 0-based indexing, MATLAB is 1-based 
    % => shift faces and kintree_table
    faces = f + 1;
    kintree_table = kintree_table + 1;
    
    % Flatten shapedirs and posedirs into vectors
    shapedirs_vec = reshape(shapedirs, N * 3, 300);
    posedirs_vec  = reshape(posedirs,  N * 3, 93);
    
    % Add zero row for the homogeneous coordinate
    shapedirs_vec = [shapedirs_vec; zeros(N, 300)];
    posedirs_vec  = [posedirs_vec;  zeros(N, 93)];

    % Flatten template vertices into (4N, 1) with homogeneous coords
    v_tem_vec = reshape(v_template, N*3, 1);
    v_tem_vec = [v_tem_vec; ones(N,1)];
    
    % Build joint regressor for 4N vector
    % We replicate J_regressor 3 times + zero block for the homogeneous row
    zeroBlock = sparse(K, N);
    J_reg_vec = [J_regressor, zeroBlock,   zeroBlock,   zeroBlock;
                 zeroBlock,   J_regressor, zeroBlock,   zeroBlock;
                 zeroBlock,   zeroBlock,   J_regressor, zeroBlock;
                 zeroBlock,   zeroBlock,   zeroBlock,   zeroBlock];
    
    % Pack everything into a struct
    SMPL_Model.N                = N;
    SMPL_Model.K                = K;
    SMPL_Model.faces            = faces;
    SMPL_Model.kintree_table    = kintree_table;
    SMPL_Model.weights          = weights;
    SMPL_Model.shapedirs_vec    = shapedirs_vec;
    SMPL_Model.posedirs_vec     = posedirs_vec;
    SMPL_Model.v_tem_vec        = v_tem_vec;
    SMPL_Model.J_reg_vec        = J_reg_vec;
end

%% =========================================================================
function [verts, joints] = SMPL_Calc(thetas, betas, SMPL_Model)
    % SMPL_Calc  High-level SMPL pipeline
    
    % 1) shape blend
    [j_shaped, v_shaped] = shapeblend(betas, SMPL_Model);
    
    % 2) compute joint transforms
    [global_transform, global_transform_remove] = ...
        transforms(thetas, j_shaped, SMPL_Model);
    
    % 3) pose blend
    v_shaped2 = poseblend(thetas, betas, v_shaped, SMPL_Model);
    
    % 4) pose rotation
    v_rot = poserot(global_transform_remove, v_shaped2, SMPL_Model);

    % Output
    verts  = v_rot;
    joints = reshape(global_transform(:, 1:3, 4), [], 1);
end

%% =========================================================================
function [j_shaped, v_shaped, J_DjshDbetas, J_DvshDbetas] = shapeblend(betas, SMPL_Model)
    % shapeblend  Apply shape blendshapes
    
    shapedirs_vec = SMPL_Model.shapedirs_vec;
    v_tem_vec     = SMPL_Model.v_tem_vec;
    J_reg_vec     = SMPL_Model.J_reg_vec;
    
    % v_shaped: template + linear combination of shape dirs
    v_shapeblend = shapedirs_vec * betas;
    v_shaped     = v_tem_vec + v_shapeblend;
    
    % j_shaped: regressed from v_shaped
    j_shaped = J_reg_vec * v_shaped;
    
    % Optionally output some Jacobians (unused in this example)
    if nargout > 2
        J_DvshDbetas = shapedirs_vec;
        J_DjshDbetas = J_reg_vec * shapedirs_vec;
    end
end

%% =========================================================================
function [global_transform, global_transform_remove] = transforms(thetas, j_shaped, SMPL_Model)
    % transforms  Compute the per-joint global transforms
    
    kintree_table = SMPL_Model.kintree_table;
    K             = SMPL_Model.K;
    
    global_transform         = zeros(K, 4, 4);
    global_transform_remove  = zeros(K, 4, 4);
    
    for i = 1:K
        % Current rotation
        rotmat = so3exp(thetas(3*i-2 : 3*i));
        
        if i == 1
            % Root joint
            global_transform(i, 1:3, 1:3) = rotmat;
            global_transform(i, 1, 4)     = j_shaped(i);
            global_transform(i, 2, 4)     = j_shaped(i + K);
            global_transform(i, 3, 4)     = j_shaped(i + 2*K);
            global_transform(i, 4, 4)     = 1;
        else
            parentIdx = kintree_table(1, i);
            trans = [
                j_shaped(i         ) - j_shaped(parentIdx         );
                j_shaped(i +   K   ) - j_shaped(parentIdx +   K   );
                j_shaped(i + 2*K   ) - j_shaped(parentIdx + 2*K   );
                0
            ];
            localTrans = eye(4);
            localTrans(1:3, 1:3) = rotmat;
            localTrans(1:3, 4)   = trans(1:3);
            
            global_transform(i, :, :) = squeeze(global_transform(parentIdx, :, :)) * localTrans;
        end
        
        % Remove the "joint location" so it rotates around the joint
        jHere = [ j_shaped(i), j_shaped(i + K), j_shaped(i + 2*K), 0 ].';
        fx = squeeze(global_transform(i, :, :)) * jHere;
        
        pack = zeros(4);
        pack(1:3, 4) = fx(1:3);
        
        global_transform_remove(i, :, :) = squeeze(global_transform(i, :, :)) - pack;
    end
end

%% =========================================================================
function v_rot = poserot(global_transform_remove, v_shaped, SMPL_Model)
    % poserot_matlab 
    %   Blend all the per-joint transformations based on vertex skinning weights
    %   This version explicitly flattens each 4x4 in column-major order
    %
    % Inputs:
    %   - global_transform_remove: [K, 4, 4]
    %   - v_shaped: [4N x 1]
    %   - SMPL_Model: struct with fields
    %       .weights => [N x K]
    %       .N, .K   => int
    %
    % Output:
    %   - v_rot => [N x 4]

    weights = SMPL_Model.weights; % [N x K]
    N = SMPL_Model.N;
    K = SMPL_Model.K;

    % 1) Flatten each 4x4 into 16 in **column-major** order
    %    global_transform_remove: shape (K,4,4)
    %    => global_transform_remove_vec: shape (K,16)
    global_transform_remove_vec = zeros(K, 16);
    for k = 1:K
        M = squeeze(global_transform_remove(k, :, :));  % M is [4 x 4]
        % M(:)는 MATLAB에서 col-major 순서로 [M(1,1), M(2,1), M(3,1), M(4,1), M(1,2), ...]가 나옴
        global_transform_remove_vec(k, :) = M(:).';
    end

    % 2) Weighted sum => [N x 16]
    coefficients = weights * global_transform_remove_vec;

    % 3) reshape to [N,4,4]
    coefficients_3d = reshape(coefficients, [N,4,4]);

    % 4) v_shaped ([4N x 1]) => [N x 4], still col-major
    %    => v_shaped(1) -> (1,1), v_shaped(2) -> (2,1), ...
    v_shaped_nor = reshape(v_shaped, [N,4]);

    % 5) Multiply
    v_rot = zeros(N, 4);
    for n = 1:N
        % 4x4
        M = squeeze(coefficients_3d(n, :, :));  % [4 x 4]
        v = v_shaped_nor(n, :)';                % [4 x 1]
        result = M * v;                         % [4 x 1]
        v_rot(n, :) = result';
    end
end


%% =========================================================================
function [v_shaped2] = poseblend(thetas, betas, v_shaped, SMPL_Model)
    % poseblend  Add pose-dependent blendshapes (e.g., corrective blendshapes)
    
    posedirs_vec = SMPL_Model.posedirs_vec;
    
    % For example, treat the last 89 angles as part of a quaternion-based feature
    pose            = thetas(4:end);  
    pose_reshaped   = reshape(pose, 3, []);
    quaternionAngle = axis2quat(pose_reshaped.');  % shape: [nJoints, 4]
    quaternionAngle = reshape(quaternionAngle.', 1, []);
    
    shape_feat = betas(1);  % e.g., picking the first shape coefficient
    feat = [quaternionAngle, shape_feat];
    feat = reshape(feat, [93, 1]);  % total dimension must match posedirs
    
    v_shaped2 = v_shaped + posedirs_vec * feat;
end

%% =========================================================================
function q = axis2quat(p)
    % axis2quat  Convert axis-angle to quaternion for each row in p
    % p is Nx3 axis angles => output Nx4 quaternions
    
    angle = sqrt(sum(p.^2, 2));
    norm_p = bsxfun(@rdivide, p, angle);
    
    cos_angle = cos(angle / 2);
    sin_angle = sin(angle / 2);
    
    qx = norm_p(:, 1) .* sin_angle;
    qy = norm_p(:, 2) .* sin_angle;
    qz = norm_p(:, 3) .* sin_angle;
    qw = cos_angle - 1;

    q = [qx, qy, qz, qw];
end

%% =========================================================================
function [R, dRdr] = so3exp(omega)
    % so3exp  Exponential map from a rotation vector (axis-angle) to a 3x3 rotation matrix
    % This version is adapted from the user code (Emanuele Ruffaldi 2017 @ SSSA).
    % If you do not need the Jacobian, omit second output.

    needJacobian = (nargout > 1);
    theta = norm(omega);
    
    if theta < 1e-6
        R = eye(3);
        if needJacobian
            % Hard-coded small-angle approximation for derivative
            dRdr = [ 0  0  0
                     0  0  1
                     0 -1  0
                     0  0 -1
                     0  0  0
                     1  0  0
                     0  1  0
                    -1  0  0
                     0  0  0 ];
        end
        return;
    end
    
    u = omega(:) / theta;
    w1 = u(1); w2 = u(2); w3 = u(3);
    
    A = [  0  -w3   w2
           w3   0  -w1
          -w2   w1   0 ];
    
    B = u * u.';
    
    alpha = cos(theta);
    beta  = sin(theta);
    gamma = 1 - alpha;
    
    R = alpha*eye(3) + beta*A + gamma*B;
    
    % If Jacobian needed, include user’s code (unchanged here):
    if needJacobian
        % The user code for dRdr has been omitted for brevity
        % but you can leave it here if you actually need it.
        dRdr = computeSo3expJacobian(omega, R);
    end
end

%% =========================================================================
function dRdr = computeSo3expJacobian(omega, R)
    % computeSo3expJacobian  (Optional) Detailed derivative code from user snippet
    % If you do not need the derivative, you can remove this function.
    
    % Implementation identical to the user’s posted code or your own version
    % ...
    
    % Example minimal placeholder:
    dRdr = zeros(9, 3);
end

%% =========================================================================
function plotVertices2(v_shaped, faces, N, trans)
    % Plot SMPL vertices with a translation offset in a rotated fashion.
    % If v_shaped is Nx4 or Nx3, handle accordingly:
    
    if size(v_shaped, 2) == 3
        v = v_shaped;
    else
        v = reshape(v_shaped, N, 4);
        v = v(:,1:3);
    end
    
    % Apply translation
    v2 = v + trans(:).';
    
    % Permute axes if needed (the user code swaps x->z, y->x, z->y)
    v(:,1) = v2(:,3);
    v(:,2) = v2(:,1);
    v(:,3) = v2(:,2);
    
    trimesh(faces, v(:,1), v(:,2), v(:,3));
    xlim([-2.0 1.0]);
    ylim([-1.5 1.5]);
    zlim([-0.2 1.6]);
    view([1.0 1.0 1.0]);
    axis vis3d
end
