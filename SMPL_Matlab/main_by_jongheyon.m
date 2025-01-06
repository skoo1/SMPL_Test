clear all;
load('model.mat');

global kintree_table v_tem_vec shapedirs_vec J_reg_vec weights;
global posedirs_vec;
global N K;

% num of vertices
N = size(v_template, 1);
K = size(weights, 2);

% matlab starts from 1
faces = f + 1;
kintree_table = kintree_table + 1;

% shapedirs
shapedirs_vec = reshape(shapedirs, N * 3, 300);

% posedirs
posedirs_vec = reshape(posedirs, N * 3, 93);

% add zero to homo
shapedirs_vec = [shapedirs_vec; zeros(N, 300)];
posedirs_vec = [posedirs_vec; zeros(N, 93)];

% template vertices, to vector:(3N,1)
v_tem_vec = reshape(v_template, N * 3, 1);

% add 1, to (4N,1) homo
v_tem_vec = [v_tem_vec; ones(N, 1)];

% make joints regressor
% ATTN! the last block is set to Zero
% so the joints is not the homo

zeros_24x6890 = sparse(K, N);
J_reg_vec = [J_regressor, zeros_24x6890, zeros_24x6890, zeros_24x6890;
        zeros_24x6890, J_regressor, zeros_24x6890, zeros_24x6890;
        zeros_24x6890, zeros_24x6890, J_regressor, zeros_24x6890;
        zeros_24x6890, zeros_24x6890, zeros_24x6890, zeros_24x6890];

%%
filename = 'walking_20240125.txt';
fid = fopen(filename, 'r');
rawData = fread(fid, inf);
strData = char(rawData');
fclose(fid);
data = jsondecode(strData);

% 데이터 수
numFrame = size(data, 1);

% betas와 N 정의 필요
betas = zeros(300,1);

% 모델 플로팅을 위한 설정
f = figure(5);
f.Position = [500 000 1800 1400]; 

% 데이터 순회
for i = 1:numFrame
    temp_data = reshape(data(i,:), [], 1);
    thetas = temp_data(4:end);
    trans = temp_data(1:3);
    thetas = thetas + 0.00001;

    % SMPL 모델과 플롯 생성
    [verts, joints] = SMPLmodel(thetas, betas);
    % plotVertices(verts, faces, N);
    plotVertices2(verts, faces, N, trans);

    % 시간 지연
    pause(0.001);
    
    % 현재 플롯 지우기
%     clf;
end

%%
function [verts, joints] = SMPLmodel(thetas, betas)
    [j_shaped, v_shaped, J_DjshDbetas,J_DvshDbetas] = shapeblend(betas);
    [global_transform, global_transform_remove] = transforms(thetas, j_shaped);
    [v_shaped2] = poseblend(thetas, betas, v_shaped);
    [v_rot,J_DvrotDtrans,J_DvrotDvshaped] = poserot(global_transform_remove, v_shaped2);
    verts = v_rot;
    joints = reshape(global_transform(:, 1:3, 4), 72, 1);
end

function [j_shaped, v_shaped, J_DjshDbetas,J_DvshDbetas] = shapeblend(betas)
    %SHAPEBLEND Summary of this function goes here
    %   Detailed explanation goes here
    %% shape blending
    global shapedirs_vec v_tem_vec J_reg_vec;
    v_shapeblend = shapedirs_vec * betas;

    v_shaped = v_tem_vec + v_shapeblend;
    j_shaped = J_reg_vec * v_shaped;
    if nargout > 2
        % this jacobian is static
        % we could pre-calculate it
        J_DvshDbetas = shapedirs_vec;
        J_DjshDbetas = J_reg_vec*shapedirs_vec;
    end
end

function [global_transform, global_transform_remove] = transforms(thetas, j_shaped)
    global kintree_table;
    global K;
    %TRANSFORMS Summary of this function goes here
    %   Detailed explanation goes here
    global_transform = zeros(K, 4, 4);
    global_transform_remove = zeros(K, 4, 4);

    for i = 1:K
        if i == 1 % for root
            [rotmat] = so3exp(thetas(3 * i - 2:3 * i));
            global_transform(i, 1:3, 1:3) = rotmat;
            global_transform(i, 1, 4) = j_shaped(i);
            global_transform(i, 2, 4) = j_shaped(i + K);
            global_transform(i, 3, 4) = j_shaped(i + 2 * K);
            global_transform(i, 4, 4) = 1;
        else
            [rotmat] = so3exp(thetas(3 * i - 2:3 * i));
            trans = ones(4, 1);
            trans(1, 1) = j_shaped(i) - j_shaped(kintree_table(1, i));
            trans(2, 1) = j_shaped(i + K) - j_shaped(kintree_table(1, i) + K);
            trans(3, 1) = j_shaped(i + 2 * K) - j_shaped(kintree_table(1, i) + 2 * K);
            local_trans = eye(4);
            local_trans(1:3, 1:3) = rotmat;
            local_trans(1:3, 4) = trans(1:3, 1);
            global_transform(i, :, :) = squeeze(global_transform(kintree_table(1, i), :, :)) ...
                * local_trans;
        end
            
        % calculate the removed rotmat
        jZero = [j_shaped(i); j_shaped(i + K); j_shaped(i + 2 * K); 0];
        fx = squeeze(global_transform(i, :, :)) * jZero;
        pack = zeros(4);
        pack(1:3, 4) = fx(1:3, 1);
        global_transform_remove(i, 1:4, 1:4) = squeeze(global_transform(i, :, :)) - pack;
    end % end transformation

end

function [ v_rot, J_DvrotDtrans, J_DvrotDvshaped ] = poserot(global_transform_remove, v_shaped)
    %POSEROT Summary of this function goes here
    %   Detailed explanation goes here
    global weights N K;
    %% rotate all the vertices
    global_transform_remove_vec = reshape(global_transform_remove, K, 16);
    coefficients = weights * global_transform_remove_vec;
    coefficients = reshape(coefficients, N, 4, 4);
    v_shaped_nor = reshape(v_shaped, N, 4);
    v_rot = zeros(N, 4);
    for j = 1:4
        v_rot(:, j) = coefficients(:, j, 1) .* v_shaped_nor(:, 1) + ...
            coefficients(:, j, 2) .* v_shaped_nor(:, 2) + ...
            coefficients(:, j, 3) .* v_shaped_nor(:, 3) + ...
            coefficients(:, j, 4) .* v_shaped_nor(:, 4);
    end
    if nargout > 1
        % this jacobian is static
        % we could pre-calculate it
        % J_DvrotDtrans: (N,4,16)
        J_DvrotDtrans = zeros(N,4,K,16);
        for n = 1:N
           for k = 1:K
               J_DvrotDtrans(n,:,k,:) = kron(v_shaped_nor(n,:),eye(4));
           end
        end
        J_DvrotDvshaped = coefficients;
    end

end

function plotVertices(v_shaped, faces, N)
%PLOTVERTICES Summary of this function goes here
%   Detailed explanation goes here
    if size(v_shaped,2) == 3
        v = v_shaped;
    else
        v = reshape(v_shaped,N,4);
        v = v(:,1:3);
    end

    trimesh(faces,v_shaped(:,1), v_shaped(:,2), v_shaped(:,3));
    xlim([-0.8 0.8]);
    ylim([-1.4 0.8]);
    zlim([-0.8 0.6]);
    view([0.0 0.0 1.0]);

end

function plotVertices2(v_shaped ,faces, N, trans)
%PLOTVERTICES Summary of this function goes here
%   Detailed explanation goes here
    if size(v_shaped,2) == 3
        v = v_shaped;
    else
        v = reshape(v_shaped,N,4);
        v = v(:,1:3);
    end

    v2(:,1) = v(:,1) + trans(1);
    v2(:,2) = v(:,2) + trans(2);
    v2(:,3) = v(:,3) + trans(3);

    v(:,1) = v2(:,3);
    v(:,2) = v2(:,1);
    v(:,3) = v2(:,2);

    trimesh(faces,v(:,1), v(:,2), v(:,3));
    xlim([-2.0 1.0]);
    ylim([-1.5 1.5]);
    zlim([-0.2 1.6]);
    view([1.0 1.0 1.0]);
end

function [v_shaped2] = poseblend(thetas, betas, v_shaped)
    global posedirs_vec;

    pose = thetas(4:end);
    pose_reshaped = reshape(pose, 3, []);
    quaternion_angles = axis2quat(pose_reshaped.');
    quaternion_angles = reshape(quaternion_angles.', 1, []);
    shape_feat = betas(1);
    feat = [quaternion_angles, shape_feat];
    feat = reshape(feat, 93, 1);

    v_shaped2 = v_shaped + posedirs_vec * feat;

end

function q = axis2quat(p)
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

% Emanuele Ruffaldi 2017 @ SSSA
function [rotationMatrix, dRdr] = so3exp(rotationVector)
    rotationVector = rotationVector';
    needJacobian = (nargout > 1);
    theta = norm(rotationVector);
    if theta < 1e-6
        rotationMatrix = eye(3, 'like', rotationVector);

        if needJacobian
            dRdr = [0 0 0; ...
                    0 0 1; ...
                    0 -1 0; ...
                    0 0 -1; ...
                    0 0 0; ...
                    1 0 0; ...
                    0 1 0; ...
                    -1 0 0; ...
                    0 0 0];
            dRdr = cast(dRdr, 'like', rotationVector);
        end

        return;
    end

    u = rotationVector./ theta;
    u = u(:);
    w1 = u(1);
    w2 = u(2);
    w3 = u(3);

    A = [0, -w3, w2; ...
            w3, 0, -w1; ...
            -w2, w1, 0];

    B = u * u';

    alpha = cos(theta);
    beta = sin(theta);
    gamma = 1 - alpha;

    rotationMatrix = eye(3, 'like', rotationVector) * alpha + beta * A + gamma * B;

    if needJacobian
        I = eye(3, 'like', rotationVector);

        % m3 = [rotationVector,theta], theta = |rotationVector|
        dm3dr = [I; u']; % 4x3

        % m2 = [u;theta]
        dm2dm3 = [I./theta -rotationVector./theta^2; ...
                zeros(1, 3, 'like', rotationVector) 1]; % 4x4

        % m1 = [alpha;beta;gamma;A;B];
        dm1dm2 = zeros(21, 4, 'like', rotationVector);
        dm1dm2(1, 4) = -beta;
        dm1dm2(2, 4) = alpha;
        dm1dm2(3, 4) = beta;
        dm1dm2(4:12, 1:3) = [0 0 0 0 0 1 0 -1 0; ...
                            0 0 -1 0 0 0 1 0 0; ...
                            0 1 0 -1 0 0 0 0 0]';

        dm1dm2(13:21, 1) = [2 * w1, w2, w3, w2, 0, 0, w3, 0, 0];
        dm1dm2(13:21, 2) = [0, w1, 0, w1, 2 * w2, w3, 0, w3, 0];
        dm1dm2(13:21, 3) = [0, 0, w1, 0, 0, w2, w1, w2, 2 * w3];

        dRdm1 = zeros(9, 21, 'like', rotationVector);
        dRdm1([1 5 9], 1) = 1;
        dRdm1(:, 2) = A(:);
        dRdm1(:, 3) = B(:);
        dRdm1(:, 4:12) = beta * eye(9, 'like', rotationVector);
        dRdm1(:, 13:21) = gamma * eye(9, 'like', rotationVector);

        dRdr = dRdm1 * dm1dm2 * dm2dm3 * dm3dr;
    end
    % this jacobian is column first
end