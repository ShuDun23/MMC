%% Multi-matrix Completion
function [USV,Err_1,k] = MMC(IncompleteData,array_Omega_c,CompleteData,r,c,max_out_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Core Algorithm for
%%% "Multi-Matrix Completion: A Novel Framework for Structurally Missing Elements"
%%% Hao Nan Sheng, Zhi-Yong Wang, Hing Cheung So, and Abdelhak M. Zoubir
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------
% Input:  IncompleteData: 2D/3D incomplete data
%         array_Omega_c: I - sampling_matrix
%         CompleteData: groundtruth data for relative error
%         r: row dimension of reduced representation S
%         c: column dimension of reduced representation S
%         max_out_iter: max number of iterations, 100 is suggested
% Output: USV: reconstruction
%         err: relative reconstruction error
%         k: current iteration
%-------------------------------------------------------------------
lambda = 0.3; % random missing: 0.1; random and structural missing: 0.3
rho1 = 1; % rho1
rho2 = 1; % rho2
[m,n,K] = size(IncompleteData);

% 3x3 laplacian kernel
L = [-eye(m-1), zeros(m-1,1)] + [zeros(m-1,1), eye(m-1)]; % L
R = [-eye(n-1); zeros(1,n-1)] + [zeros(1,n-1); eye(n-1)]; % R

% Initialization
P = zeros(m,n,K); % P
Q = zeros(m,n,K); % Q
Z = zeros(m,n,K); % Z
X_Omega_c = zeros(m,n,K); % X_{Omega^c}
A = IncompleteData; % initial A
U = [eye(r); zeros(m-r, r)]; % GLRAM initial L0

% Preallocation
S = zeros(r,c,K);
USV = zeros(m,n,K);
Y = zeros(m,n,K);

GLRAM_ITER = 3; % GLRAM max iteration num

Err_1(1:max_out_iter) = zeros;

for k = 1:max_out_iter % outer iteration

    fprintf('MMC Iteration: %d/%d\n', k, max_out_iter);

    [U,V]  = GLRAM(A, U, r, c, GLRAM_ITER, K);

    err(1:K) = zeros;

    for i= 1:K % inner iteration

        inv_C_C = inv( (2 + rho1 + rho2) * eye(m) + 2 * lambda * L' * L );

        inv_D_D = inv( rho2 * eye(n) + 2 * lambda * R * R');

        S(:,:,i) = U' * A(:,:,i) * V;

        USV(:,:,i) = U * S(:,:,i) * V';

        Y(:,:,i) = inv_C_C * ( 2 * ( IncompleteData(:,:,i) + X_Omega_c(:,:,i))...
            - P(:,:,i) + rho1 * USV(:,:,i) + Q(:,:,i) + rho2 * Z(:,:,i) );

        P(:,:,i) = P(:,:,i) + rho1 * ( Y(:,:,i) - USV(:,:,i) );

        X_Omega_c(:,:,i) = Y(:,:,i) .* array_Omega_c(:,:,i);

        Z(:,:,i) = ( rho2 * Y(:,:,i) - Q(:,:,i)) * inv_D_D;

        Q(:,:,i) = Q(:,:,i) + rho2 * (Z(:,:,i) - Y(:,:,i));

        A(:,:,i) = P(:,:,i) / rho1 + Y(:,:,i);

        % Calibration
        A(:,:,i) = IncompleteData(:,:,i) + A(:,:,i) .* array_Omega_c(:,:,i);

        % relative reconstruction error
        err(i) = norm(CompleteData(:,:,i) - USV(:,:,i),'fro')...
            / norm(CompleteData(:,:,i),'fro');

    end

    Err_1(k) = sum(err)/K;

    if Err_1(k) < 0.001 % 1e-3
        fprintf('MMC converged at iteration %d\n', k)
        break;
    end

end
end