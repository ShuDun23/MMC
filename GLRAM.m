function [L,R]  = GLRAM(A, L, low1, low2, ITE, n)

%------------------------------------------------------------------
% Input   A: data matrix
%         L: initial L0
%         low1: row dimension of reduced representation M{i}
%         low2: column dimension of reduced representation M{i}
%         ITE: number of iterations
% Output: L: left transformation
%         M: reduced representation
%         R: right transformation
%         err: reconstruction error measured by RMSRE
%-------------------------------------------------------------------

%[row, col] = size(A{1}); % row and column dimensions of A{i}
% n          = size(A,2); % number of data points

%-------------------------------------------------------------------
%  M(1:low1,1:low2,1:n)=zeros;
%  A_const(1:128,1:128,1:n)=zeros;
for ite = 1:ITE

    MU = 0;
    for i =1:n
        TU = L'* A(:,:,i);
        MU = MU + TU'*TU;
    end;

    [U0,D0,V0] = svd(MU);
    R = V0(:,1:low2);

    %-------------------------------------------------------------------

    NU = 0;
    for i =1:n
        TV = A(:,:,i)*R;
        NU =  NU + TV*TV';
    end;

    [U1,D1,V1] = svd(NU);
    L = V1(:,1:low1);

end;

%-------------------------------------------------------------------

% for i =1:n
%      M(:,:,i) = L'* A(:,:,i)*R;
% end;


%  Compute the RMSRE value (reconstruction error)
%
% err =  ERROR(L, M, R, A, n);


% %-------------------------------------------------------------------
% %                  THE END OF THE MAIN PROGRAM
% %-------------------------------------------------------------------
% function error = ERROR(L, M, R, A, n)
% %  SUBROUTINE TO COMPUTE THE RMSRE VALUE
%
% error = 0;
%
% for i =1:n
%     A_const(:,:,i) = L*M(:,:,i)*R';
%     TEMP       = A(:,:,i) - A_const(:,:,i);
%     error      = error + norm(TEMP,'fro')^2;
% end;
%
% error = sqrt(error/n);