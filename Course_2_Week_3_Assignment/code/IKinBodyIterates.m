function IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
% *** CHAPTER 6: INVERSE KINEMATICS ***
% Takes Blist: The joint screw axes in the end-effector frame when the
%              manipulator is at the home position, in the format of a 
%              matrix with the screw axes as the columns,
%       M: The home configuration of the end-effector,
%       T: The desired end-effector configuration Tsd,
%       thetalist0: An initial guess of joint angles that are close to 
%                   satisfying Tsd,
%       eomg: A small positive tolerance on the end-effector orientation
%             error. The returned joint angles must give an end-effector 
%             orientation error less than eomg,
%       ev: A small positive tolerance on the end-effector linear position 
%           error. The returned joint angles must give an end-effector
%           position error less than ev.
%
% Uses an iterative Newton-Raphson root-finding method.
% The maximum number of iterations before the algorithm is terminated has 
% been hardcoded in as a variable called maxiterations. It is set to 20 at 
% the start of the function, but can be changed if needed.  
% Example Inputs:
% 
% clear; clc;
% Blist = [[0; 0; -1; 2; 0; 0], [0; 0; 0; 0; 1; 0], [0; 0; 1; 0; 0; 0.1]];
% M = [[-1, 0, 0, 0]; [0, 1, 0, 6]; [0, 0, -1, 2]; [0, 0, 0, 1]];
% T = [[0, 1, 0, -5]; [1, 0, 0, 4]; [0, 0, -1, 1.6858]; [0, 0, 0, 1]];
% thetalist0 = [1.5; 2.5; 3];
% eomg = 0.01;
% ev = 0.001;
% IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)

% iteration counter
i = 0;
% stop iterations if to much steps needed
max_iteration = 20;
% initial error
err = true;
% initial guess
thetalist = thetalist0;
% array/matrix with thetalist (joint vector) for all iterations
thetalist_all = horzcat(thetalist);

while err && i < max_iteration
    % calculate new end-effector configuration in SE(3)
    end_effector = MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T);
    % represent new end-effector configuration as twist
    Vb = se3ToVec(end_effector);
    % calculate new angular error magnitude
    current_eomg = norm(Vb(1:3));
    % calculate new linear error magnitude
    current_ev = norm(Vb(4:6));
    
    % print result
    disp(['Itertaion ', mat2str(i)]);
    disp(['joint vector', mat2str(thetalist, 4)]);
    disp(['SE(3) end−effector config', mat2str(end_effector, 4)]);
    disp(['error twist V_b', mat2str(Vb, 4)]);
    disp(['angular error magnitude ||omega_b||', mat2str(current_eomg, 4)]);
    disp(['linear error magnitude ||v_b||', mat2str(current_ev, 4)]);
    
    % calculate new error
    err = current_eomg > eomg || current_ev > ev;
    if err == false
        break
    end
    % calculate next (improved) joint vector values
    thetalist = thetalist + pinv(JacobianBody(Blist, thetalist)) * Vb;
    % save new calculated joint vector in matrix with all iterations
    thetalist_all = horzcat(thetalist);
    % increase iteration counter
    i = i + 1;
end
csvwrite("iterates.csv", thetalist_all);
end
