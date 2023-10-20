import numpy as np
from modern_robotics import MatrixLog6, TransInv, FKinBody, se3ToVec, JacobianBody


def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    """
    # iteration counter
    i = 0
    # stop iteration if to much steps needed
    maxiterations = 10
    # initial error
    err = True
    # initial guess
    thetalist = np.array(thetalist0).copy()
    # array/matrix with thetalist (joint vector) for all iterations
    thetalist_all = np.array([thetalist])
    
    # iterate as long as error is too big and max. iterations not reached
    while err and i < maxiterations:
        # calculate new end-effector configuration in SE(3)
        end_effector = MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, thetalist)), T))
        # represent new end-effector configuration as twist
        Vb = se3ToVec(end_effector)
        # calculate new angular error magnitude
        current_eomg = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        # calculate new linear error magnitude
        current_ev = np.linalg.norm([Vb[3], Vb[4], Vb[5]])

        print(f"Iteration {i}")
        print(f"joint vector\n{thetalist}")
        print(f"SE(3) end−effector config\n{end_effector}")
        print(f"error twist V_b\n{Vb}")
        print(f"angular error magnitude ∣∣omega_b∣∣\n{current_eomg}")
        print(f"linear error magnitude ∣∣v_b∣∣ \n{current_ev}", end="\n\n")

        # calculate new error
        err = current_eomg > eomg or current_ev > ev
        if not err:
            break
        # calculate next (improved) joint vector values
        thetalist = thetalist + np.dot(np.linalg.pinv(JacobianBody(Blist, thetalist)), Vb)
        # save new calculated joint vector in matrix with all iterations
        thetalist_all = np.vstack([thetalist_all, thetalist])
        # increase iteration counter
        i = i + 1

    # save matrix with all joint vectors
    np.savetxt("iterates.csv", thetalist_all, delimiter=",")
    
    # return last calculated joint vector and indication if algorithm was successful
    return thetalist, not err


def test_iterates_assignment():
    print("Test Start")
    # Example 4.5 from book
    L_1 = 0.425
    L_2 = 0.392
    W_1 = 0.109
    W_2 = 0.082
    H_1 = 0.089
    H_2 = 0.095

    Blist = np.array([[        0,          0,    0,   0,    0, 0],
                      [        1,          0,    0,   0,   -1, 0],
                      [        0,          1,    1,   1,    0, 1],
                      [W_1 + W_2,        H_2,  H_2, H_2, -W_2, 0],
                      [        0, -L_1 - L_2, -L_2,    0,   0, 0],
                      [L_1 + L_2,          0,    0,    0,   0, 0],
                      ])

    M = np.array([[-1, 0, 0, L_1 + L_2],
                  [ 0, 0, 1, W_1 + W_2],
                  [ 0, 1, 0, H_1 - H_2],
                  [ 0, 0, 0,         1]])

    T = np.array([[ 0, 1,  0, -0.5],
                  [ 0, 0, -1,  0.1],
                  [-1, 0,  0,  0.1],
                  [ 0, 0,  0,    1]])

    thetalist0 = np.array([-5.538995307148402159e-01,
                           -2.644751876510111011e+00,
                           4.873798448931413674e+00,
                           1.455431929204372921e+00,
                           3.713614905134268618e+00,
                           2.083369971070475390e+00])
    eomg = 0.001
    ev = 0.0001

    thetalist, sucess = IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
    assert sucess
    np.testing.assert_array_almost_equal(thetalist, [-1.427428957457415881e-01,
                                                     -2.479369686394898409e+00,
                                                     4.542886597017788297e+00,
                                                     1.076396125385698621e+00,
                                                     3.284313049576952359e+00,
                                                     1.569114542273961188e+00])

test_iterates_assignment()