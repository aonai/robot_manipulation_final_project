"""
Example Input:
    Xd = np.array([[0, 0, 1, 0.5], 
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0.5],
                    [0, 0, 0, 1]])
    Xd_next = np.array([[0, 0, 1, 0.6], 
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0.3],
                        [0, 0, 0, 1]])

    chasisConfig = np.array([0, 0, 0])
    armConfig = np.array([0, 0, 0.2, -1.6, 0])

    phi = chasisConfig[0]
    x = chasisConfig[1]
    y = chasisConfig[2]
    Tsb = np.array([[cos(phi),  -sin(phi), 0,   x],
                    [sin(phi),  cos(phi), 0,   y],  
                    [0,         0,         1, 0.0963],
                    [0, 0, 0, 1]])
    Tb0 = np.array([[1, 0, 0, 0.1662],
                    [0, 1, 0, 0],  
                    [0, 0, 1, 0.0026],
                    [0, 0, 0, 1]])  
    M0e = np.array([[1, 0, 0, 0.033],
                    [0, 1, 0, 0],  
                    [0, 0, 1, 0.6546],
                    [0, 0, 0, 1]])
    Blist = np.array([[0, 0, 1, 0, 0.033, 0],  
                    [0, -1, 0, -0.5076, 0, 0],  
                    [0, -1, 0, -0.3526, 0, 0],  
                    [0, -1, 0, -0.2176, 0, 0],  
                    [0, 0, 1, 0, 0, 0]]).T  
    T0e = mr.FKinBody(M0e, Blist, armConfig)
    Tse = multi_dot([Tsb, Tb0, T0e]) 

    Kp = np.eye(6)
    Ki = np.zeros((6, 6))
    dt = 0.01

Example Output:
    Vb = FeedbackControl(Tse, Xd, Xd_next, Kp, Ki, dt)


    # parameters of chasis 
    r = 0.0475      # m
    l = 0.47/2      # m
    w = 0.3/2       # m
    F = (r/4) * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                            [1,      1,       1,        1],
                            [-1,     1,       -1,       1]])
    F6 = np.array([np.zeros(4), np.zeros(4), F[0], F[1], F[2], np.zeros(4)])
    Teb = dot(mr.TransInv(T0e), mr.TransInv(Tb0))
    AdTeb = mr.Adjoint(Teb)
    Jbase = dot(AdTeb, F6)
    Jarm = mr.JacobianBody(Blist, armConfig)
    Je = concatenate((Jbase, Jarm), axis=1)
    vel = dot(pinv(Je, rcond=1e-4), Vb)
"""


import numpy as np
import modern_robotics as mr
from numpy import cos, sin, pi, append, dot, concatenate
from numpy.linalg import multi_dot, pinv


def FeedbackControl(X, Xd, Xd_next, Kp, Ki, dt, intErr):
    """ 
    Calculates the kinematics of the robot using task-space feedforward and feedback controls.
    The equation can be found in Chapter 11.3.
    
        Args:
            X (Tse) - The current actual end-effector configuration.
            Xd (Tse,d) - The current end-effector reference configuration.
            Xd_next (Tse,d, next) - The end-effector reference configuration at the next timestep 
                                    in the reference trajectory at a time dt later.
            Kp (6x6 matrix) - P controller.
            Ki (6x6 matrix) - I controller.
            dt (double) - The timestep between reference trajectory configurations.
            intErr (6 vector twist) - Accumulated integration error (Xerr) to be used by I controller.

        Returns:
            Vb (6 vector twist) - The commanded end-effector twist expressed in the end-effector frame.
            Xerr (6 vector twsit) -  The error twist that takes X to Xd. 
            intErr (6 vector twist) - Accumulated integration error after calculating Vb.
    """
    Xid = dot(mr.TransInv(X), Xd)
    AdXid = mr.Adjoint(Xid)
    
    Xid_next = dot(mr.TransInv(Xd), Xd_next)
    se3mat = (1/dt)*mr.MatrixLog6(Xid_next)
    Vd = mr.se3ToVec(se3mat)
    
    se3mat = mr.MatrixLog6(Xid)
    Xerr = mr.se3ToVec(se3mat)
    intErr += Xerr*dt

    Vb = dot(AdXid, Vd) + dot(Kp, Xerr) + dot(Ki, intErr)
    return Vb, Xerr, intErr

