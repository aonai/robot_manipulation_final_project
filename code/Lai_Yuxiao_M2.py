"""
Example Input:
    phi = 0
    x = 0
    y = 0
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
    Tse_i = multi_dot([Tsb, Tb0, M0e]) 

    Tsc_i = np.array([[1, 0, 0, 1],
                        [0, 1, 0, 0],  
                        [0, 0, 1, 0.025],
                        [0, 0, 0, 1]])
    Tsc_f = np.array([[0, 1, 0, 0],
                        [-1, 0, 0, -1],  
                        [0, 0, 1, 0.025],
                        [0, 0, 0, 1]]) 

    beta = 3*pi/4 
    offset = 0.1
    Tce_g = np.array([[cos(beta),    0,  sin(beta),      0],
                        [0,          1,      0,          0],  
                        [-sin(beta), 0, cos(beta),       0],
                        [0,          0,      0,          1]])
    Tce_standoff = np.array([[cos(beta),    0,  sin(beta),      0],
                                [0,          1,      0,          0],  
                                [-sin(beta), 0, cos(beta),       offset],
                                [0,          0,      0,          1]])
    k = 1

Example Output:
    result, gripper = TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_g, Tce_standoff, k)
    # write csv file
    f = open("traj.csv", "w")
    for i in range(len(result)):
        T = result[i]
        output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % \
                (T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2], T[2][0], T[2][1], T[2][2], \
                    T[0][3], T[1][3], T[2][3], gripper[i])
        f.write(output)
    f.close()
"""


import numpy as np
import modern_robotics as mr
from numpy import cos, sin, pi, append
from numpy.linalg import multi_dot


def TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_g, Tce_standoff, k):
    """
    Generate the trajectories to let an end effector grasp a box at initial configuration to 
    final configuration.
    
        Args:
            Tse_i - initial configuration of end-effector with respect to spatial frame.
            Tsc_i - initial configuration of cude with respect to spatial frame.
            Tsc_f - desired final configuration of cude with respect to spatial frame.
            Tce_g - configuration of end-effector with respect to cube when it is grasping the cube.
            Tce_standoff - standoff configuration of end-effector above the cube, before and after grasping,  
                            with respect  to the cube.
            k - number of trajectory confiugrations per 0.01 seconds.

        Returns:
            config - generated trajectories with 1600*k number of configurations.
            gripper - array of 1600*k number of 0 or 1 that represent the status of gripper during trajectories.
    """
    method = 3 

    # initial standoff before grasp
    Tf = 5
    N = Tf*k/0.01
    Tse_standoff = multi_dot([Tsc_i, Tce_standoff])
    config = mr.ScrewTrajectory(Tse_i, Tse_standoff, Tf, N, method)

    # grasp
    Tf = 1
    N = Tf*k/0.01
    Tse_g = multi_dot([Tsc_i, Tce_g])
    result = mr.ScrewTrajectory(Tse_standoff, Tse_g, Tf, N, method)
    config = config + result
    gripper = np.zeros(len(config))

    # close gipper
    Tf = 1
    N = Tf*k/0.01
    close = np.ones(int(N))
    gripper = append(gripper, close)
    last_T = config[len(config)-1]
    for i in range(int(N)):
        config = config + [last_T]

    # initial standoff after grasp
    Tf = 1
    N = Tf*k/0.01
    result = mr.ScrewTrajectory(last_T, Tse_standoff, Tf, N, method)
    config = config + result
    close = np.ones(int(N))
    gripper = append(gripper, close)

    # final standoff before release grasp
    Tf = 5
    N = Tf*k/0.01
    Tse_standoff_f = multi_dot([Tsc_f, Tce_standoff])
    result = mr.ScrewTrajectory(Tse_standoff, Tse_standoff_f, Tf, N, method)
    config = config + result
    close = np.ones(int(N))
    gripper = append(gripper, close)

    # release grasp
    Tf = 1
    N = Tf*k/0.01
    Tse_g_f = multi_dot([Tsc_f, Tce_g])
    result = mr.ScrewTrajectory(Tse_standoff_f, Tse_g_f, Tf, N, method)
    config = config + result
    close = np.ones(int(N))
    gripper = append(gripper, close)

    # open gipper
    Tf = 1
    N = Tf*k/0.01
    open_gripper = np.zeros(int(N))
    gripper = append(gripper, open_gripper)
    last_T = config[len(config)-1]
    for i in range(int(N)):
        config = config + [last_T]

    # final standoff after release grasp
    Tf = 1
    N = Tf*k/0.01
    Tse_standoff_f = multi_dot([Tsc_f, Tce_standoff])
    result = mr.ScrewTrajectory(last_T, Tse_standoff_f, Tf, N, method)
    config = config + result
    open_gripper = np.zeros(int(N))
    gripper = append(gripper, open_gripper)

    return config, gripper