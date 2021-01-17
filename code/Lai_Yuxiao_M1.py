"""
Example Input:
    chasisConfig = np.array([0, 0, 0])
    armConfig = np.array([0, 0, 0, 0, 0])
    wheelAngles = np.array([0, 0, 0, 0])
    currentConfig = concatenate((chasisConfig, armConfig, wheelAngles), axis=0)
    thetadot = np.array([0, 0, 0, 0, 0])    # arm speed
    u = np.array([10, 10, 10, 10])          # wheel speed
    vel = concatenate((thetadot, u), axis=0)
    maxVel = 11
    dt = 0.01

Example Output:
    config = []
    config.append(currentConfig)
    for i in range(100):
        currentConfig = NextState(currentConfig, vel, dt, maxVel)
        config.append(currentConfig)
    f = open("Output.csv", "w")
    for c in config:
        output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % \
                (*c, 0)
        f.write(output)
    f.close()

    The robot should drive forward (in the +x direction) by 0.475 m. 
"""


import numpy as np
import modern_robotics as mr
from numpy import cos, sin, pi, concatenate
from numpy.linalg import multi_dot


def NextState(current, vel, dt, maxVel):
    """
    Simulate the robot's configuration in the next stepsize (dt) with given velocity. The
    function uses first-order Euler step, so next arm joint angles and next wheel angles 
    are old angles + vel*dt. The next chasis config is calculated from odometry and assuming 
    planar motion. 
    
        Args:
            current (Vector of size 12) - current configuration of the robot. 
                                        3 variables for the chassis configuration, 
                                        5 variables for the arm configuration, and 
                                        4 variables for the wheel angles.
            vel (Vector of size 9) - speed of the robot. 5 variables for the arm joint speeds, 
                                    and 4 variables for the wheel speeds.
            dt (double) - timestep
            maxVel (double) - maximum angular speed of the arm joints and the wheels

        Returns:
            nextConfig (Vector of size 12) - configuration of the robot in next stepsize. Is 
                                            formatted the same as current.  
    """
    # parameters of chasis 
    r = 0.0475      # m
    l = 0.47/2      # m
    w = 0.3/2       # m

    # current config 
    currentChasis = current[0:3]
    currentArm = current[3:8]
    currentWheel = current[8:12]

    # check if any given vel exceeds maxVel 
    for i, v in enumerate(vel):
        if v >= maxVel:
            vel[i] = maxVel
        elif v <= maxVel*-1:
            vel[i] = maxVel*-1
    armVel = vel[0:5]
    wheelVel = vel[5:9]

    # new angles = old angles + vel * dt
    nextArm = currentArm + armVel * dt
    nextWheel = currentWheel + wheelVel * dt

    # assuming planar motion, so Vb = F*(vel*dt) and 
    # Vb6 = [0, 0, omg_bz, v_bx, b_by, 0].T. 
    # So Tbk_bk+1 = matrix exp of Vb6 is the amount 
    # traveled by the robot within one stepsize.
    F = (r/4) * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                            [1,      1,       1,        1],
                            [-1,     1,       -1,       1]])
    Vb = multi_dot([F, wheelVel * dt])
    Vb6 = np.array([0, 0, Vb[0], Vb[1], Vb[2], 0])
    se3mat = mr.VecTose3(Vb6)
    Tbk_bk1 = mr.MatrixExp6(se3mat)
       
    phi = currentChasis[0]
    x = currentChasis[1]
    y = currentChasis[2]
    Tsbk = np.array([[1, 0, 0, x],
                    [0, cos(phi),  -sin(phi), y],
                    [0, sin(phi),   cos(phi),  0.0963],  
                    [0,          0,         0,  1]])
    Tsbk1 = multi_dot([Tsbk, Tbk_bk1])
    se3mat = mr.MatrixLog6(Tsbk1)
    nextVb6 = mr.se3ToVec(se3mat)
    nextChasis = nextVb6[2:5]

    # format overall config to return
    nextCofig = concatenate((nextChasis, nextArm, nextWheel), axis=0)
    return nextCofig
