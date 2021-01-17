import logging
import matplotlib.pyplot as plt
import numpy as np
import modern_robotics as mr
from numpy import cos, sin, pi, append, concatenate, dot
from numpy.linalg import multi_dot, pinv
from Lai_Yuxiao_M1 import NextState
from Lai_Yuxiao_M2 import TrajectoryGenerator
from Lai_Yuxiao_M3 import FeedbackControl

# -------------------- start log file --------------------------
logging.basicConfig(filename='runscript.log', level=logging.DEBUG)

# -------------------- set up parameters --------------------------
logging.info('Set up parameters')
r = 0.0475      # m
l = 0.47/2      # m
w = 0.3/2       # m
F = (r/4) * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1,      1,       1,        1],
                        [-1,     1,       -1,       1]])
F6 = np.array([np.zeros(4), np.zeros(4), F[0], F[1], F[2], np.zeros(4)])

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

logging.info('Set up robot initial config')
chasisConfig = np.array([0, 0, 0])
armConfig = np.array([0, 0, 0.2, -1.6, 0])
wheelAngles = np.array([0, 0, 0, 0])
currentConfig = concatenate((chasisConfig, armConfig, wheelAngles), axis=0)
T0e = mr.FKinBody(M0e, Blist, currentConfig[3:8])
Tse_i = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0.5],  
                [0, 0, 0, 1]]) 

logging.info('Set up cube configs')
# cube configs for new task
# Tsc_i = np.array([[0, 1, 0, 0],
#                     [-1, 0, 0, -1],  
#                     [0, 0, 1, 0.025],
#                     [0, 0, 0, 1]]) # initial config of cube
# Tsc_f = np.array([[0, -1, 0, 0],
#                     [1, 0, 0, 1],  
#                     [0, 0, 1, 0.025],
#                     [0, 0, 0, 1]]) # final config of cube

# cube configs for best and overshoot results
Tsc_i = np.array([[1, 0, 0, 1],
                    [0, 1, 0, 0],  
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]]) # initial config of cube
Tsc_f = np.array([[0, 1, 0, 0],
                    [-1, 0, 0, -1],  
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]]) # final config of cube

# the grasp config with be rotated 3*pi/4 by y-axis of cube config
# the standoff config should be 0.1m above cube and rotated 3*pi/4 by y-axis of cube config
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

# ----------------------- Generate trajectory and configs --------------------------
logging.info('Generate trajectory')
k = 1
allTse, gripper = TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_g, Tce_standoff, k)

# input param for FeedbackControl and NextState
maxVel = 10
dt = 0.01

# param to store output of FeedbackControl and NextState
allConfig = []
allConfig.append(currentConfig)

intErr = np.zeros(6)
allErr = []

# To find the next config of the robot, intput Tse and Tse_next (which is the 
# next Tse in allTse) into FeedbackControl to calcualte the end-effector twist. 
# Then, use pseudo jacobian of the robot to calculate velocity from the twist. 
# The next config can thus be found using NextState. 
logging.info('Generate list of configs during trajectory')
for i in range(len(allTse)-1):
    Xd = allTse[i]
    Xd_next = allTse[i+1]

    # calcualte current actual Tse from current chasis config
    phi = currentConfig[0]
    x = currentConfig[1]
    y = currentConfig[2]
    Tsb = np.array([[cos(phi),  -sin(phi), 0,   x],
                    [sin(phi),  cos(phi), 0,   y],  
                    [0,         0,         1, 0.0963],
                    [0, 0, 0, 1]])
    T0e = mr.FKinBody(M0e, Blist, currentConfig[3:8])
    Tse = multi_dot([Tsb, Tb0, T0e]) 

    # controller values for best result
    Kp = 20*np.eye(6)
    Ki = 1*np.eye(6)

    # controller values for overshoot result
    # Kp = 8*np.eye(6)
    # Ki = 1*np.eye(6)


    # calculate end-effector twist and error twist
    Vb, Xerr, intErr = FeedbackControl(Tse, Xd, Xd_next, Kp, Ki, dt, intErr)
    allErr.append(Xerr)

    # calculate velocity in form (u, thetadot)
    Teb = dot(mr.TransInv(T0e), mr.TransInv(Tb0))
    AdTeb = mr.Adjoint(Teb)
    Jbase = dot(AdTeb, F6)
    Jarm = mr.JacobianBody(Blist, currentConfig[3:8])
    Je = concatenate((Jbase, Jarm), axis=1)
    vel = dot(pinv(Je, rcond=1e-4), Vb)

    # calculate next robot config
    # the input velocity should be in form (thetadot, u)
    vel = concatenate((vel[4:9], vel[0:4]), axis=0)
    currentConfig = NextState(currentConfig, vel, dt, maxVel)
    allConfig.append(currentConfig)

# ----------------------- plot Xerr --------------------------
logging.info('plot Xerr')
plt.plot(np.linspace(0, 16, len(allErr)), allErr[:])
plt.xlabel("Time (sec)")
plt.ylabel("Error")
plt.legend(['Xerr_1','Xerr_2','Xerr_3','Xerr_4','Xerr_5','Xerr_6'])
plt.show()
    
# ----------- Write csv files ---------------  
logging.info('write csv files')
f = open("Script_Result.csv", "w")
for i, c in enumerate(allConfig):
    output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % \
            (*c, gripper[i])
    f.write(output)
f.close()

f = open("Error.csv", "w")
for err in allErr:
    output = " %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f\n" % \
            (err[0], err[1], err[2], err[3], err[4], err[5])
    f.write(output)
f.close()

logging.info('Done')
