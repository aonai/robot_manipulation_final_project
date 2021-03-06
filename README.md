# Robot Manipulation Final Project 

This project should make a youBot starts from a initial configuration, then grab a box in the environment, and finally move the box to another configuration. Use CoppeliaSim scene 6 and Script_Result.csv to animate the trajectory.  

The software used for the code is Python3. The code folder consists of three python script for functions NextState, TrajectoryGenerator, and FeedbackControl. This folder also includes a main script used to generate a csv file to be put into CoppeliaSim (Script_Result.csv), a csv file recording the error twist when generating robot configurations along the trajectory (Error.csv), a log file (runscript.log), and a plot of error twist as a function of time (Xerr.png). 

The best folder includes results when using well-tuned controllers. The overshoot folder includes results when using less-well-tuned controllers. Both results have initial cube configuration at (1m, 0m, 0rad) and goal cube configuration (0m, -1m, -pi/2rad). The newTask folder includes results when using well-tuned controller and different cube configurations: from (0m, -1m, -pi/2rad) to (0m, 1m, pi/2rad). All three results have the same initial robot configuration which has 30 degree orientation error and 0.2m position error from the default initial robot configuration. The initial arm configuration is set to be (0, 0, 0.2, -1.6, 0)rad to avoid singularity when calculating velocities. When generating robot configurations along the trajectory, the code numpy.linalg.pinv(Je, rcond=1e-4)  is used to avoid singularities when calculating inverse of Jacobian. It should ignore singular values less than 1e-4.

At start of the robot configuration results, the robot will oscillate a little bit to move to the desired configuration along the trajectory, then once it has converged to the trajectory, the errors are limited to smaller numbers as shown in the plots. 

