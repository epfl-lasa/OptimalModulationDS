To make the robot move on your screen you should:
1. python3 ObstacleStreamer.py
check the import logs, use pip3 install -packagename-, repeat until it runs
2. python3 frankaintegratorswitching.py
3. python3 pbSim.py
If these three work, you may even play with frankaPlanner.py, that adds policy MPC. This is optional and useful only in case of complex obstacle shapes.

All scripts are communicating via zmq. 
Main magic happens in functions/MPPI.py, where integration, distance calculations and ds modulation happens. And MPC, too. 

A lot of parameters are controlled from config.yaml. 
Each script parses this config.

To use various obstacles check the obstaclestreamer.py file, and modify the config.yaml for obstacle type. 

