for any **.py** run make sure to read logs at first run and use pip3 install -packagename-. Repeat this procedure until the code runs. If issue is not related to packages import, use github to create the issue.

To make the robot move on your screen you should:
1. python3 obstacleStreamer.py

2. python3 frankaIntegratorSwitching.py

3. python3 pbSim.py

If these three work, you may use frankaPlanner.py, that adds MPPI to optimize for policy. This is optional and useful only in case of complex obstacle shapes.

All scripts are communicating via zmq. 
Main code happens in **functions/MPPI.py**, where integration, distance calculations and DS modulation happens. And MPC, too. 

A lot of parameters are controlled from config.yaml. Each script parses this config.

To use various obstacles check the obstaclestreamer.py file, and modify the config.yaml for obstacle type. 
