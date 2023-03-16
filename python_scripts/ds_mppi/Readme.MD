To make the robot move on your screen you should:
1. python3 ObstacleStreamer.py
check the import logs, use pip3 install -packagename-, repeat until it runs
2. python3 frankaintegratorswitching.py
3. python3 pbSim.py

All three scripts are communicating via zmq. 
Main magic happens in MPPI.py, where integration, distance calculations and ds modulation happens.

