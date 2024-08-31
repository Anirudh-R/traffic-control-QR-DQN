## Risk-sensitive traffic control system using Deep Reinforcement Learning

This project implements the QR-DQN algorithm to tackle the traffic congestion problem. More information can be found in the `/docs` folder and in the first referenced paper.

We use SUMO ([Simulation of Urban MObility](https://eclipse.dev/sumo/)) V1.8 package to simulate traffic patterns on which the algorithm has been tested. Details of SUMO installation can be found in the [documentation](https://sumo.dlr.de/docs/Installing.html). The TraCI API is used to control/monitor the simulation from Python.

We have used PyCharm and Anaconda for development of this project. Any IDE can be used, provided the dependencies are properly installed in your respective environments. Tested on Windows and Ubuntu.

The project folder contains the following important directories:
* The `/src` directory contains all the source codes needed to run the project.
* The `/scripts` directory contains SUMO files needed to generate traffic patterns. Also, output files from SUMO will be generated in this directory.
* The `/src/pt_trainedmodel` directory contains the saved neural network weights after training.
* The `/src/datapoints` directory contains the saved datapoints of the averaged metrics.
* The `/src/img` contains the value distrubution plots generated after each action taken.
* The `/src/logs` contains the logs of no. of steps, running reward, and Huber loss after each training run of QR-DQN.

### Procedure to run the project
1. Set the `Nruns` parameter for each algo in the `main.py` file to specify the no. of training/trial runs. Recommended minimum `Nruns=25`.
2. Run `main.py`. This will run the static signalling algorithm for `Nruns` trials and LQF algorithm for `Nruns` trials. Then training for QR-DQN will start. Based on the trained weights, QR-DQN Live will perform `Nruns` trials.
3. During the training, an increasing running reward and a decreasing Huber loss (both printed) should indicate that the algorithm is converging.
4. Run `plot_metrics.py` and `draw_final_graphs.py` to get the performance comparison graphs.

### References
1. Intelligent Traffic Control System using Deep Reinforcement Learning: https://ieeexplore.ieee.org/document/9744226
2. Distributional Reinforcement Learning with Quantile Regression, 2017,
arXiv:1710.10044v1
3. Distributional Bellman and the C51 Algorithm:
https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
4. https://github.com/senya-ashukha/quantile-regression-dqn-pytorch
5. Reinforcement Learning With Function Approximation for Traffic Signal Control:
http://www.cse.iitm.ac.in/~prashla/papers/2011RLforTrafficSignalControl_ITS.pdf
6. SUMO User Documentation:
   https://sumo.dlr.de/docs/SUMO_User_Documentation.html
7. TraCI: https://sumo.dlr.de/docs/TraCI.html