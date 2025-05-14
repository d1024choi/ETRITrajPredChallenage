# ETRI Trajectory Prediction Challenge 2025

This repository provides potential challenge participants a gentle guide to ETRI trajectory prediction dataset and its corresponding devkit.

<p align="center">
  <img src="IMG/ETD.png" alt="ETD" width="400"/>
</p>

## ETRI Trajectory Dataset (ETD)
 
+ We acquired more than 100 driving logs, each lasting around 30 seconds, containing trajectories of our autonomous vehicle as well as its surrounding objects (e.g., vehicles, pedestrians), and sampled at 10Hz, in the vicinities of ETRI. We split the driving logs into two groups, one for training and the other for test. From a scene, we roughly generated 280 driving scenes, each containing 2 second observed trajectories, 6 second future trajectories, and HD map elements (e.g., lane centerlines) at a specific time stamp. As a result, we could generate 25,000 scenes for training and 5,000 scenes for test whose format is similar to that of Argoverse2 (https://www.argoverse.org/av2.html#forecasting-link). 

## Scene File Structure

A driving scene, stored as pkl format, is a dictionary with four keys **log_id**, **frm_idx**, **agent**, and **map**. **log_id** (_string format_) denotes the ID of the log the current scene is derived from and **frm_idx** (_integer_) means the time step at which the AV is located currently. **agent** is a dictionary, each containing agents' trajectories and class information. **map** is a list of dictionaries, each containing a lane segment of HD map.

+ **agent**
    * 'num_nodes' : The number of agents in the scene.
    * 'av_index' : The identification number of the AV.
    * 'id' : A list of the agents' IDs.
    * 'type' : A numpy array of size 'num_nodes', identifying object classes of the agents. (0: vehicle, 1: pedestrian, 2: cyclist.)
    * 'position' : A numpy array of size 'num_nodes' x 80 x 3, indicating the trajectories of the agents.
    * 'heading' : A numpy array of size 'num_nodes' x 80, indicating heading directions of the agents in radians.
    * 'valid_mask' : A boolean type numpy array of size 'num_nodes' x 80, indicating whether a (x,y,z) position of an agent at a specific time step is available or not.
    * 'predict_mask' : A boolean type numpy array of size 'num_nodes' x 80, indicating whether a position of an agent at a specific time step must be predicted or not.
    * 'category' : A numpy array of size 'num_nodes', indicating categories of the agents. (0: fragmented track, 1: full track, but not to be predicted, 2: full track and to be predicted.) The prediction results of category 2 agents are only considered for the prediction performance calculation (e.g., minADE, minFDE).
    * 'wlh' : a numpy array of size 'num_nodes' x 3, indicating the width, length, and height of each agent.
    * 'num_valid_node' : The number of catetory 2 agents in the scene.
