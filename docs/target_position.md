## Set Target Positions
A robot was trained for the task to reach a target cube spawned randomly by setting joint position targets. This training was executed with 20 parallel environments for 700.000 timesteps on a NVIDIA GeForce RTX 4070 Ti.

The average rewards for Pybullet (left) and Isaac (right):
<div style="display:flex;">
    <img src="images/rewards_pyb.png" alt="Image 1" style="width:45%;">
    <img src="images/rewards_isaac.png" alt="Image 2" style="width:45%;">
</div>


The average euclidean distance for Pybullet (left) and Isaac (right):
![Image 3](images/dist_euclid_pyb.png){width=49%}
![Image 4](images/dist_euclid_isaac.png){width=49%}

The average angular distance for Pybullet (left) and Isaac (right):
![Image 3](images/dist_angular_pyb.png){width=49%}
![Image 4](images/dist_angular_isaac.png){width=49%}