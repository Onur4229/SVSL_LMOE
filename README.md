# Specializing Versatile Skill Libraries Using Local Mixture of Experts
Code for "Specializing Versatile Skill Libraries using Local Mixture of Experts" - published at CoRL 2021, see <https://openreview.net/forum?id=j3Rguo81Yi_> 

You will need Mujoco to run the robotic experiments. Note that Mujoco is free now. However, when the experiments were conducted, 
Mujoco was not officially open sourced. Therefore, we use the mujoco-py version, which is based on Mujoco 2.0. Using the newest
version of mujoco-py should work, but was not tested yet.


This code was tested with Python 3.6. A list of all required packages, including version numbers can be found in requirements.txt and
can be installed with 
```
pip install -r requirements.txt
```

To run the Beerpong experiment, run

```
python Beer_Pong_Run.py
```
To run the Table Tennis experiment, run

```
python Table_Tennis_Run.py
```
