To see the reinforcement learning bot competiting go to: https://halite.io/user/?user_id=1141

Dueling Deep Q-Network with prioritized experience replay playing halite 3 which is an AI competition launched by Two Sigma. 
The DDQN algorithms has reached rank 400/6000 with no hard coded decision and only reinforcement learning. Train a model take about 24 hours.
I am generating experiences with 6 bots playing in parallel and then I update the DDQN with the PER sampled from the experiences.

To start training:

python trainer-Para.py

To visualize progress wrt game score:

python display_progress.py

To making Bot1 playing against Bot2:

halite "python Bot1.py" "python Bot2.py"
