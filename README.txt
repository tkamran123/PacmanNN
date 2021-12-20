
=================================================================================
Requires:
	python 3.5
	pytorch 0.4
Important files:
	DQN.py
	pacmanDQN_Agents.py 

To test the DQN network, launch:
	python3 pacman.py -p PacmanDQN -n 200 -x 100 -l smallGrid

To train the DQN network, launch:
	python3 pacman.py -p PacmanDQN -n 3000 -x 2900 -l smallGrid

Where:
	-n = number of episodes
	-x = episodes used for training (graphics = off)

Remarks:
	the model has already been trained and wins most of the time
	the model has been optimized, it requires less then 30k episodes to converge


================================================================================

I used the 'SMOOTHL1LOSS' from torch.nn in the 'pacmanDQN_Agents.py' file to compute our loss, which creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise. 
It is less sensitive to outliers than torch.nn.MSELoss and in some cases prevents exploding gradients. So, instead of learning as the agent plays Pac-man, 
it’s actually just moving Pac-man around the screen based on what it has already learned, but adding all of those experiences to the buffer. 
Then, I can take experiences from storage and replay them to the agent so that it can learn from them and take better actions in the future.

As I start from the beggining the graphs of loss/iteration starts to converge after some time and then there's a point where the graphs converge and after a that certain point even if I train the model more and more it doesn't effect the graphs anymore.

I tried the classical DQN architecture uses a single NN to predict directly the value of all possible actions.

DQN.py and pacmanDQN_Agents.py are the two main files that I added which does most of our NN work. 

Procedure for running and basic code info has been defined above.
