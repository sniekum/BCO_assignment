# Homework 1 for CS 690 Human-Centric Machine Learning

### Recommended: Installation Ubuntu
First it is recommended that you install anaconda: <https://www.anaconda.com/products/distribution> a popular python distribution and software management platform.

Next, git clone this repository.

Next navigate to the repository
```
cd BCO_assignment
```
then install the dependencies and create a Conda environment called imitation_learning by running:
```
conda create -n imitation_learning python=3.9 -y
conda activate imitation_learning
pip install -r requirements.txt
```


### PyTorch Primer
If you have never used PyTorch before, I'd recommend going through the 60-minute blitz tutorial: <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>


### OpenAI Gym Primer
If you've never used OpenAI gym before, I'd recommmend reading through the beginning of this tutorial: <https://blog.paperspace.com/getting-started-with-openai-gym/>. You can stop when you get to the section on Wrappers.



Look at the code in `test_gym.py`. This code runs one episode of MountainCar (200 time steps) using a random policy that samples uniformly from the action space.

In MountainCar there are three actions: 0 (accelerate left), 1 (don't accelerate), 2 (accelerate right). You can read more about MountainCar here: <https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py>. The state-space is 2-dimensional: given a state s, s[0] is the x-position of the car and s[1] is the car's velocity.

The goal of MountainCar is to have the car drive to the flag on the top of the hill to the right. The car gets -1 reward for every step and gets 200 steps to try and get out of the hill. Because the reward is -1 per timestep the optimal policy is to get out of the valley in as few timesteps as possible.

<strong>You will need to type up your responses (preferably typeset in LaTeX) to the following parts of the homework and submit your responses and code via Gradescope. You are encouraged to talk about the homework with other students and share resources, but please do not share or copy code. As for code, submit the edited mountain_car_bco.py file. </strong>

## Part 1:

Run the following code again
```
python test_gym.py
```
What do you notice happening? What specifically about MountainCar makes the problem difficult for RL? 

## Part 2:
You will now learn how to solve the MountainCar task by driving the car yourself.
Run
```
python mountain_car_play.py
```
Use the arrow keys to control the car by accelerating left and right. Note that if you run out of time you get a reward of -200.0 and the car will reset to the bottom of the hill. If you get to the flag in less than 200 steps you can get a higher score. Once you reach the flag the environment will restart at the bottom of the hill.

Keep practicing until you can reliably get out of the valley. You can see your score for each epsiode output on the command line. 
Experiment with different strategies: e.g.
1. Going right, then left, then right all the way up the hill
2. Going left, then right all the way up the hill.
3. Left, right, left, right up the hill.
Which strategy do you like best? What is the best score you can get as a human demonstrator?

## Part 3: 

Now we will teach a simple behavioral cloning (BC) agent to drive itself out of the valley.

First take a look at `mountain_car_bc.py` and try to get a basic understanding of what is happening. By default this code will collect a single demonstration, then parse the states and actions into tensors for easy input to a PyTorch neural network. It then trains a policy to imitate the demonstrations and evaluates the policy by testing it on different initial states. The command line will output the average, min, and max total rewards.

Try it out by running
```
python mountain_car_bc.py
```
Try to provide a good demonstration. Then watch what the agent has learned. Does it do a good job imitating? Does it ever get stuck? What is the average, min, and max?

## Part 4

Let's give more than one demonstration. Run the following to give 5 good demonstrations. If you mess up during one demo, feel free to restart until you give 5 good demos and try to keep to a consistent strategy.
```
python mountain_car_bc.py --num_demos 5
```
Report the average min and max returns. Did it perform any better? Why or why not? Does the agent copy the strategy you used? 

## Part 5
What do you think will happen if we give good and bad demonstrations?
You will now give two demonstrations. For the first one, just press the right arrow key for the entire episode until it restarts. Then for the second demo, give a good demonstration that quickly gets out of the valley.
```
python mountain_car_bc.py --num_demos 2
```
What does the policy learn? Why might bad demonstrations be a problem? Briefly suggest one potential idea for making BC robust to bad demonstrations, as long as they are a minority of demonstrations.

## Part 6
At a conceptual level, describe what changes you would need to make to the BC code to implement BCO(0), as described in the paper we read in class: https://arxiv.org/pdf/1805.01954.pdf. Answer this question before starting Part 7.

## Part 7
Implement and test BCO(0) by modifying the starter code in `mountain_car_bco.py`. This starter code is identical to the BC code, except: (1) the function collect_random_interaction_data has been added to help with learning the inverse dynamics model; (2) a new argument for number of inverse dynamics training iterations has been added; and (3) some comments have been added about where to use the inverse dynamics model instead of ground-truth actions.  You may need to experiment with different neural network sizes and training iterations to get the inverse dynamics model to work well. Report how well it works and what you tried to get it to work, including stats on the accuracy of prediction when tested on the demonstration data (training should be done with the random interaction data, not the demonstrations).

## Submission
Prepare a PDF report with your answers to the questions (preferably typeset in LaTeX) and submit the PDF along with your code in a zip file. Submit via Gradescope.

### Acknowledgement
Based on an assignment originally created by Daniel S. Brown. 
