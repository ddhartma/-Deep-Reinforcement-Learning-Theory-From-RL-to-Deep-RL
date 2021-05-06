[image1]: assets/intro_deep_rl.png "image1"
[image2]: assets/replay_buffer.png "image2"
[image3]: assets/fixed_targets.png "image3"
[image4]: assets/fixed_targets_eq.png "image4"
[image5]: assets/optimal_action_value_func.png "image5"
[image6]: assets/loss_at_iter_i.png "image6"
[image7]: assets/dqn_conv_net.png "image7"
[image8]: assets/result_paper.png "image8"


# Deep Reinforcement Learning Theory - Deep Q-Networks

## Content
- [Introduction](#intro)
- [From RL to Deep RL](#from_rl_to_deep_rl)
- [Deep Q-Networks](#deep_q_networks)
- [Experience Replay](#experience_replay)
- [Fixed Q Targets](#fixed_q_targets)
- [Reference: Human-level control through deep reinforcement learning](#paper)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network


## From RL to Deep RL <a name="from_rl_to_deep_rl"></a>
What is Deep RL? 
- In some sense, it is using **nonlinear function approximators** to calculate the value actions based directly on observation from the environment.
- This is represented as a **Deep Neural Network**.
- **Deep Learning** to find the **optimal parameters** for these function approximators.
- Supervised Deep Learning concepts use label training data for supervised learning. --> pixels-to-labels
- When an oral agent handles the entire end-to-end pipeline, it's called pixels-to-actions, referring to the networks ability to take raw sensor data and choose the action,
it thinks will best maximize its reward.

    ![image1]

- Some literature:
    - [Neural Fitted Q Iteration - First Experienceswith a Data Efficient Neural ReinforcementLearning Method](http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf)
    - [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


## Deep Q-Networks <a name="deep_q_networks"></a> 
### Neural Networks
- In 2015, Deep Mind made a breakthrough by
designing an agent that learned to play video games better than humans.
- They call this agent a Deep Q Network.
- At the heart of the agent is **a deep neural network that acts as a function approximator**.
- You **pass in images from the video game** one screen at a time,
and it produces **a vector of action values**, with the **max value indicating the action to take**.
- As a reinforcement signal, it is fed back the change in game score at each time step.
- In the beginning when the neural network is **initialized with random values**, the actions taken are all over the place.
- Overtime it begins to **associate situations and sequences** in
the game with appropriate actions and learns to actually play the game well.

### Complexity
- Atari games are displayed at a resolution of 210 by 160 pixels, with 128 possible colors for each pixel.
- This is still technically **a discrete state space but very large**.
- To **reduce complexity**,
    - convert the frames to **gray scale**,
    - scale them down to **a square 84 by 84** pixel block.
    - use GPU

### Memory
- Give the agent **access to a sequence of frames**, they **stacked four such frames together** resulting in
a final state space size of **84 by 84 by 4**.

### All at once
- Unlike a traditional reinforcement learning setup where only one Q value is produced at a time, the Deep Q network is designed to produce a **Q value for every possible action in a single forward pass**.
- Without this, one would have to run the network individually for every action.
- Use this vector to take an action, either stochastically, or by choosing the one with the maximum value. 

### Convolutional approach:
- The screen images are first processed by **convolutional layers**.
- This allows the system to exploit spatial relationships.
since four frames are stacked and provided as input,
these convolutional layers also extract some temporal properties across those frames.
- The original DQN agent used **three convolutional layers** with **ReLU** activation (regularized linear units).
- Then followed by **one fully-connected hidden layer with ReLU activation**,
- Then **one fully-connected linear output layer** that produced the vector of action values.

- This same architecture was used for all the Atari games they tested on, but each game was learned from scratch with a freshly initialized network.

### Modifications due to unstable and ineffective policy
- Training such a network requires a lot of data,
but even then, it is not guaranteed to converge on the optimal value function.
- In fact, there are situations where the network weights can oscillate or diverge, due to the high correlation between actions and states.
- This can result in a very unstable and ineffective policy.

- Two succesful modifications: 
    - Experience replay
    - Fixed Q targets


## Experience Replay <a name="experience_replay"></a> 
- In basic Q-learning algorithm the agent 
    - interacts with the environment and at each time step and obtains a state action reward
    - learns from it
    - moves on to the next tuple in the following timestep.
- This seems a little wasteful.
- Agent could learn more from these experienced tuples if we stored them somewhere.
- Moreover, some states are pretty rare to come by and some actions can be pretty costly, so it would be nice to recall such experiences.

### Replay buffer (store and sample)
- **Store** each experienced tuple in this buffer 
- Then **sample** a small batch of tuples from it in order to learn.
- Agent is able to learn from individual tuples multiple times,
recall rare occurrences, and in general makes better use of fire experience.
- Consider: a sequence of experienced tuples can be highly correlated
- By keeping track of a **replay buffer** and using **experience replay** to **sample from the buffer at random**, we can prevent action values from oscillating or diverging catastrophically.

    ![image2]

## Fixed Q Targets <a name="fixed_q_targets"></a>
- Experience replay helps us to address one type of correlation:
That is between consecutive experience tuples.
- There is another kind of correlation that Q-learning is susceptible to.
- Q-learning is a form of **Temporal Difference or TD learning**
- **Goal** Reduce the difference (TD error) between the **TD target** and the currently **predicted Q-value**.

    ![image3]

- In Q-Learning, we update a **guess** with a **guess**, and this can potentially lead to **harmful correlations**. 
- To avoid this, we can update the parameters *w** in the network **q<sup>^</sup>** to better approximate the action value corresponding to state **S** and action **A** with the following update rule:

    ![image4]

    where **w<sup>−</sup>** are the weights of a separate target network that are not changed during the learning step, and **(S, A, R, S′)** is an experience tuple.

## Reference: Human-level control through deep reinforcement learning <a name="paper"></a> 
- Check the following reference: [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

### Idea of this paper:
- **Idea from biology**: Human decisions are based on reinforcemnt learning (dopaminergetic neuroms and temporal difference RL) and 
- **Basis**: 
   - Use deep Q-networks (DQN) 
    - DQNs learn successful policies from sensor inputs using end-to-end RL
    - Atari 2600 games are the baseline for this study (49 games)
    - Use same algorithm, network architecture and hyperparameters for all 49 games
- **Input**: 
    - Pixels (210 x 160) colour video at 60 Hz)
    - Game score
- **Preprocessing**:
    - Raw Atari2600 frames 210xx 160 pixel images with a 128-colour palette, can be demanding in terms of computationand memory requirements
    - We apply a basic preprocessing step aimed at reducing the input dimensionality and dealing with some artefacts of the Atari 2600 emu-lator. First, to encode a single frame we take the maximum value for each pixel colour alue over the frame being encoded and the previous frame. This was necessary toremove flickering that is present in games where some objects appear only in evenframes while other objects appear only in odd frames, an artefact caused by thelimited number of sprites Atari 2600 can display at once. Second, we then extractthe Y channel, also known as luminance, from the RGB frame and rescale it to84384. The functionwfrom algorithm 1 described belowappliesthis preprocess-ing to themmost recent frames and stacks them to produce the input to theQ-function, in whichm5
- **Architecture**:
    - DQN by using
    - Deep neural convolutional networks with stochastic gradient descent
- **Workflow**:
    - Agent interacts with environment
    - Observations
    - Actions
    - Rewards
- **Goal of agent**: select actions which maximize cumulative future rewards
- **Startegy**: 
    - Use deep convolutional neural network to approximate the **optimal action value function**

        ![image5]
        which is the maximum sum of rewards **r<sub>t</sub>** discounted by **γ** at each time-step **t**, achievable by a behaviour policy **π=P((a|s)**, after making an observation **s** and taking an action **a**.

    - Get a **guess** of the action value function **Q(s,a,θ<sub>i</sub>)** using the neural network shown in figure below
    - **θ<sub>i</sub>** are weights of the Q-network at iteration **i**
    - Use **Experience Replay** and **Fixed Targets** due to instabilities (triggered by correlations) 
        - Store agents experience **e<sub>t</sub> = (s<sub>t</sub>,a<sub>t</sub>, r<sub>t</sub>,s<sub>t+1</sub>)** at time step **t**
        - Store it in data set **D<sub>t</sub> = {e<sub>1</sub>, ..., e<sub>t</sub>}** 
        - Apply **Q-Learning updates** on samples of experience drawn from **D<sub>t</sub> uniformly at **random**
        - **Loss function** at iteration **i**:
        ![image6]

            **γ**: discount factor determining the agent’s horizon

            **θ<sub>i</sub>**: weights of the Q-network at iteration **i** 

            **θ<sub>i</sub><sup>-</sup>**: the network parameters **used to** compute the target at iteration **i**. 
        - The target net-work parameters **θ<sub>i</sub><sup>-</sup>** are only updated with the Q-network parameters **θ<sub>i</sub>** every **C** steps and are **held fixed between individual updates**.

    ![image7]

- **Result** of DQN based on conv layer with stochastic gradient descent:
    - stable result
    - Figure below show **temporal evolution** of 
        - **average score per episode** and 
        - **average predicted Q-values**

    ![image8]

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)
