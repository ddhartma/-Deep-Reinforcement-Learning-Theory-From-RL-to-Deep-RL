[image1]: assets/intro_deep_rl.png "image1"
[image2]: assets/replay_buffer.png "image2"
[image3]: assets/fixed_targets.png "image3"
[image4]: assets/fixed_targets_eq.png "image4"
[image5]: assets/optimal_action_value_func.png "image5"
[image6]: assets/loss_at_iter_i.png "image6"
[image7]: assets/dqn_conv_net.png "image7"
[image8]: assets/result_paper.png "image8"
[image9]: assets/deep_q_network.png "image9"
[image10]: assets/future_disc_return.png "image10"
[image11]: assets/future_disc_return_bell.png "image11"
[image12]: assets/future_disc_return_bell_2.png "image12"
[image13]: assets/loss_func.png "image13"
[image14]: assets/gradient.png "image14"
[image15]: assets/algo_1.png "image15"
[image16]: assets/algo_video.png "image16"

# Deep Reinforcement Learning Theory - Deep Q-Networks

## Content 
- [Introduction](#intro)
- [From RL to Deep RL](#from_rl_to_deep_rl)
- [Deep Q-Networks](#deep_q_networks)
- [Experience Replay](#experience_replay)
- [Fixed Q Targets](#fixed_q_targets)
- [Reference: Human-level control through deep reinforcement learning](#paper)
- [The DQN Algorithm](#algo_1)
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

    ![image9]

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
    - Prior knowledge relies only in:
        - visual images (conv network)
        - game specific score
        - number of actions 
        - life count

- **Strategy**: 
    - Use deep convolutional neural network to approximate the **optimal action value function**

        ![image5]

        which is the maximum sum of rewards **r<sub>t</sub>** discounted by **γ** at each time-step **t**, achievable by a behaviour policy **π=P((a|s)**, after making an observation **s** and taking an action **a**.

    - Get a **guess** of the action value function **Q(s,a,θ<sub>i</sub>)** using the neural network shown in figure below
    - **θ<sub>i</sub>** are weights of the Q-network at iteration **i**
    - Use **Experience Replay** and **Fixed Targets** due to instabilities (triggered by correlations) 
        - Store agents experience **e<sub>t</sub> = (s<sub>t</sub>,a<sub>t</sub>, r<sub>t</sub>,s<sub>t+1</sub>)** at time step **t**
        - Store it in data set **D<sub>t</sub> = {e<sub>1</sub>, ..., e<sub>t</sub>}** 
        - Apply **Q-Learning updates** on samples of experience drawn from **D<sub>t</sub>** uniformly at **random**
        - **Loss function** at iteration **i**:

            ![image6]

            **γ**: discount factor determining the agent’s horizon

            **θ<sub>i</sub>**: weights of the Q-network at iteration **i** 

            **θ<sub>i</sub><sup>-</sup>**: the network parameters **used to** compute the target at iteration **i**. 
        - The target network parameters **θ<sub>i</sub><sup>-</sup>** are only updated with the Q-network parameters **θ<sub>i</sub>** every **C** steps and are **held fixed between individual updates**.

    ![image7]

- **Result** of DQN based on conv layer with stochastic gradient descent:
    - stable result
    - Figure below show **temporal evolution** of 
        - **average score per episode** and 
        - **average predicted Q-values**

    ![image8]
- **Preprocessing**:
    - Raw Atari2600 frames 210xx 160 pixel images with a 128-colour palette, can be demanding in terms of computationand memory requirements. 
        - Reduce the input dimensionality
        - Reduce artefacts of the Atari 2600 emulator.
    - First: Take maximum value for each pixel colour over the actual and the previous frame to remove flickering of objects (which appear only in even frames while other objects appear only in odd frames)
    - Second: Extract the Y channel, also known as luminance, from the RGB frame and rescale it to **84 x 84** (greyscale). 
    - Take the **m=4** most recent frames and stacks them to produce the input to the Q-function
    - Function **Φ** applies the preprocessing to the 4 most recent frames
- Code availability: [download](https://sites.google.com/a/deepmind.com/dqn/)
- **Input**: 
    - Pixels (210 x 160) colour video at 60 Hz)
    - Game score
- **Model architecture**:
    - DQN by using
    - Deep neural convolutional networks with stochastic gradient descent
    - Compute Q-values for all possible actions in a given state with only a single forwatd pass through the network
        - Use a **separate output unit** for each possible **action**
        - Only **state representation is an input** to the neural network
    - **Input**: 84x84x4 images produced by the preprocessing map **Φ**
    - **1st hidden layer**: conv layer, 32 filter, 8x8 kernel, stride 4, ReLU
    - **2nd hidden layer**: conv layer, 64 filter, 4x4 kernel, stride 2, ReLU,
    - **3rd hidden layer**: conv layer, 64 filter, 4x4 kernel, stride 1, ReLU
    - **4th hidden layer**: fully connected, 512 rectifier units
    - **Output layer**: linear layer with a single output for each action
    - Number of outputs vary between 4 and 18 depending on the game
- **Training details**:
    - Use same network architecture, learning rate and hyperparameters for all 49 games
    - Scale (clip) scores (as they vary from game to game). This limits scale of error derivatives (therefore same learning rate can be used)
        - Positive reward to 1
        - Negative reward to -1 
        - No reward = 0
    - Optimizer: RMSprop with minibatch size 32
    - **ε-greedy** policy, scaled linear from 1.0 to 0.1 over the first million frames, fixed at 0.1 thereafter
    - Number of frames played: 50 million (38 days)
    - Number of frames for replay buffer (1 million most recent frames)
    - Frame skipping technique: Agent sees and selects actions on every **k**th frame instead of every frame (here: **k=4**). Last action is repeated on skipped frames. --> Reduces computation, reduces runtime, without lack of learning 
    - No systematic grid search for hyperparameter choice (all fixed for all games)

- **Evaluation procedure**:
    - Trained agents were evaluated by playing each game **30 times** for up to **5min** each with different initial random conditions and an **ε-greedy** policy with**ε=0.05**
    - Used to minimize overfitting effects during evaluation.
    - This random agent is used as a baseline comparison (with random action at 10 Hz, every 6th frame)
- **Goal of agent**: select actions which maximize cumulative future rewards

- **Algorithm**
    - Agent interacts with environment (Atari emulator) in a sequence of 
        - actions
        - observations
        - rewards
    - At each time-step the agent selects an action **a<sub>t</sub>** from the set of legal game actions **A={1, ..., K}**. 
    - Action is passed to the emulator. Emulator answers with an image **x<sub>t</sub>** from the emulator, which is a vector of pixel values representing the current screen and a reward **r<sub>t</sub>** from game score
    - Agent only observes the current screen.
    - Game score may depend on the whole previous sequence of actions and observations
    - Therefore, sequences of actions and observations,**s<sub>t</sub>~x<sub>1</sub>,a<sub>1</sub>,x<sub>2</sub>,...,a<sub>t-1</sub>, x<sub>t</sub>**, are input to the algorithm
    - Algorithm then learns game strategies depending upon these sequences. 
    - All sequences in the emulator are assumed to terminate in a finite number of time-steps.
    - **Goal of agent**: select actions which maximize cumulative future rewards
    - Future rewards are discounted by a factor of **γ** per time-step (**γ**=0.99)
    - **Future discounted return** at time **t** as 

        ![image10]
    
        **T**: time-step at which the game terminates. 
    - Optimal action-value function **Q<sup>*</sup>(s,a)** as the maximum expected return achievable by following any policy, after seeing some sequences and then taking some action **a**,

        ![image5]
    
    - The optimal action-value function obeys an important identity known as the Bellman equation. This is based on the following intuition: if the optimal value **Q<sup>*</sup>(s',a')** of the sequences **s'** at the next time-step was known for all possible actions **a'**, then the optimal strategy is to select the action **a'** maximizing the expected value of **r + γ Q<sup>*</sup>(s',a')**:

        ![image11]

    - Estimate the action-value function by using the Bellman equation as an iterative update

        ![image12]

    - In Theory: Such value iteration algorithms converge to the optimal action-value function **Q<sub>i</sub> -> Q<sup>*</sup>** for **i -> ∞**
    - In Practice: Use function approximation to estimate the action-value function **Q(s,a,θ) ≈ Q<sup>*</sup>(s,a)**

    - Use a nonlinear function approximator such as a neural network
    - A neural network function approximator with weights **θ** is a Q-network
    - A Q-network can be trained by adjusting the parameter **θ<sub>i</sub>** at iteration **i** to reduce the mean-squared error in the Bellman equation, where the optimal target values 
    **r + γ max<sub>a'</sub>Q<sup>*</sup>(s',a')** are substituted by approximate target values **r + γ max<sub>a'</sub>Q(s',a', θ<sub>i</sub><sup>-</sup>)**
    using parameters **θ<sub>i</sub><sup>-</sup>** from some previous iteration (**Fixed Targets**).
    
    - Use **Experience Replay** and **Fixed Targets** due to instabilities (triggered by correlations) 
        - Store agents experience **e<sub>t</sub> = (s<sub>t</sub>,a<sub>t</sub>, r<sub>t</sub>,s<sub>t+1</sub>)** at time step **t**
        - Store it in data set **D<sub>t</sub> = {e<sub>1</sub>, ..., e<sub>t</sub>}** 
        - Apply **Q-Learning updates** on samples of experience drawn from **D<sub>t</sub>** uniformly at **random**
        - **Loss function** at iteration **i**:

            ![image13]

            Note that the targets depend on the network weights; this is in contrast with the targets used for supervised learning, which are fixed before learning begins. At each stage of optimization, we hold the parameters from the previous iteration **θ<sub>i</sub><sup>-</sup>**  fixed when optimizing the **i**th loss function **L<sub>i</sub>(θ<sub>i</sub>), resulting in a sequence of well-defined optimization problems. The final term is the variance of the targets, which does not depend on the parameters θ<sub>i</sub> we are currently optimizing,and may therefore be ignored

    - Differentiating the loss function with respect to the weights results in the following gradient:

        ![image14]

    - Optimize the loss function by stochastic gradient descent. The familiar Q-learning algorithm can be recovered in this framework by updating the weights after every time step, replacing the expectations using single samples, and setting **θ<sub>i</sub><sup>-</sup> = θ<sub>i-1</sub>**.

- **Training algorithm for deep Q-networks**
    - The agent selects and executes actions according to an **ε-greedy** policy based on **Q**. Because using histories of arbitrary length as inputs to a neural network can be difficult, our Q-function instead works on a fixed length representation of histories produced by the function **Φ** described above. 
    - The algorithm modifies standard online Q-learning in two ways to make itsuitable for training large neural networks without diverging
    - First, we use a technique known as **experience replay** in which we store the agents experience **e<sub>t</sub> = (s<sub>t</sub>,a<sub>t</sub>, r<sub>t</sub>,s<sub>t+1</sub>)** at time step **t** in a data set **D<sub>t</sub> = {e<sub>1</sub>, ..., e<sub>t</sub>}** pooled over many episodes (where the end of an episode occurs when a terminal state is reached) into a replay memory
    - During the inner loop of the algorithm, we apply Q-learning updates, or minibatch updates, to samples of experience (s,a,r,s') drawn at random from the pool of stored samples. 
    - This approach has several advantages over standard online Q-learning. 
        - First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency. - Second, randomizing the samples breaks these correlations and therefore reduces the variance of the updates. 
        - Third, when learning on-policy the current parameters determine the next data sample that the parameters are trained on. 
    - By using experience replay the ehaviour distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters. 
    - The algorithm only stores the last **N** experience tuples in the replay memory, and samples uniformly at random from **D** when performing updates.
    - Use a **separate network for generating the targets** **y<sub>j</sub>** in the Q-learning update. More precisely, every **C** updates we clone the network **Q** to obtain a target network  **^Q** and use **^Q** for generating the Q-learning targets **y<sub>j</sub>** for the following **C** updates to **Q**. 
    - This modification makes the algorithm more stable compared to standard online Q-learning, where an update that increases **Q(s<sub>t</sub>,a<sub>t</sub>)** often also increases **Q(s<sub>t+1</sub>,a)** for all **a** and hence also increases the target **y<sub>j</sub>**, possibly leading to oscillations or divergence of the policy. Generating the targets using an older set of parameters adds a delay between the time an update to **Q** is made and the time the update affects the targetsyj,making divergence oroscillations much more unlikely.
    - **Clip** the error term from the update
     
        **r + γ max<sub>a'</sub>Q(s',a', &theta;<sub>i</sub><sup>-</sup>) - Q(s,a, &theta;<sub>i</sub>)**
    
        to be between -1 and 1. This form oferror clipping further improved the stability of the algorithm

## The DQN Algorithm <a name="algo_1"></a>
![image15]

![image16]



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
