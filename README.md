Active-Learning-and-Best-Response-Dynamics
==========================================

Code for NIPS'14 paper - Active Learning and Best Response Dynamics

The paper examines a setting with many low-powered distributed sensors. The sensors can communicate locally, and querying them from a central agent is costly. In addition to local communication only, the sensors make noisy readings (label noise). The goal is to detect a linear hyperplane classifying the sensors with least possible queries. 

We approach the problem by first denoising the sensor readings by playing a local consensus game. Each sensor communicates its readings to all neighbors within communication range r, and updates its own label based on majority rule. This consensus dynamic can be synchronous (all at the same time) or asynchronous in nature. This dynamic can be run for several iterations to reach an equilibrium state.

Post denoising we are able to prove that, upon certain conditions, the overall noise decreases. Also sensors at a distance >2r from the true hyperplane will have no error. 

This reduction in noise level enables the use of active learning algorithms which are agnostic to noise, and have provable guarantees for learning the true separator.

We are able to show experimentally that the combination of denoising and active learning outperforms denoising and passive learning, achieving 33% of the generalization error for low label budgets. 


