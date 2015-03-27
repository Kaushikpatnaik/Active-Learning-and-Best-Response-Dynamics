Active-Learning-and-Best-Response-Dynamics
==========================================

Code for NIPS'14 paper - Active Learning and Best Response Dynamics

[Link to Paper](http://arxiv.org/abs/1406.6633)

The paper examines a setting with many low-powered distributed sensors. The sensors can communicate locally, and querying them from a central agent is costly. In addition to local communication only, the sensors make noisy readings (label noise). The goal is to detect a linear hyperplane classifying the sensors with least possible queries. 

We approach the problem by first denoising the sensor readings by playing a local consensus game. Each sensor communicates its readings to all neighbors within communication range r, and updates its own label based on majority rule. This consensus dynamic can be synchronous (all at the same time) or asynchronous in nature. This dynamic can be run for several iterations to reach an equilibrium state.

Post denoising we are able to prove that, upon certain conditions, the overall noise decreases. Also sensors at a distance >2r from the true hyperplane will have no error. 

This reduction in noise level enables the use of active learning algorithms which are agnostic to noise, and have provable guarantees for learning the true separator.

We are able to show experimentally that the combination of denoising and active learning outperforms denoising and passive learning, achieving 33% of the generalization error for low label budgets. 

-----------------------------------------------

To run the experiments demonstrating the results,

Run game_theory from terminal as "python game_theory.py" with parameters of your choice.

The program takes 5 parameters -

i) Initial noise method for labels - 
    Option 1 - Random label noise across all sensors
    Option 2 - Noise based on distance from the true hyperplane
    Option 3 - Pockets of noise

ii) Communication protocol between sensors -
    Option 1 - Sensors communicate their labels to all sensors within radius 0.1
    Option 2 - Sensors communicate their labels to 20 nearest neighbors
    Option 3 - Sensors communicate their labels randomly based on distance

iii) Type of dynamic - Update sensors based on 
    Option 1 - Majority rule of all sensors within the communication radius
    Option 2 - Weighted majority rule of all sensors within communication radius
    Option 3 - Probabilistic version of k-nearest neighbors

iv) Type of update - Update sensors either
    Option 1 - Synchronous updates
    Option 2 - Asynchronous updates

v)  Number of sensors - Input an integer

----------------------------------------------

File Descriptions and other information

file: active_learners.py
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

Since the initial experiments were around different initializations of parameters, the implementation in active_learner.py 
contains a base class implementation from which all other all classes are derived. The base class is

class MarginBasedActiveLearnerBase -

Base implementation of the Margin Based Method incorporating different strategies and options for execution
Based on STOC 2014 paper - The power of localization in effectively learning linear separators

Initialization of initial weight vector based on the averaging algorithm
Hinge loss solver can be - cvxopt, sgd, hard_cvxopt and svm
Implements outlier removal

Has two derived classes, for adverserial and outlier noise:
    class MarginBasedOutlierRemoval(MarginBasedActiveLearnerBase):
    class MarginBasedBasic(MarginBasedActiveLearnerBase):

The above two classes are used as parent classes of other classes with another class providing the base for the 
initialization of variables. For example see the class below

---------------------------------------------------------------------------------------------------------------------------
class Theoretical -

Initializes the variables in terms of guidance provided in the paper

Has two derived classes, which need to be called during experiments
    class MarginBasedTheoreticalParams(MarginBasedBasic, Theoretical)
    class MarginBasedTheoreticalParamsOR(MarginBasedOutlierRemoval, Theoretical)

PLEASE USE THESE CLASSES WHILE PERFORMING EXPERIMENTS FOR THEORETICAL PARAMETERS

---------------------------------------------------------------------------------------------------------------------------
the file also contains implementations of SVM which are efficient for small number of samples. It also contains
implementations of different active learning procedures given in Tong and Koller

Self Implementation of the SVM - 
class PassiveSVM

Implementation of the simple margin procedures given in Tong and Koller -

Batch Methods -
class SimpleMarginBatch
class SimpleMarginSoftSVMBatch
class MaxMinMarginSoftSVMBatch
class RatioMarginSoftSVMBatch

Source Methods -
class SimpleMarginSource
class SimpleMarginSoftSVMSource
class AverageMarginSoftSVMSource

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------


file game_theory.py
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

The program takes 5 parameters -

i) Initial noise method for labels - Option 1 - Random label noise across all sensors Option 2 - Noise based on distance from the true hyperplane Option 3 - Pockets of noise

ii) Communication protocol between sensors - Option 1 - Sensors communicate their labels to all sensors within radius 0.1 Option 2 - Sensors communicate their labels to 20 nearest neighbors Option 3 - Sensors communicate their labels randomly based on distance

iii) Type of dynamic - Update sensors based on Option 1 - Majority rule of all sensors within the communication radius Option 2 - Weighted majority rule of all sensors within communication radius Option 3 - Probabilistic version of k-nearest neighbors

iv) Type of update - Update sensors either Option 1 - Synchronous updates Option 2 - Asynchronous updates

v) Number of sensors - Input an integer

-------------------------------------------------------------------------------------------------------------------------
Class definitions

class Create_Noisy_Labels(object) - contains the different methodologies for generating noise (ball, line, and random)

class Consensus_Dynamics(object) - contains different methods for running the consensus dynamics both synchronous 
and asynchronous (majority, weighted majority, and probabilistic k-NN)

class create_nn_graph(object) - different methods of creating nearest neighbor graphs (radius, k-NN, and probabilistic)
outputs a dictionary containing nearest neighbors of all points

procedures for training classifiers
def train_classifiers_gen_error(dist, inf_y, opt_w, label_budget, internal_iters =6, num_trials=50)
def train_classifiers_class_error(dist, inf_y, label_budget, internal_iters =6, num_trials=50)

procedure for calculating final noise level
def calculate_noisy(y, inf_y)

-------------------------------------------------------------------------------------------------------------------------

Other files which are important for adding new synthethic datasets, noise types etc

file synthetic_experiments.py
-------------------------------------------------------------------------------------------------------------------------
File contains sample code for running experiments on synthetic datasets.

To add a new synthetic dataset, please add a class similar to class GaussianLinearSep(DataSet) in datasets.py

file adversary.py
-------------------------------------------------------------------------------------------------------------------------
File contains base class for adversary as well as separate classes for different noise types like adverserial, malicious

To add a new noise type, please look at class MarginLinearMaliciousNoise(Adversary) in adversary.py
The adversary must be specified in the datset definition for the noise to be included

file stat_tracker.py
-------------------------------------------------------------------------------------------------------------------------
File contains base class for tracking different parameters for each learner

file passive_learners.py
-------------------------------------------------------------------------------------------------------------------------
File contains classes for outlier removal method and also Quadratic program used in the implmentation of Margin Based Algorithm

Also contains other optimization methods like SGD.

