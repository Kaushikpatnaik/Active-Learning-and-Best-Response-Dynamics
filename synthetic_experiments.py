from experiments import *
from passive_learners import *
from active_learners import *
from online_learners import *
from real_data import *
from adversary import *
from kernels import *
import sys
from time import clock
#from memory_profiler import profile

#@profile
def test():

    # Learners to test
    
    learners = []
    
    # add as many learners as wanted for test
    learners.append(lambda: MarginBasedTheoreticalParams(d, num_iters))
    learners.append(lambda: MarginBasedTheoreticalParamsOR(d, num_iters))   
    learners.append(lambda: LinearNoiseMethodVarianceBatch(d, num_iters))
    learners.append(lambda: LinearNoiseMethodVarianceBatchOR(d, num_iters))

    trackers = compare_active_batch_learners(learners, dist, num_trials, num_unlabeled, label_budget, test_size, display = False)

    # Print the results
    
    for tracker in trackers:
        print
        tracker.display_aggregate_stats()
    print
    

if __name__ == '__main__':
    
    # Set up parameters
    num_trials = 5
    num_iters = 6
    eps = 0.01


    # Set up synthetic distribution
    num_unlabeled = 50000
    test_size = 1000
    
    d = 20
    w_star = ones(d)
    w_star /= norm(w_star, 2)
    adversary = MarginLinearLabelNoise(w_star, 0.0, 0.25)
    dist = GaussianLinearSep(d, w = w_star, noise = adversary)
    dist.normalize(2)  # Normalized Gaussian is uniform on unit sphere

    e = [200, 600, 800]
    
    for i in range(len(e)):
        label_budget = e[i]
        train_size = label_budget
        test()
        print "Adversary noise rate"
        print
        print adversary.noise_rate()

