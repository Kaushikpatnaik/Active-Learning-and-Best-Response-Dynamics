import sys

from classrooms import *
from learners import *
from datasets import *
from stat_trackers import *
from svmutil import *
import math
#from memory_profiler import profile
#from pympler import muppy
#from pympler import asizeof
#from pympler import summary
#from pympler.classtracker import ClassTracker
import copy

def compare_passive_learners(learner_types, dataset, num_trials, train_size, test_size, display = False):
    
    trackers = [StatTracker() for l in learner_types]
    
    for trial in range(num_trials):
        dataset.initialize()
        
        for tracker, learner_type in zip(trackers, learner_types):
            learner = learner_type()
            classroom = PassiveSupervisedBatchClassroom(learner, dataset, tracker, train_size, test_size)
            classroom.learn()
            classroom.test()
            #classroom.separatorprint()
            if display:
                sys.stdout.write('.')
                sys.stdout.flush()
    if display:
        print
    
    return trackers


def compare_active_source_learners(learner_types, dataset, num_trials, label_budget, test_size, display = False):
    
    trackers = [StatTracker() for l in learner_types]
    
    for trial in range(num_trials):
        dataset.initialize()
        
        for tracker, learner_type in zip(trackers, learner_types):
            learner = learner_type()
            classroom = ActiveSourceClassroom(learner, dataset, tracker, label_budget, test_size)
            classroom.learn()
            classroom.test()
            #dataset.iterate()
            if display:
                sys.stdout.write('.')
                sys.stdout.flush()
    if display:
        print
    
    return trackers
    
def compare_active_noise_linear_learners(learner_types, dataset, num_trials, label_budget, test_size, eps, display = False):
    
    trackers = [StatTracker() for l in learner_types]
    
    for trial in range(num_trials):
        dataset.initialize()
        
        for tracker, learner_type in zip(trackers, learner_types):
            learner = learner_type()
            classroom = ActiveNoiseLinearSourceClassroom(learner, dataset, tracker, label_budget, test_size, eps)
            classroom.learn()
            classroom.test()
            if display:
                sys.stdout.write('.')
                sys.stdout.flush()
    if display:
        print
    
    return trackers

def compare_active_batch_learners(learner_types, dataset, num_trials, num_unlabeled, label_budget, test_size, display = False):
    
    trackers = [StatTracker() for l in learner_types]
    
    for trial in range(num_trials):
        dataset.initialize()
        
        for tracker, learner_type in zip(trackers, learner_types):
            dataset.initialize(shuffle = False)
            learner = learner_type()
            tracker.set_name(learner)
            classroom = ActiveBatchClassroom(learner, dataset, tracker, num_unlabeled, label_budget, test_size)
            classroom.learn()
            classroom.test()
            #classroom.separator_inner_iter_print()
            if display:
                sys.stdout.write('.')
                sys.stdout.flush()
    if display:
        print
    
    return trackers


def mistake_bound_run(learner_type, dataset, num_trials, num_iters):
    tracker = MistakeTracker()
    for i in range(num_trials):
        dataset.initialize()
        learner = learner_type()
        classroom = IterativeTrackingClassroom(learner, dataset, tracker, num_iters)
        classroom.learn()
    
    return tracker

#@profile
def libsvm_angle_compare_test(learner_types, dataset, num_trials, num_unlabeled, label_budget, w_star, display = False):
    
    #trackers = [StatTracker() for l in learner_types]
    tracker1 = []
    acc= []
    
    
    for trial in range(num_trials):
        local_data = copy.copy(dataset)
		
        local_data.initialize()
        
        learner1 = learner_types[0]()
        classroom1 = ActiveBatchClassroom(learner1, local_data, tracker1, num_unlabeled, label_budget, 0)
        classroom1.learn()
        if display:
            sys.stdout.write('.')
            sys.stdout.flush()
       
        
        # Calculating difference in angle for the theoretical algorithm
        algo_w = learner1.w
        #print algo_w
        angle_diff_algo = 0
        
        angle_diff_algo = dot(w_star, algo_w)
        angle_diff_algo /= norm(w_star,2) * norm(algo_w,2)
        angle_diff_algo = math.acos(angle_diff_algo)
        #print angle_diff_algo
        
        local_data.initialize(shuffle = False)
        
        x_train =[]
        y_train =[]
        for i in range(label_budget):
            temp1, temp2 = local_data.next()
            x_train.append(temp1.tolist())
            y_train.append(temp2)
        
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-q -t 0 -c 10')
        m = svm_train(prob, param)
        # libsvm returns the support vectors as a list of dictionaries
        # and coefficients as a list of tuples
        support = m.get_SV()
        coef = m.get_sv_coef()
        labels = m.get_nr_class()
                
        support_arr = zeros((len(coef), 2))
        coef_arr = zeros(len(coef))
        angle_diff_svm = 0
        svm_w = zeros(2)
        
        #Calculating distance between the best separator and the svm and algorithm weight vectors
        # for svm
        # checked for non-zero bias - not present
        # checked for mismatch between coef and support vector length - none
        
        for i in range(len(coef)):
			for key in support[i].keys():
				if key != -1:
					support_arr[i][key-1] = support[i][key]
					#print support[i][key]
			coef_arr[i] = coef[i][0]
			
			if support[i][-1] != 0:
				print support[i][-1]
			
			svm_w += coef_arr[i] * support_arr[i]
		
		
        angle_diff_svm = dot(w_star, svm_w)
        angle_diff_svm /= norm(w_star, 2) * norm(svm_w, 2)
        angle_diff_svm = math.acos(angle_diff_svm)
        
        simple_margin_diff = 0
        acc.append([angle_diff_svm, angle_diff_algo, simple_margin_diff])
        #print angle_diff_algo, angle_diff_svm
        
        local_data = None
      
        
        
    if display:
        print
    return acc

def libsvm_compare_learners(learner_types, dataset, num_trials, num_unlabeled, label_budget, test_size, display = False):
    
    trackers = [StatTracker() for l in learner_types]
    acc= []
    return_acc_tuple = []
    d = dataset.d
    
    for trial in range(num_trials):
        local_data = copy.copy(dataset)
		
        local_data.initialize()
        
        avgpointsusage = []
    
        for tracker1, learner_type1 in zip(trackers, learner_types):
            local_data.initialize(shuffle = False)
            learner1 = learner_type1()
            classroom1 = ActiveBatchClassroom(learner1, local_data, tracker1, num_unlabeled, label_budget, test_size)
            classroom1.learn()
            classroom1.test()
            if display:
                sys.stdout.write('.')
                sys.stdout.flush()
        
        x = zeros((num_unlabeled, d))
        y = zeros(num_unlabeled)

        local_data.initialize(shuffle = False)
        for i in range(num_unlabeled):
            temp1, temp2 = local_data.next()
            x[i] = temp1
            y[i] = temp2
    
        x_mod = x.tolist()
        y_mod = y.tolist()
        
        x_train = x_mod[0:label_budget]
        y_train = y_mod[0:label_budget]
        x_test = x_mod[label_budget:label_budget+test_size]
        y_test = y_mod[label_budget:label_budget+test_size]

        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-q -t 2 -g 0.1 -c 10')
        m = svm_train(prob, param)
        
        p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
        acc.append(p_acc[0])

    accuracy = sum(acc[:]*test_size)/(test_size*num_trials)
    print accuracy

    return trackers















