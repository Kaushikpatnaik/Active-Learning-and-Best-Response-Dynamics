from time import time
from utilities import *
#from memory_profiler import profile
import gc
import sys
#from pympler import muppy
#from pympler import asizeof
#from pympler import summary

class Classroom(object):
    
    def __init__(self, learner, dataset):
        self.learner = learner
        self.moderator = dataset
    
    def learn(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError


class HoldOutClassroom(Classroom):
    
    def __init__(self, learner, dataset, tracker, test_size):
        self.learner = learner
        self.dataset = dataset
        self.tracker = tracker
        self.test_size = test_size
    
    def test(self):
        self.tracker.reset()
        for i in range(self.test_size):
            x, y = self.dataset.next()
            
            start = time()
            prediction = self.learner.classify(x)
            predict_time = time() - start
            
            self.tracker.update(y, prediction)
            self.tracker.update_times(predict_time, 0.0)
        
        self.tracker.record_performance()


class PassiveSupervisedBatchClassroom(HoldOutClassroom):
    
    def __init__(self, learner, dataset, tracker, train_size, test_size):
        HoldOutClassroom.__init__(self, learner, dataset, tracker, test_size)
        self.train_size = train_size
    
    def learn(self):
        X, Y = self.dataset.labeled_examples(self.train_size)
        self.learner.batch_train(X, Y)

    def separatorprint(self):
        U, ids = self.dataset.labeled_examples(self.train_size)
        display_distribution(U, ids, self.learner.w, self.train_size)

class SemiSupervisedBatchClassroom(HoldOutClassroom):
    
    def __init__(self, learner, dataset, tracker, num_unlabeled, num_labeled, test_size):
        HoldOutClassroom.__init__(self, learner, dataset, tracker, test_size)
        self.num_unlabeled = num_unlabeled
        self.num_labeled = num_labeled
    
    def learn(self):
        U = self.dataset.unlabeled_examples(self.num_unlabeled)
        L, Y = self.dataset.labeled_examples(self.num_labeled)
        self.learner.batch_train(U, L, Y)


class ActiveBatchClassroom(HoldOutClassroom):
    
    def __init__(self, learner, dataset, tracker, num_unlabeled, label_budget, test_size):
        HoldOutClassroom.__init__(self, learner, dataset, tracker, test_size)
        self.num_unlabeled = num_unlabeled
        self.label_budget = label_budget
        self.num_labels = 0
#        self.pointsusage = 0
    #@profile
    def learn(self):
		
        U, ids = self.dataset.unlabeled_examples(self.num_unlabeled, oracle = True)
        
        def oracle(example_id):
            if self.num_labels >= self.label_budget:
                raise ValueError, 'Label budget exceeded.'
            self.num_labels += 1
            return self.dataset.oracle[example_id]
        
        self.learner.active_batch_train(U, ids, oracle, self.label_budget)

    def separatorprint(self):
        U, ids = self.dataset.labeled_examples(self.num_unlabeled)
        display_distribution(U, ids, self.learner.w, self.num_unlabeled)


    def separator_inner_iter_print(self):
        U, ids = self.dataset.labeled_examples(self.num_unlabeled)
        display_inner_iterations(U, ids, self.learner.separators, self.num_unlabeled)


class ActiveSourceClassroom(HoldOutClassroom):
    
    def __init__(self, learner, dataset, tracker, label_budget, test_size):
        HoldOutClassroom.__init__(self, learner, dataset, tracker, test_size)
        self.label_budget = label_budget
        self.num_labels = 0
    
    def learn(self):
                
        def source(num_unlabeled):
			return self.dataset.unlabeled_examples(num_unlabeled, oracle = True)
        
        def oracle(example_id):
            if self.num_labels >= self.label_budget:
                raise ValueError, 'Label budget exceeded.'
            self.num_labels += 1
            return self.dataset.oracle[example_id]
        
        self.learner.active_source_train(source, oracle, self.label_budget)

class ActiveNoiseLinearSourceClassroom(HoldOutClassroom):
    
    def __init__(self, learner, dataset, tracker, label_budget, test_size, eps):
        HoldOutClassroom.__init__(self, learner, dataset, tracker, test_size)
        self.label_budget = label_budget
        self.num_labels = 0
        self.eps = eps
    
    def learn(self):
        
        def source(num_unlabeled):
            return self.dataset.unlabeled_examples(num_unlabeled, oracle = True)
        
        def oracle(example_id):
            if self.num_labels >= self.label_budget:
                raise ValueError, 'Label budget exceeded.'
            self.num_labels += 1
            return self.dataset.oracle[example_id]
        
        self.learner.active_noise_linear_train(source, oracle, self.label_budget, self.eps)

### Supervised online learning

class IterativeClassroom(Classroom):
    
    def __init__(self, learner, dataset, max_iters = 100, feedback = 2):
        self.learner = learner
        self.dataset = dataset
        self.max_iters = max_iters
        self.feedback_model = feedback
    
    def feedback(self, guess, y):
        if self.feedback_model == 2:
            return y
        elif self.feedback_model == 1:
            if guess == 1:
                return 0
            else:
                return y
        else:
            raise ValueError, 'Unrecognized feedback model: %s.' % self.feedback_model
    
    def learn(self, max_iters = None):
        max_iters = max_iters or self.max_iters
        for i in range(max_iters):
            try:
                x, y = self.dataset.next()
                fb = self.feedback(self.learner.classify(x), y)
                self.learner.update(x, fb)
            except StopIteration:
                break
        return i


class IterativeTrackingClassroom(IterativeClassroom):
    
    def __init__(self, learner, dataset, tracker = None, max_iters = 100, feedback = 2):
        self.learner = learner
        self.dataset = dataset
        self.max_iters = max_iters
        self.feedback_model = feedback
        self.tracker = tracker
        if self.tracker:
            self.tracker.reset()
    
    def learn(self, max_iters = None):
        max_iters = max_iters or self.max_iters
        for i in xrange(max_iters):
            try:
                x, y = self.dataset.next()
                
                start = time()
                prediction = self.learner.classify(x)
                predict_time = time() - start
                
                if self.tracker:
                    self.tracker.update(y, prediction)
                
                fb = self.feedback(prediction, y)
                
                start = time()
                self.learner.update(x, fb)
                update_time = time() - start
                
                if self.tracker:
                    self.tracker.update_times(predict_time, update_time)
            
            except StopIteration:
                max_iters = i
                break
        
        if self.tracker and max_iters > 0:
            self.tracker.record_performance()
























