from time import time


class StatTracker(object):
    
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.measures = [[] for i in range(8)]
        self.predict_time = 0
        self.update_time = 0
        self.learner_name = 'Unknown learner'
        self.noise_rate = None
        self.optimal_accuracy = None
    
    def reset(self):
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
    
    def update(self, true, prediction):
        if prediction > 0:
            if true > 0:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if true < 0:
                self.tn += 1
            else:
                self.fn += 1
    
    def update_times(self, predict, update):
        self.predict_time += predict
        self.update_time += update
    
    def record_performance(self):
        try:
            precision = self.tp / float(self.tp + self.fp)
        except ZeroDivisionError:
            precision = 0.0
        
        try:
            recall = self.tp / float(self.tp + self.fn)
        except ZeroDivisionError:
            recall = 0.0
        
        try:
            inv_precision = self.tn / float(self.tn + self.fn)
        except ZeroDivisionError:
            inv_precision = 0.0
        
        try:
            inv_recall = self.tn / float(self.tn + self.fp)
        except ZeroDivisionError:
            inv_recall = 0.0
        
        f_1 = f_measure(precision, recall, 1.0)
        inv_f_1 = f_measure(inv_precision, inv_recall, 1.0)
        n = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / float(n)
        #print '\n'
        #print 'Accuracy per iteration ' +str(accuracy)
        #print '\n'
        stats = [n, precision, recall, f_1, inv_precision, inv_recall, inv_f_1, accuracy]
        for lst, item in zip(self.measures, stats):
            lst.append(item)
        self.reset()
    
    def set_name(self, learner):
        self.learner_name = learner.__class__.__name__
    
    def display_aggregate_stats(self):
        trials = float(len(self.measures[1]))
        stats = [sum(lst) / trials for lst in self.measures[1:]]
        precision, recall, f_1, inv_precision, inv_recall, inv_f_1, accuracy = stats
        n = sum(self.measures[0])
        predict = self.predict_time / (n / 1000.0)
        update = self.update_time / (n / 1000.0)
        
        string = '{0:<17}{1:0.4f}'
        print self.learner_name
        #print string.format('Prediction time:', predict)
        #print string.format('Update time:', update)
        #print string.format('Precision:', precision)
        #print string.format('Recall:', recall)
        #print string.format('F_1 Score:', f_1)
        #print string.format('Inv. Precision:', inv_precision)
        #print string.format('Inv. Recall:', inv_recall)
        #print string.format('Inv. F_1 Score:', inv_f_1)
        print string.format('Accuracy:', accuracy)
        #if self.noise_rate is not None:
            #print string.format('Optimal Accuracy:', self.optimal_accuracy)
            #print string.format('Excess error:', self.optimal_accuracy - accuracy)
        
        if self.noise_rate is None:
            return 1.0 - accuracy
        else:
            return 1.0 - accuracy - self.noise_rate

    def return_aggregate_stats(self):
        trials = float(len(self.measures[1]))
        stats = [sum(lst) / trials for lst in self.measures[1:]]
        precision, recall, f_1, inv_precision, inv_recall, inv_f_1, accuracy = stats

        return accuracy


class MistakeTracker(StatTracker):
    
    def display_aggregate_stats(self):
        ns = self.measures[0]
        accuracies = self.measures[7]
        mistake_counts = [n - int(round(n * acc)) for n, acc in zip(ns, accuracies)]
        trials = float(len(mistake_counts))
        average = sum(mistake_counts) / trials
        maximum = max(mistake_counts)
        
        float_string = '{0:<18}{1:0.1f}'
        int_string = '{0:<18}{1:d}'
        print float_string.format('Average mistakes:', average)
        print int_string.format('Maximum mistakes:', maximum)
        
        return maximum


#def test(tracker, learner, dataset, max_iters = 1000):
#    tracker.reset()
#    for i in range(max_iters):
#        try:
#            x, y = dataset.next()
#            
#            start = time()
#            prediction = learner.classify(x)
#            predict_time = time() - start
#            
#            tracker.update(y, prediction)
#            tracker.update_times(predict_time, 0.0)
#        
#        except StopIteration:
#            max_iters = i
#            break
#    
##    print 'test iters:', max_iters
#    if max_iters > 0:
#        tracker.record_performance()
    

def f_measure(precision, recall, beta = 1.0):
    if precision == 0 or recall == 0:
        return 0
    return ((1.0 + beta**2) * precision * recall) / (beta**2 * precision + recall)

















