from numpy import *
from scipy.stats import norm as normal

from learners import *



class QuasiAdditive(LinearLearner, OnlineLearner):
    
    def __init__(self, d, transform, z = None, rate = 1.0):
        self.d = d
        self.transform = transform
        if z == None:
            self.z = normal.rvs(loc = 0, scale = 0.1, size = d)
        else:
            self.z = z
        self.w = self.transform(self.z)
        self.rate = rate
    
    def update(self, x, y):
        if y != sign(dot(self.w, x)):
            self.z += self.rate * y * x
            self.w = self.transform(self.z)


class Perceptron(LinearLearner, OnlineLearner):
    
    def __init__(self, d, w = None, rate = 1.0):
        self.d = d
        if w == None:
            normal.rvs(loc = 0, scale = 0.1, size = d)
        else:
            self.w = w
        self.rate = rate
    
    def update(self, x, y):
        if y != sign(dot(self.w, x)):
            self.w += self.rate * y * x
















