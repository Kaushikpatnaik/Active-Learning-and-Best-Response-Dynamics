from numpy import *
from scipy.stats import norm as normal
import random


class Adversary(object):
    
    def __init__(self):
        self.total = 0
        self.noisy = 0
    
    def corrupt(self, x, y):
        raise NotImplementedError
    
    def noise_rate(self):
        if self.total == 0:
            return 0.0
        else:
            return self.noisy / float(self.total)


class RandomClassificationNoise(Adversary):
    
    def __init__(self, alpha):
        self.alpha = alpha
        Adversary.__init__(self)
    
    def corrupt(self, x, y):
        self.total += 1
        if random.random() < self.alpha:
            self.noisy += 1
            return (x, -y)
        else:
            return (x, y)


class ConditionalRandomClassificationNoise(Adversary):
    
    def __init__(self, alpha, condition = lambda: True):
        self.alpha = alpha
        self.condition = condition
        Adversary.__init__(self)
        
    
    def corrupt(self, x, y):
        self.total += 1
        #print y, self.condition(x,y)
        if random.random() < self.alpha and self.condition(x, y):
            self.noisy += 1
            return (x, -y)
        else:
            return (x, y)


class MarginLinearLabelNoise(Adversary):
    
    def __init__(self, w_star, a, b):
        '''Noise rate is linear in |w dot x| with probability of flipping equal 
        to a at the decision boundary and equal to b far from the boundary.'''
        self.w_star = w_star
        self.intercept = a
        self.slope = b - a
        Adversary.__init__(self)
    
    def corrupt(self, x, y):
        self.total += 1
        if random.random() < self.slope * pow(abs(dot(self.w_star, x)),-1) + self.intercept:
            self.noisy += 1
            return (x, -y)
        else:
            return (x, y)


class ConditionalMarginLinearLabelNoise(Adversary):

    def __init__(self, w_star, a, b, condition = lambda: True):
        '''Noise rate is linear in |w dot x| with probability of flipping equal
        to a at the decision boundary and equal to b far from the boundary.
        Only labels in the majority label are flipped '''
        self.w_star = w_star
        self.intercept = a
        self.slope = b - a
        self.condition = condition
        Adversary.__init__(self)

    def corrupt(self, x, y):
        self.total += 1
        if random.random() < self.slope * pow(abs(dot(self.w_star, x)),-1) + self.intercept and self.condition(x,y):
            self.noisy += 1
            return (x, -y)
        else:
            return (x, y)

class MarginLinearMaliciousNoise(Adversary):
    
    def __init__(self, w_star, a, b):
        '''Noise rate is linear in |w dot x| with probability of flipping equal 
        to a at the decision boundary and equal to b far from the boundary.'''
        self.w_star = w_star
        self.intercept = a
        self.slope = b - a
        Adversary.__init__(self)
        
        # Create a vector 45 degrees from w_star
#        vec = normal.rvs(size = len(self.w_star))
#        vec -= dot(self.w_star, vec)
#        vec /= sqrt(dot(vec, vec))
#        self.confuser = 0.5 * self.w_star + 0.5 * vec
        
    
    def corrupt(self, x, y):
        self.total += 1
        margin = dot(self.w_star, x)
        if random.random() < self.slope * abs(margin) + self.intercept:
            self.noisy += 1
            return (x + 5.0 * margin * self.w_star, -y)
        else:
            return (x, y)