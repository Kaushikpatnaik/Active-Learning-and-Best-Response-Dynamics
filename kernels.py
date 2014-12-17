from numpy import *
from numpy import linalg as lin

# Kernels should be implemented so that k(x, y) can be called with y as a 
# matrix (n x d) with one point per row. This should return the vector of 
# kernel evaluations between x and each point in y. I think this will often
# be taken care of automatically by using dot as in the polynomial kernel below.

class LinearKernel:

    def __init__(self, a = 1.0, c = 1.0):
        self.a = a
        self.c = c

    def __call__(self, x, y):
        # Row vector x
        # Row vector y (or matrix with points in rows)
        return (self.c + self.a * dot(x, y.T))


class PolynomialKernel:

    def __init__(self, d, a = 1.0, c = 1.0):
        self.d = d
        self.a = a
        self.c = c

    def __call__(self, x, y):
        # Row vector x
        # Row vector y (or matrix with points in rows)
        return (self.c + self.a * dot(x, y.T))**self.d


class GaussianKernel:

    def __init__(self, sigma = 5.0):
        self.sigma = sigma

    def __call__(self, x, y):
        # Row vector x
        # Row vector y (or matrix with points in rows)
        # return the kernel value
        
        return exp(- lin.norm(x-y)**2/ (2 * (self.sigma**2)))


if __name__ == '__main__':

    k = PolynomialKernel(2)

    x = array([1.0, 4.0, 3.0])
    y = array([2.0, 3.0, 4.0])
    z = array([4.0, -1.0, -2.0])


    print
    print k(x, y)
    print k(y, x)
    assert abs(k(x, y) - k(y, x)) < 1e-10

    print
    print k(x, z)
    print k(y, z)
    
    print

