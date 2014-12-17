from numpy import *
from numpy.linalg import svd
from scipy.stats import norm as normal
from scipy import linalg as lin
import time
import itertools
import random

from learners import *

from cvxopt import matrix, solvers, spdiag
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 2000


class HingeLossSGD(LinearLearner, PassiveSupervisedLearner):
    
    def __init__(self, d, tau, tolerance = 10e-6):
        LinearLearner.__init__(self, d, w = None)
        self.tau = tau
        self.tolerance = tolerance
        gamma_0 = self.tau
        self.rate = lambda t: gamma_0 * (1.0 + 0.1 * t)**(-1.0)
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        def hinge_loss(w, X, Y):
            total = 0.0
            for i in xrange(m):
                total += max(0.0, 1.0 - Y[i] * dot(w, X[i]) / self.tau)
            return total / m
        
        w_star = ones(self.d)
        w_star /= lin.norm(w_star, 2)
        #print
        #print 'w_star:', hinge_loss(w_star, X, Y) 
        
        t = 0
        delta = -1
        index = range(m)
        
        # Pick starting weight vector randomly
        self.w = normal.rvs(size = self.d)
        #self.w = ones(self.d)
        self.w /= lin.norm(self.w, 2)
        end_loss = hinge_loss(self.w, X, Y)
        
        # Proceed until the change in loss is small
        while delta > self.tolerance or delta < 0.0:
            
            start_loss = end_loss
            
            # Randomize the order
            random.shuffle(index)
            
            # Iterate through data once (a single epoch)
            for i in xrange(m):
                
                t += 1
                
                # If the margin is violated, make perceptron-like update
                if Y[index[i]] * dot(self.w, X[index[i]]) < self.tau:
                    self.w += self.rate(t) * Y[index[i]] * X[index[i]] / self.tau
                
                # If norm constraint is violated, normalize w
                norm_w = lin.norm(self.w, 2)
                if norm_w > 1.0:
                    self.w /= norm_w
            
            # Check the change in loss over the epoch
            end_loss = hinge_loss(self.w, X, Y)
            delta = start_loss - end_loss
            #print end_loss, delta, self.rate(t)


class HingeLossSGD2(LinearLearner, PassiveSupervisedLearner):
    
    def __init__(self, d, tau, v, r, tolerance = 0.0001):
        LinearLearner.__init__(self, d, w = None)
        self.tau = tau
        self.v = v
        self.r = r
        self.tolerance = tolerance
        gamma_0 = self.tau
        self.rate = lambda t: gamma_0 * (1.0 + 0.1 * t)**(-1.0)
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        def hinge_loss(w, X, Y):
            total = 0.0
            for i in xrange(m):
                total += max(0.0, 1.0 - Y[i] * dot(w, X[i]) / self.tau)
            return total / m
        
        w_star = ones(self.d)
        w_star /= lin.norm(w_star, 2)
        #print
        #print 'w_star:', hinge_loss(w_star, X, Y) 
        
        t = 0
        delta = -1
        index = range(m)
        
        # Pick starting weight vector randomly
        self.w = normal.rvs(size = self.d)
        #self.w = ones(self.d)
        self.w /= lin.norm(self.w, 2)
        end_loss = hinge_loss(self.w, X, Y)
        
        # Proceed until the change in loss is small
        while delta > self.tolerance or delta < 0.0:
            
            start_loss = end_loss
            
            # Randomize the order
            random.shuffle(index)
            
            # Iterate through data once (a single epoch)
            for i in xrange(m):
                
                t += 1
                
                # If the margin is violated, make perceptron-like update
                if Y[index[i]] * dot(self.w, X[index[i]]) < self.tau:
                    self.w += self.rate(t) * Y[index[i]] * X[index[i]] / self.tau
                
                # If norm constraint is violated, normalize w
                norm_w = lin.norm(self.w, 2)
                if norm_w > 1.0:
                    self.w /= norm_w
                
                # If other constraint is violated, project w
                vw = self.w - self.v
                norm_vw = lin.norm(vw, 2)
                if norm_vw > self.r:
                    self.w = self.v + self.r * vw / norm_vw
            
            # Check the change in loss over the epoch
            end_loss = hinge_loss(self.w, X, Y)
            delta = start_loss - end_loss
            #print end_loss, delta, self.rate(t)


class SVM(LinearLearner, PassiveSupervisedLearner):
    
    def __init__(self, d):
        LinearLearner.__init__(self, d, w = None)
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        # Set up the appropriate matrices and call CVXOPT's quadratic programming
        P = matrix(dot(X, X.T) * dot(Y, Y.T))
        q = matrix(-ones(m))
        G = matrix(-identity(m))
        h = matrix(zeros(m))
        alpha = solvers.qp(P, q, G, h)['x']
        
        # Find the weight vector of the hyperplane from the Lagrange multipliers
        self.w = dot(X.T, alpha * Y)
        self.w = self.w.reshape((self.d,))

class soft_SVM(LinearLearner, PassiveSupervisedLearner):
    
    def __init__(self, d, C):
        LinearLearner.__init__(self, d, w = None)
        self.C = C
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        # Set up the appropriate matrices and call CVXOPT's quadratic programming
        P = matrix(dot(X, X.T) * dot(Y, Y.T))
        q = matrix(-ones(m))
        G = matrix(vstack((-identity(m), identity(m))))
        h = matrix(hstack((zeros(m), self.C * ones(m))))
        alpha = solvers.qp(P, q, G, h)['x']
        
        # Find the weight vector of the hyperplane from the Lagrange multipliers
        self.w = dot(X.T, alpha * Y)
        self.w = self.w.reshape((self.d,))

class LinearProgram(LinearLearner, PassiveSupervisedLearner):

	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)
    
	def batch_train(self, X, Y):
		'''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a linear program.'''
        # Infer the dimension and sample size from the data
		m = len(Y)
		
		# Set up the appropriate matrices and call CVXOPT's linear programming
		c = matrix(sign(normal.rvs(loc = 0, scale = 1.0, size = self.d)))
		G = matrix(vstack([-Y * X, identity(self.d)]))
		h = matrix(vstack([zeros((m, 1)), m**2*ones((self.d, 1))]))
		self.w = solvers.lp(c, G, h)['x']
		self.w = array(self.w).reshape((self.d,))


class Average(LinearLearner, PassiveSupervisedLearner):

	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)
    
	def batch_train(self, X, Y):
	    self.w = (Y * X).sum(axis = 0)
	    self.w /= lin.norm(self.w, 2)

class BandSelection(LinearLearner, PassiveSupervisedLearner):
    def __init__(self, d, num_iters):
        self.num_iters = num_iters
        self.bandparam = 0
        self.radiusparam = 0
        LinearLearner.__init__(self, d, w = None)

    def param_calc(self, X, k, typeof, label):

        sorteddistance = sorted(X)
        #print sorteddistance
        length = len(sorteddistance)
        
        print "Range of distances in kernel space is"
        print sorteddistance[0]
        print sorteddistance[length-1]
        
        '''
        if (k==1 or k==2):
            ratio =1
        else: 
            ratio = (self.num_iters - k + 1) * float(label)/float(length)
        '''
        ratio = 1

        if typeof == "exp":
            frac = pow(2, 1 - k) * ratio
            
        elif typeof == "inv":
            frac = pow(k, -1) * ratio
            
        elif typeof == "lin":
#            frac = (1 - (k - 1) / (self.num_iters - 1.0)) * ratio
            frac = (self.num_iters - k) * float(label)/float(length)
        
        else:
            raise ValueError
        
        print
        print frac
#        self.radiusparam = 1
        self.radiusparam = 2 * frac
        print 'radius:', self.radiusparam
        
        print 'ratio computation:' , float(label)/float(length)
        
        num_points = int(ceil(length * frac))
        print 'points within band:', num_points
        self.bandparam = sorteddistance[num_points - 1]        
#        print 'band:', self.bandparam
        
        
        

class PCA(LinearLearner, PassiveSupervisedLearner):
    
    def __init__(self, d):
        self.var = None
        LinearLearner.__init__(self, d, w = None)
        
    def pca_run(self, X):
        n = mean(X, axis=0)
        X -= n
        Cov = cov(X.T)
        eigenval, eigenvec = lin.eig(Cov)
        idx = argsort(eigenval)[::-1]
        eigenvec = eigenvec[:,idx]
        eigenval = eigenval[idx]
        self.w = eigenvec[0].real
        self.w = self.w.reshape((self.d,))
        return eigenval[0].real, eigenvec[0].real

    def variance_calc(self, X):
        X -= mean(X)
        row = len(X)
        total = 0
        for i in range(row):
            total += pow(X[i],2)
        self.var = total/row

    def train(self, X, Y):
    
		# Getting initial values from PCA	
        val, vec = self.pca_run(X,Y)
        row,col = X.shape
        comp = 10*row*log(row)/col
        
        # Check value of first eigenvalue with (10*number of examples*log(number of examples)/dimensions), and iterating likewise
        while val >= comp:
            # Remove all values from X with greater than the eligible variance		
            for p in range(row):
                print vec,X[p]
                rem = pow(dot(vec,X[p]),2)
                if rem >= (comp/row):
                    #c.remove(ids[p])  # Removal of outliers
                    X = delete(X, p)
                    Y = delete(Y, p)
            # Recalculate the PCA with the new arrays of X and Y				
            val, vec = self.pca_run(X,Y)
            row,col = X.shape
            comp = 10*row*log(row)/col
            
        # Calculate w if first eigenvalue is less than the eligible variance		
        self.w = 1.0/row * dot(X.T,Y)
        self.w = self.w.reshape((self.d,))
        self.var = val
        print self.w, self.var


class soft_SVM_q():
    
    def __init__(self, d, q, C):
        self.d = d
        self.q = q
        self.C = C  # Smaller C makes margin more important
                    # Larger C makes hinge loss more important
        self.w = None
    
    def batch_train(self, X, Y):
		'''
		Given unlabeled training examples (one per row) in matrix X and their
		associated (-1, +1) labels (one per row) in vector Y, returns a weight
		vector w that determines a separating hyperplane, if one exists, using
		a q-norm support vector machine with standard linear kernel.
		'''
		m = len(Y)
	    
	    # First find a feasible solution and create the objective function
		lp = soft_SVM(self.d, self.C)
		lp.batch_train(X, Y)
		s = 1.0 - dot(Y * X, lp.w)
		s[s < 0.0] = 0.0
		x_0 = hstack((lp.w, s))
		F = make_soft_q_svm_primal_objective(self.d, m, self.q, self.C, x_0)
	    
	    # Set up the appropriate matrices and call CVXOPT's convex programming
		G_top = -hstack((Y * X, identity(m)))
		G_bottom = -hstack((zeros((m, self.d)), identity(m)))
		G_fix1 = hstack((identity(self.d), zeros((self.d, m))))
		G_fix2 = -hstack((identity(self.d), zeros((self.d, m))))
		G = matrix(vstack((G_top, G_bottom, G_fix1, G_fix2)))
		h = matrix(hstack((-ones(m), zeros(m), 1e3 * ones(self.d), 1e3 * ones(self.d) )))
	    
	    # Change solver options
		solvers.options['maxiters'] = 100
		solvers.options['abstol'] = 1e-3
		solvers.options['reltol'] = 1e-2
	    
		result = solvers.cp(F, G, h)
	    
	    # Reset solver options to defaults
		solvers.options['maxiters'] = 2000
		solvers.options['abstol'] = 1e-7
		solvers.options['reltol'] = 1e-6
	    
		z = result['x']
		self.w = array(z[:self.d]).reshape((self.d,))
	    
		def classify(self, x):
			return sign(dot(self.w, x))
	    
		def margin(self, x):
			return dot(self.w, x)


def make_soft_q_svm_primal_objective(n, m, q, C, x_0 = None):
    
    if x_0 is None:
        x_0 = r.normal(0, 0.1, n + m)
    
    # Choose normalization constant so objective function values starts at 10.0
    w_0 = x_0[:n]
    s_0 = x_0[n:]
    scale = 1.0 / (sum(abs(w_0)**q) / q + C * sum(s_0))
    
    x_0 = matrix(x_0.reshape((n + m, 1)))
    
    def F(x = None, z = None):
        
        # Case 1
        if x is None and z is None:
            return (0, x_0)
        
        # Case 2 and 3
        else:
            w = x[:n]
            s = x[n:]
            abs_w = abs(w)
            f = scale * (sum(abs_w**q) / q + C * sum(s))
            Df_w = sign(w) * abs_w**(q - 1.0)
            Df_s = C * ones((m, 1))
            Df = scale * vstack((Df_w, Df_s))
            Df = matrix(Df.reshape((1, n + m)))
            
            # Case 2 only
            if z is None:
                return (f, Df)
            
            # Case 3 only
            else:
                try:
                    H_w = scale * z * (q - 1.0) * abs_w**(q - 2.0)
                except (ValueError, RuntimeWarning):
                    #print 'abs_w:', abs_w
                    #print 'power:', (q - 2.0)
                    H_w = scale * z * (q - 1.0) * (abs_w + 1e-20)**(q - 2.0)
                H_s = zeros((m, 1))
                diag_H = matrix(vstack((H_w, H_s)))
                H = spdiag(diag_H)
                return (f, Df, H)
    
    return F

class QP(LinearLearner, PassiveSupervisedLearner):
	
	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)

	def train(self, X, Y, radius, normfac, prevw):

		solvers.options['show_progress'] = False
		
		# Reduce maxiters and tolerance to reasonable levels
		solvers.options['maxiters'] = 200
		solvers.options['abstol'] = 1e-2
		solvers.options['feastol'] = 1e-2

		row, col = X.shape		
		n = row + self.d
		prevw = prevw.reshape((self.d, 1))
		
		
		x_0 = matrix(0.0, (n, 1))
		x_0[:row] = 1.0 - Y * dot(X, prevw) / normfac
		x_0[row:] = prevw
		
		# x represents all the variables in an array, the first ones Ei and then each dimenstion of w, updated to 1/row
		c = matrix(row*[1.0] + self.d*[0.0])  										# the objective function represented as a sum of Ei
		scale_factor = float(dot(c.T, x_0))
		if scale_factor > 0.0:
		    c /= scale_factor
		
		helper = matrix(array(row*[0.0] + self.d*[1.0]).reshape((n, 1)))
		r2 = radius**2
		
		def F(x = None, z = None):
			
			if x is None:
			    return (2, x_0)                                                       # Ei starts from 1 and w starts from 1
			
			w = x[row:]
			diff = w - prevw
			f = matrix(0.0, (2, 1))
			f[0] = dot(diff.T, diff)[0] - r2				# the first non-linear constraint ||w-w[k-1]||^2 < r[k]^2
			f[1] = dot(w.T, w)[0] - 1.0										# the second non-linear constraint ||w||^2 < 1
			
			Df = matrix(0.0, (2, n))										# creating the Df martrix, one row for each non-linear equation with variables as columns
			Df[0, row:] = 2.0 * diff.T     							    # derivative of first non-linear equation, populates a sparse matrix
			Df[1, row:] = 2.0 * w.T											# derivative of second non-linear equation, populates a sparse matrix
			
			if z is None:
			    return f, Df
			
			diag_H = 2.0 * z[0] + 2.0 * z[1] * helper   # Each nonlinear constraint has second derivative 2I w.r.t. w and 0 w.r.t. eps
			H = spdiag(diag_H)
			return f, Df, H

		# for linear inequalities
		G = matrix(0.0, (row*2, n))											# there are two linear constaints for Ei, and for each Ei the entire w
		h = matrix(0.0, (row*2, 1))
		for i in range(row):
			G[i,i] = -1.0															# -Ei <= 0
			G[row+i, i] = -1.0
			h[row+i] = -1.0
			for j in range(self.d):
				G[row+i, row+j] = (-Y[i][0]/normfac)*X[i,j]							# -Ei - yi/Tk(w.xi) <= -1
				
		# solve and return w
		sol = solvers.cpl(c, F, G, h)
		self.w = sol['x'][row:]
		self.w = array(self.w).reshape((self.d,))
		#print
		#print sol['status']
		'''
		print 'Radius wanted'
		print radius
		print 'Output of quadratic solver'
		print self.w
		print ' Norm of output of quadratic solver pre-normalization'
		print sqrt(dot(self.w.T, self.w))
		print ' Distance to the previous weight vector pre-normalization'
		print sqrt(dot((self.w-prevw).T, (self.w-prevw)))
		'''
		self.w = self.w/sqrt(dot(self.w.T,self.w))									# Normalizing the vector output
		'''
		print 'Output of quadratic solver post -norm'
		print self.w
		print ' Norm of output of quadratic solver post-normalization'
		print sqrt(dot(self.w.T, self.w))
		print ' Distance to the previous weight vector post-normalization'
		print sqrt(dot((self.w-prevw).T, (self.w-prevw)))
		'''

class OutlierRemoval(LinearLearner, PassiveSupervisedLearner):
	
	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)
		self.weightdist = None

	def train(self, X, band, radius, normfac, prevw, bound):

		# Set max iterations to 5000
		max_iterations = 2000
		out_itercount = 1

		row, col = X.shape

		# Calculate the variance limit in the data
#		sigma = pow(radius,2) + lin.norm(prevw,2)

		# Set q(x) to 1 for start
		q = ones(row)
		
		# Objective function for q(x)
		def objectiveq(q, sep, X):
			return sum(q * pow(dot(sep, X.T), 2)) / row

		# Constraint on q(x) 
		def constraintq(q, bound):
			# Repeat the following until convergence
			while True:
				
				# Threshold at 0 and 1
				q[q < 0.0] = 0.0
				q[q > 1.0] = 1.0
				
				# Check the total weight
				if sum(q) >= (1.0 - bound) * row - 0.01:
					break
				
				# Scale up the weights, but only increase those less than 1
				else:
					q[q < 1.0] *= 1.0 / sum(q[q < 1.0]) * ((1.0 - bound) * row - sum(q[q == 1.0]))
			return q

		# Starting the outer gradient descent loop for q(x)
		end_obj = inf
		diff = 1
#		print
#		print end_obj
		
		start_outer = time.time()
		while (diff > pow(10,-4) or diff < 0) and out_itercount < max_iterations:
			
			start_obj = end_obj
			
			# Use SVD to maximize over w
			linsep, new_obj = constrained_variance_maximization(X, q, prevw, radius)
			
			# update q
			outer_rate = 0.1
			w_dot_x_2 = pow(dot(linsep, X.T), 2)
			q -= outer_rate * w_dot_x_2 / lin.norm(w_dot_x_2, 2)
			
			# check constraints
			q = constraintq(q, bound)

			#print "the distribution weights"
#			print q
#			print min(q)

			end_obj = objectiveq(q, linsep , X)
#			print end_obj
			diff = start_obj - end_obj
			#print 'Start Obj and End Obj w.r.t to q ' + str(start_obj) + " " + str(end_obj)
			#print('\n')

			out_itercount = out_itercount + 1 
#			print out_itercount

		end_outer = time.time()
		#print " Total time for outer loop run " + str(end_outer - start_outer)
		#print 'Optimal q satisfying all conditions is '
		#print q
		self.weightdist = q
		
		
def constrained_variance_maximization(X, q, u, r):
	# X is n x d
	# q is n x 1
	# u is d x 1
	# r is scalar
	# Returns (w, val) where w maximizes sum_{i=1}^n q[i] * dot(w, x[i])^2 
	# subject to ||w|| = 1 and ||w - u|| <= r,
	# and where val is the value of that maximum.
	
	n, d = X.shape
	q = q.reshape((n, 1))
	u = u.reshape((d, 1))
	
	Xq = sqrt(q) * X
	XqT_Xq = dot(Xq.T, Xq)
	
	# First check if the first principle component satisfies the constraints
	left, diagonal, right = svd(XqT_Xq)
	w1 = right[0].reshape((d, 1))
	val1 = diagonal[0]
	
	if lin.norm(u - w1, 2) <= r or lin.norm(u + w1, 2) <= r:
		return w1.reshape((d,)), val1
	
	# Now project the data
	Xq_proj = Xq - dot(Xq, u) * tile(u.T, (n, 1))
	
	# Find the first principle component of the projected data
	left, diagonal, right = svd(dot(Xq_proj.T, Xq_proj))
	v = right[0].reshape((d, 1))
	
	# This should be close to zero
#	assert abs(dot(u.T, v)) <= 0.01
	
	# Construct the vector and the value in the original space
	c1 = (1.0 + dot(u.T, u) - r**2) / 2.0
	c2 = sqrt(1.0 - c1**2)
	w = c1 * u + c2 * v
	val = dot(dot(w.T, XqT_Xq), w)[0, 0]
	
	# Check the result
#	print
#	print dot(dot(u.T, XqT_Xq), u)[0, 0]
#	print val
#	print val1
#	print lin.norm(w, 2)
#	print lin.norm(u - w, 2), r
#	assert dot(dot(u.T, XqT_Xq), u) <= val <= val1
#	assert 0.99 <= lin.norm(w, 2) <= 1.01
#	assert lin.norm(u - w, 2) <= r + 0.01
	
	return w.reshape((d,)), val

'''
class QPwithoutBandConstraint(LinearLearner, PassiveSupervisedLearner):
	
	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)

	def train(self, X, Y, radius, normfac, prevw):

		#Have commented out all the equations relating to the band constraint from this solver
		
		solvers.options['show_progress'] = True
		solvers.options['maxiters'] = 10000

		row, col = X.shape		
		# x represents all the variables in an array, the first ones Ei and then each dimenstion of w, updated to 1/row
		c = matrix(row*[1.0] + self.d*[0.0])  										# the objective function represented as a sum of Ei
		
		def F(x=None, z=None):
			if x is None:  return 1, matrix(row*[1.0] + self.d*[1.0])				# Ei starts from 1 and w starts from 1
			f = matrix(0.0, (1,1))
			#f[0] = sqrt(dot((x[row:].T-prevw),(x[row:].T-prevw).T))-radius				# the first non-linear constraint ||w-w[k-1]||^2 < r[k]
			f[0] = sqrt(dot(x[row:].T,x[row:])) -1										# the second non-linear constraint ||w||^2 <1
			Df = matrix(0.0, (1,row+self.d))										# creating the Df martrix, one row for each non-linear equation with variables as columns
			#Df[0,row:] = 2.0 * (x[row:].T-prevw[:])								    # derivative of first non-linear equation, populates a sparse matrix
			Df[0,row:] = 2.0 * x[row:].T											# derivative of second non-linear equation, populates a sparse matrix
			if z is None: return f, Df
			secder = matrix(row*[0.0] + self.d*[2.0])
			H = matrix(0.0, (row+self.d, row+self.d))
			for i in range(self.d):
				H[row+i,row+i] = z[0]*secder[row+i]			# returns the second derivative, a sparse matrix
			return f, Df, H

		# for linear inequalities
		G = matrix(0.0,(row*2, row+self.d))											# there are two linear constaints for Ei, and for each Ei the entire w
		h = matrix(0.0, (row*2, 1))
		for i in range(row):
			G[i,i] = -1.0															# -Ei <= 0
			G[row+i, i] = -1.0
			h[row+i] = -1.0
			for j in range(self.d):
				G[row+i, row+j] = (-Y[i][0]/normfac)*X[i,j]							# -Ei - yi/Tk(w.xi) <= -1
				
		# solve and return w
		sol = solvers.cpl(c, F, G, h)
		self.w = sol['x'][row:]
		self.w = array(self.w).reshape((self.d,))
		print sol
		
		print 'Radius wanted'
		print radius
		print 'Output of quadratic solver'
		print self.w
		print ' Norm of output of quadratic solver pre-normalization'
		print sqrt(dot(self.w.T, self.w))
		print ' Distance to the previous weight vector pre-normalization'
		print sqrt(dot((self.w-prevw).T, (self.w-prevw)))
		
		self.w = self.w/sqrt(dot(self.w.T,self.w))									# Normalizing the vector output
		
		print 'Output of quadratic solver post -norm'
		print self.w
		print ' Norm of output of quadratic solver post-normalization'
		print sqrt(dot(self.w.T, self.w))
		print ' Distance to the previous weight vector post-normalization'
		print sqrt(dot((self.w-prevw).T, (self.w-prevw)))	


class QPwithoutNormConstraint(LinearLearner, PassiveSupervisedLearner):
	
	def __init__(self, d):
		LinearLearner.__init__(self, d, w = None)

	def train(self, X, Y, radius, normfac, prevw):

		#Have commented out all the equations relating to the norm constraint on W from this solver
		
		#solvers.options['show_progress'] = True
		#solvers.options['maxiters'] = 10000

		row, col = X.shape		
		# x represents all the variables in an array, the first ones Ei and then each dimenstion of w, updated to 1/row
		c = matrix(row*[1.0] + self.d*[0.0])  										# the objective function represented as a sum of Ei
		
		def F(x=None, z=None):
			if x is None:  return 1, matrix(row*[1.0] + self.d*[1.0])				# Ei starts from 1 and w starts from 1
			f = matrix(0.0, (1,1))
			f[0] = sqrt(dot((x[row:].T-prevw),(x[row:].T-prevw).T))-radius				# the first non-linear constraint ||w-w[k-1]||^2 < r[k]
			#f[0] = sqrt(dot(x[row:].T,x[row:])) -1										# the second non-linear constraint ||w||^2 <1
			Df = matrix(0.0, (1,row+self.d))										# creating the Df martrix, one row for each non-linear equation with variables as columns
			Df[0,row:] = 2.0 * (x[row:].T-prevw[:])								    # derivative of first non-linear equation, populates a sparse matrix
			#Df[0,row:] = 2.0 * x[row:].T											# derivative of second non-linear equation, populates a sparse matrix
			if z is None: return f, Df
			secder = matrix(row*[0.0] + self.d*[2.0])
			H = matrix(0.0, (row+self.d, row+self.d))
			for i in range(self.d):
				H[row+i,row+i] = z[0]*secder[row+i]			# returns the second derivative, a sparse matrix
			return f, Df, H

		# for linear inequalities
		G = matrix(0.0,(row*2, row+self.d))											# there are two linear constaints for Ei, and for each Ei the entire w
		h = matrix(0.0, (row*2, 1))
		for i in range(row):
			G[i,i] = -1.0															# -Ei <= 0
			G[row+i, i] = -1.0
			h[row+i] = -1.0
			for j in range(self.d):
				G[row+i, row+j] = (-Y[i][0]/normfac)*X[i,j]							# -Ei - yi/Tk(w.xi) <= -1
				
		# solve and return w
		sol = solvers.cpl(c, F, G, h)
		self.w = sol['x'][row:]
		#print self.w														
		self.w = array(self.w).reshape((self.d,))
		self.w = self.w/sqrt(dot(self.w.T,self.w))									# Normalizing the vector output
		#print sol
'''

#####################################################################################################################################

class KernelSVM(KernelLearner):
    
    def __init__(self, d, kernel):
        KernelLearner.__init__(self, d, kernel)
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        K = zeros((m,m))
        
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i],X[j])
        
        # Set up the appropriate matrices and call CVXOPT's quadratic programming
        P = matrix(K * dot(Y, Y.T))
        q = matrix(-ones(m))
        G = matrix(-identity(m))
        h = matrix(zeros(m))
        alpha = solvers.qp(P, q, G, h)['x']
        
        #storing the required values in the KernelLearner.support variable
        for i in range(m):
            temp = alpha[i] * Y[i]
            self.support.append([temp, X[i]])

class Kernel_soft_SVM(KernelLearner):
    
    def __init__(self, d, C, kernel):
        KernelLearner.__init__(self, d, kernel)
        self.C = C
    
    def batch_train(self, X, Y):
        '''Given unlabeled training examples (one per row) in matrix X and their
        associated (-1, +1) labels (one per row) in vector Y, returns a weight
        vector w that determines a separating hyperplane, if one exists, using
        a support vector machine with standard linear kernel.'''
        # Infer the sample size from the data
        m = len(Y)
        
        K = zeros((m,m))
        
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i],X[j])
        
        # Set up the appropriate matrices and call CVXOPT's quadratic programming
        P = matrix(K * dot(Y, Y.T))
        q = matrix(-ones(m))
        G = matrix(vstack((-identity(m), identity(m))))
        h = matrix(hstack((zeros(m), self.C * ones(m))))
        alpha = solvers.qp(P, q, G, h)['x']
        
        #storing the required values in the KernelLearner.support variable
        for i in range(m):
			temp = alpha[i] * Y[i]
			self.support.append([temp, X[i]])


class KernelQP(KernelLearner, PassiveSupervisedLearner):
	
	def __init__(self, d, kernel):
		KernelLearner.__init__(self, d, kernel)

	def train(self, X, Y, normfac):

		solvers.options['show_progress'] = False
		
		# Reduce maxiters and tolerance to reasonable levels
		solvers.options['maxiters'] = 200
		solvers.options['abstol'] = 1e-2
		solvers.options['feastol'] = 1e-2

		row, col = X.shape
		
		P = matrix(0.0, (row,row))
		
		# Calculating the Kernel Matrix
		for i in range(row):
			for j in range(row):
				P[i,j] = Y[i] * self.kernel(X[i],X[j]) * Y[j]           # It's a PSD matrix, so its okay !

		# A point in the solution space for objective
		x_0 = matrix(0.5, (row, 1))
		
		normarr = matrix(normfac, (1,row))
		
		def F(x = None, z = None):
			
			if x is None:
			    return (0, x_0)                                         # Alpha's start from 0.5, first value is zero as there are zero non-linear objectives
			
			term = matrix(sqrt(x.T * P * x))
			
			f = matrix(term - normfac * sum(x))                         # return the objective function
			
			# first derivative
			Df = (x.T * P)/term - normarr 						        # since for each alpha, normfac will be subtracted, norm arr is an array
			
			if z is None:
			    return f, Df
			
			term2 = matrix((P*x) * (P*x).T)
			H = z[0] * (P/term - term2/pow(term,3))                     # Second derivative of the objective function, is a symmetric matrix, so no need for spDiag ?
			
			return f, Df, H

		# for linear inequalities
		G = matrix(0.0, (row*2, row))									# there are two linear constaints for Alpha
		h = matrix(0.0, (row*2, 1))
		for i in range(row):
			G[i,i] = -1.0												# -Alpha <= 0
			G[row+i, i] = 1.0                                           #  Alpha <= 1
			h[row+i] = 1.0
				

		# solve and return w
		sol = solvers.cp(F, G, h)
		alpha = sol['x']
		
		for i in range(row):
			self.support.append([alpha[i] * Y[i], X[i]])
			
		#print
		#print sol

class KernelQPwithLinearBand(KernelLearner, PassiveSupervisedLearner):
	
	def __init__(self, d, kernel):
		KernelLearner.__init__(self, d, kernel)

	def train(self, X, Y, normfac, radius, prevw):
        # the weight vector w is kept as a tuple - alpha_i * y_i and x_i, send only the required number of rows
		solvers.options['show_progress'] = False
		
		# Reduce maxiters and tolerance to reasonable levels
		solvers.options['maxiters'] = 2000
		solvers.options['abstol'] = 1e-2
		solvers.options['feastol'] = 1e-2

		row, col = X.shape
		
		P = matrix(0.0, (row+1,row+1))
		
		# Calculating the Kernel Matrix
		# Kernel matrix will now include multiple kernel matrices
		for i in range(row):
			for j in range(row):
				P[i,j] = Y[i] * self.kernel(X[i],X[j]) * Y[j]       # It's a PSD matrix, so its okay !
		
		# Summing over the kernel values between current set of points and prevw
		for i in range(row):
			P[i,row] = normfac * Y[i] * sum( prevw[k][0] * self.kernel(prevw[k][1], X[i]) for k in range(len(prevw)) )
			P[row,i] = P[i,row]

		# summing over the kernels value of the entire prevw matrix
		P[row, row] = pow(normfac,2) * sum( prevw[k][0] * self.kernel(prevw[k][1], prevw[r][1]) * prevw[r][0] for k,r in itertools.product(range(len(prevw)), range(len(prevw))) )

		# A point in the solution space for objective
		x_0 = matrix(0.5, (row+1, 1))
		
		normarr = matrix([normfac]*row + [normfac*(1-pow(radius,2)/2)]).T
		
		def F(x = None, z = None):
			
			if x is None:
			    return (0, x_0)                                         # Alpha's start from 0.5, first value is zero as there are zero non-linear objectives
			
			term = matrix(sqrt(x.T * P * x))
			
			f = matrix(term - normfac * sum(x[0:row]) - x[row] * normfac * (1-pow(radius,2)/2))                         # return the objective function
			
			# first derivative
			Df = (x.T * P)/term - normarr 						        # since for each alpha, normfac will be subtracted, norm arr is an array
			
			#print "Rank of Df"
			#print linalg.matrix_rank(Df)
			#print Df.size
			#print "Rank of f"
			#print linalg.matrix_rank(f)
			
			if z is None:
			    return f, Df
			
			term2 = matrix((P*x) * (P*x).T)
			H = z[0] * (P/term - term2/pow(term,3))                     # Second derivative of the objective function, is a symmetric matrix, so no need for spDiag ?
			
			#print "Rank of hessian"
			#print linalg.matrix_rank((P/term - term2/pow(term,3)))
			#print "Size of hessian"
			#print H.size
			
			return f, Df, H

		# for linear inequalities
		G = matrix(0.0, (row*2 + 1, row +1))									# there are two linear constaints for Alpha, one for Beta
		h = matrix(0.0, (row*2 +1, 1))
		for i in range(row):
			G[i,i] = -1.0       										# -Alpha <= 0
			G[row+i, i] = 1.0                                           #  Alpha <= 1
			h[row+i] = 1.0
		G[row*2, row] = -1.0                                            # -Beta <= 0

		#print "Rank of G"
		#print linalg.matrix_rank(G)
		#print "Rank of hessian"
		#print linalg.matrix_rank(h)
		

		# solve and return w
		sol = solvers.cp(F, G, h)
		
		#print sol
		
		alpha = sol['x'][0:row]
		beta = sol['x'][row]
		
		row_prev = len(prevw)
		templist = []
		
		for i in range(row):
			templist.append([alpha[i] * Y[i], X[i]])
		
		# Add Beta * Tau_k to the previous support vectors and store in current support vectors
		for i in range(row_prev):
			templist.append([prevw[i][0] * beta * normfac, prevw[i][1]])
		
		self.support = templist
		
		#print
		#print sol['x']

class StochasticDualCoordinateAscent(KernelLearner, ActiveBatchLearner):
	
	def __init__(self, d, C, kernel):
		KernelLearner.__init__(self, d, kernel)
		self.C = C
	
	def train(self, X, Y):
		
		row, col = X.shape
		
		alpha = zeros((row,1))
		w = sum( Y[i]*X[i]*alpha[i] for i in range(row))
		
		iter_local = 200
		
		for k in range(iter_local):
		
			i = random.randint(0, row-1)
			
			G = Y[i] * sum( alpha[j] * Y[j] * self.kernel(X[j], X[i]) for j in range(row)) -1
			
			if alpha[i] == 0:
				PG = min(0, G)
			elif alpha[i] == self.C:
				PG = max(0, G)
			else:
				PG = G
			
			kernel_temp = self.kernel(X[i], X[i])
			
			if PG != 0:
				alpha_temp = alpha[i]
				alpha[i] = min(max(alpha[i] - G/kernel_temp, 0), self.C)
				w = w + (alpha[i] - alpha_temp) * Y[i] * X[i]
	
		for i in range(row):
			self.support.append([Y[i]*alpha[i], X[i]])


	
